import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv, eigs


#construction de la matrice A en rajoutant tous les termes visqueux 
def A(beta,dbeta,ddbeta,dddbeta,m,s,N,Ek):
    ds = (s[-1] - s[0]) / (N - 1)

    #matrice derivee quatrieme
    ui4 = np.full(N-2,-6*Ek / ds**4)
    uiv4 = np.full(N-3, 4*Ek / ds**4)
    uivv4 = np.full(N-4, -Ek / ds**4)  
    D4 = sp.diags([uivv4,uiv4,ui4,uiv4,uivv4], offsets=[-2,-1,0,1,2], format = 'csc')

    #matrice derivee troisieme
    uiv3 = np.ones(N-3, -2*Ek / 2*ds**3)
    uivv3 = np.ones(N-4, Ek / 2*ds**3)
    D3 = sp.diags([uivv3,uiv3,-uiv3,-uivv3],offsets=[-2,-1,1,2], format = 'csc')
    fact3 = sp.diags([2 /s[1:-1]+beta], offsets=[0], format = 'csc')
    D3 = fact3 @ D3 

    #matrice derivee seconde
    ui2 = np.full(N-2, 2*Ek / ds**2)
    uiv2 = np.full(N-3, -Ek / ds**2)
    D2 = sp.diags([uiv2,ui2,uiv2], offsets = [-1,0,1],format = 'csc')
    fact2 = sp.diags([(1-2*m**2)/s[1:-1]**2 + 2*beta/s[1:-1] + dbeta], offsets = [0], format = 'csc')
    D2 = fact2 @ D2

    #matrice derivee premiere 
    uiv1 = np.full(N-3, -Ek / 2*ds)
    D1 = sp.diags([uiv1,-uiv1], offsets = [-1,1], format = 'csc')
    fact1 = sp.diags([ddbeta + 2*dbeta/s[1:-1] - beta*m**2/s[1:-1]**2 + beta/s[1:-1]**2],offsets = [0], format = 'csc')
    D1 = fact1 @ D1

    #matrice non derivee
    ui = -Ek* (m**4/s[1:-1]**4 -(2+beta)*m**2/s[1:-1]**3 + dbeta/s[1:-1]**2*(1-m**2 + 2*ddbeta/s[1:-1] + dddbeta)) + 2/s[1:-1]*beta*m
    D = sp.diags([ui],offsets=[0], format = 'csc')

    A = D4 + D3 + D2 + D1 + D

    return A

#matrice B reprise dans mon code precedent
def B(beta, dbeta, m,s, N): 
    ds = (s[-1] - s[0]) / (N - 1)
    
    ui2 = np.full(N - 2, -2) / ds**2
    uiv2 = np.ones(N - 3) / ds**2
    D2 = sp.diags([uiv2, ui2, uiv2], offsets=[-1, 0, 1], format='csc')  # Matrice derivée seconde

    uiv1 = np.ones(N - 3) / (2 * ds)  # Diag decrit les voisins du point consideré donc de 2 a n-2
    D1 = sp.diags([-uiv1, uiv1], offsets=[-1, 1], format='csc')  # Matrice derivée première
    facteur = sp.diags([1 / s[1:-1] + beta], offsets=[0], format='csc')
    D1 = facteur @ D1
        
    ui = beta / s[1:-1] + dbeta - m**2 / s[1:-1]**2
    D = sp.diags([ui], offsets=[0], format='csc')  # Matrice derivée 0 
    
    B = D2 + D1 + D  # Matrice B à utiliser pour solve le problème (lhs)

    return B

    
#definition de beta et ses derivees
def beta(s, s_o): #repris de mon code d'avant
    s = s[1:-1]  # Suppression des bords pour éviter d'avoir des soucis de définition de beta 
    beta = -s / (s_o**2 - s**2)  # Obtenu avec la def de h(s) cf. rapport et la def de beta de Schaeffer
    return beta

def dbeta(s, s_o): #repris de mon code d'avant
    s = s[1:-1]
    dbeta = (-(s_o**2 + s**2)) / (s_o**2 - s**2)**2
    return dbeta

def ddbeta(s, s_o): 
    s = s[1:-1]
    ddbeta = 2/(s_o**2-s**2) - 4*s*(s_o**4-s**4)/(s_o**2-s**2)**4
    return ddbeta

def dddbeta(s, s_o): 
    s = s[1:-1]
    dddbeta = 4/(s_o**2-s**2)**4 * (s*(s_o**2-s**2)-(s_o**4-s**4)) - 4*s/(s_o**2-s**2)**8 * (-4*s**3 * (s_o**2-s**2)**4 + 8*s*(s_o**4-s**4)*(s_o**2-s**2))
    return dddbeta 

 #resolution du system repris de mon code d'avant   (regulier)
def eigvalues_reg(A, B, N, n):  
    M = inv(B) @ A
    k = min(n + 10, N - 4)
    eigvals, eigvecs_raw = eigs(M, k=k, which='LM')  # wobc = without boundary conditions
    
    real_vals = np.real(eigvals)
    n_safe = min(n, len(real_vals))
    
    idx_sorted = np.argsort(np.abs(real_vals))[::-1]
    eigvals = real_vals[idx_sorted[:n_safe]]
    eigvecs_wobc = np.real(eigvecs_raw[:, idx_sorted[:n_safe]])

    eigvecs = np.zeros((N, n_safe))
    eigvecs[1:-1, :] = np.real(eigvecs_wobc)
    
    return eigvals, eigvecs
