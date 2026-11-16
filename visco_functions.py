import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv, eigs, splu, LinearOperator

def A2(s, m, N, beta0, Ek):
    ds = (s[-1] - s[0]) / (N - 1)

    #matrice derivee quatrieme
    ui4 = np.full(N-2, 6 / ds**4)
    ui4[0]= 7 / ds**4
    ui4[-1]= 7 / ds**4
    uiv4 = np.full(N-3, -4 / ds**4)
    uivv4 = np.full(N-4, 1 / ds**4)  
    D4 = sp.diags([uivv4,uiv4,ui4,uiv4,uivv4], offsets=[-2,-1,0,1,2], format = 'csc') * 1j*Ek
        

    #matrice derivee troisieme
    uiv3 = np.full(N-3, 2 / (2*ds**3))
    uivv3 = np.full(N-4, -1 / (2*ds**3))
    ui3 = np.zeros(N-2)
    ui3[0]  = -1 / (2*ds**3)  
    ui3[-1] =  1 / (2*ds**3) 
    D3 = sp.diags([uivv3,uiv3,ui3,-uiv3,-uivv3],offsets=[-2,-1,0,1,2], format = 'csc')      
    fact3 = sp.diags([(2 /s[1:-1] + beta0*s[1:-1]) * 1j*Ek], offsets=[0], format = 'csc')
    D3 = fact3 @ D3
    

    #matrice derivee seconde
    ui2 = np.full(N-2, -2 / ds**2)
    uiv2 = np.full(N-3, 1 / ds**2)
    D2 = sp.diags([uiv2,ui2,uiv2],offsets=[-1,0,1], format = 'csc')  
    fact2 = sp.diags([(5*beta0 - (2*m**2+1)/s[1:-1]**2) * 1j*Ek], offsets=[0], format='csc')
    D2 = fact2 @ D2    

    
    #matrice derivee premiere 
    uiv1 = np.full(N-3, 1 / (2*ds))
    D1 = sp.diags([-uiv1,uiv1], offsets = [-1,1], format = 'csc')
    fact1 = sp.diags([(beta0/s[1:-1] * (3-m**2) + (2*m**2+1)/s[1:-1]**3) * 1j*Ek],offsets=[0], format='csc')
    D1 = fact1 @ D1
    

    #matrice non derivee
    ui = (1j*Ek * (-2*beta0*(m**2)/s[1:-1]**2 + (m**4-4*m**2)/s[1:-1]**4) + 2/s[1:-1]*beta0*s[1:-1]*m)
    D = sp.diags([ui],offsets=[0], format = 'csc')

    A = D4 + D3 + D2 + D1 + D

    return A


def B2(s, m, N): 
    # Version de B avec beta nul dans le lhs 
    ds = (s[-1] - s[0]) / (N - 1)

    ui2 = np.full(N - 2, -2) / ds**2
    uiv2 = np.ones(N - 3) / ds**2
    D2 = sp.diags([uiv2, ui2, uiv2], offsets=[-1, 0, 1])  # Matrice derivée seconde

    uiv1 = np.ones(N - 3) / (2 * ds)  # Diag voisines du point consideré donc de 2 a n-2
    D1 = sp.diags([-uiv1, uiv1], offsets=[-1, 1])  # Matrice derivée première
    facteur = sp.diags([1 / s[1:-1]], offsets=[0], format='csc')
    D1 = facteur @ D1

    ui = -m**2 / s[1:-1]**2
    D = sp.diags([ui], offsets=[0], format='csc')  # Matrice derivée 0 

    B = D2 + D1 + D  # Matrice B à utiliser pour solve le problème (lhs)

    return B


#construction de la matrice A en rajoutant tous les termes visqueux 
def A(beta, dbeta, ddbeta, dddbeta, m, s, N, Ek):
    ds = (s[-1] - s[0]) / (N - 1)

    #matrice derivee quatrieme
    ui4 = np.full(N-2, 6 / ds**4)
    ui4[0]= 7 / ds**4
    ui4[-1]= 7 / ds**4
    uiv4 = np.full(N-3, -4 / ds**4)
    uivv4 = np.full(N-4, 1 / ds**4)  
    D4 = sp.diags([uivv4,uiv4,ui4,uiv4,uivv4], offsets=[-2,-1,0,1,2], format = 'csc') * 1j*Ek
        

    #matrice derivee troisieme
    uiv3 = np.full(N-3, 2 / (2*ds**3))
    uivv3 = np.full(N-4, -1 / (2*ds**3))
    ui3 = np.zeros(N-2)
    ui3[0]  = -1 / (2*ds**3)  
    ui3[-1] =  1 / (2*ds**3) 
    D3 = sp.diags([uivv3,uiv3,ui3,-uiv3,-uivv3],offsets=[-2,-1,0,1,2], format = 'csc')   
    fact3 = sp.diags([(2 /s[1:-1] + beta) * 1j*Ek], offsets=[0], format = 'csc')
    D3 = fact3 @ D3
    

    #matrice derivee seconde
    ui2 = np.full(N-2, -2 / ds**2)
    uiv2 = np.full(N-3, 1 / ds**2)
    D2 = sp.diags([uiv2,ui2,uiv2],offsets=[-1,0,1], format = 'csc')    
    fact2 = sp.diags([(-1 /s[1:-1]**2 * (1+2*m**2) + 2*beta/s[1:-1] + 3*dbeta) * 1j*Ek], offsets=[0], format = 'csc')
    D2 = fact2 @ D2    

    
    #matrice derivee premiere 
    uiv1 = np.full(N-3, 1 / (2*ds))
    D1 = sp.diags([-uiv1,uiv1], offsets = [-1,1], format = 'csc')
    fact1 = sp.diags([(1/s[1:-1]**3 * (1+2*m**2) + 3*ddbeta - beta/s[1:-1]**2*(m**2+1) + 4*dbeta/s[1:-1]) * 1j*Ek], offsets=[0], format = 'csc')
    D1 = fact1 @ D1
    

    #matrice non derivee
    ui = 1j*Ek * (1/s[1:-1]**4*(m**4-4*m**2) + beta/s[1:-1]**3*(1-m**2) - dbeta/s[1:-1]**2*(m**2+1) + 2*ddbeta/s[1:-1] + dddbeta) + 2/s[1:-1]*beta*m
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

def beta0(s_i, s_o): 
    # Quand beta est lineaire, dbeta_ds=beta0 
    # Calcul au rayon s moyen
    s_mean = (s_i + s_o) / 2
    beta0 = (-(s_o**2 + s_mean**2)) / (s_o**2 - s_mean**2)**2
    return beta0
    
    
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
    ddbeta = -2*s/(s_o**2-s**2)**2 - 4*s*(s_o**2+s**2)/(s_o**2-s**2)**3
    return ddbeta

def dddbeta(s, s_o): 
    s = s[1:-1]
    dddbeta = -(2/(s_o**2-s**2)**2 + (20*s**2+4*s_o**2)/(s_o**2-s**2)**3 + 24*s**2*(s_o**2+s**2)/(s_o**2-s**2)**4)
    return dddbeta 

def eigvalues_reg(A, B, N, n):
    M = inv(B) @ A
    k = N - 4
    eigvals, eigvecs_raw = eigs(M, k=k, which='LI')

    idx_sorted = np.argsort(eigvals.imag)[::-1]  
    eigvals = eigvals[idx_sorted[:n]]
    eigvecs_wobc = eigvecs_raw[:, idx_sorted[:n]]

    eigvecs = np.zeros((N, n), dtype=complex)
    eigvecs[1:-1, :] = eigvecs_wobc

    return eigvals, eigvecs

'''
def eigvalues_shift_invert(A, B, f_df, n, N, k=1): #f_df = array d'eigenvalues obtenues sans amortissement 
    eigvals = np.zeros(n, dtype=complex) #5 items
    eigvecs = np.zeros((N, n), dtype=complex) #500 lignes et 5 colonnes 
    M = inv(B) @ A
    if type(f_df)==np.ndarray:
        for i, f in enumerate(f_df):
            val, vec = eigs(M, k=k, sigma=f, which='LM') 
            eigvals[i] = val[0]
            eigvecs[1:-1, i] = vec[:,0]
    elif type(f_df)==complex:
        sigma=f_df
        eigvals, eigvecs = eigs(M, k=k, sigma=sigma, which='LM')
    return eigvals, eigvecs
'''

def eigvalues_shift_invert(A, B, f_df,n, N, k=1):
    M = np.linalg.solve(B.toarray(), A.toarray())
    eigvals = np.zeros(n, dtype=complex)
    eigvecs = np.zeros((N, n), dtype=complex)
    l = 0 #correction que je set à 0 de base 
    for i, f in enumerate(f_df):
        vpc = f + l       #valeur propre complexe    
        val, vec = eigs(M, k=k, sigma=vpc, which='LM')       
        eigvals[i] = val[0]
        eigvecs[1:-1, i] = vec[:, 0]
        l = eigvals[i] - f       
    return eigvals, eigvecs
