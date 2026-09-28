import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv, eigs


#construction de la matrice A en rajoutant tous les termes visqueux 
def A(beta, dbeta, ddbeta, dddbeta, m, s, N, Ek):
    ds = (s[-1] - s[0]) / (N - 1)

    #matrice derivee quatrieme
    ui4 = np.full(N-2, 6 / ds**4)
    uiv4 = np.full(N-3, -4 / ds**4)
    uivv4 = np.full(N-4, 1 / ds**4)  
    D4 = sp.diags([uivv4,uiv4,ui4,uiv4,uivv4], offsets=[-2,-1,0,1,2], format = 'csc')

    #bc D4 (bord interne)
    bci4 = np.zeros(N-2)
    bci4[0] = 7 / ds**4
    bci4[1] = -4 / ds**4
    bci4[2] = 1 / ds**4
    bci4 = sp.csc_array(bci4.reshape(1,-1))
    
    #bc D4 (bord externe)
    bce4 = np.zeros(N-2)
    bce4[-1] = 7 / ds**4
    bce4[-2] = -4 / ds**4
    bce4[-3] = 1 / ds**4
    bce4 = sp.csc_array(bce4.reshape(1,-1))

    D4 = sp.vstack([bci4, D4[1:-1,:], bce4], format = 'csc') * 1j*Ek
        

    #matrice derivee troisieme
    uiv3 = np.full(N-3, 2 / (2*ds**3))
    uivv3 = np.full(N-4, -1 / (2*ds**3))
    D3 = sp.diags([uivv3,uiv3,-uiv3,-uivv3],offsets=[-2,-1,1,2], format = 'csc')

    #bc D3
    bci3 = np.zeros(N-2)
    bci3[0] = -1 / (2*ds**3)
    bci3[1] = -2 / (2*ds**3)
    bci3[2] = 1 / (2*ds**3)
    bci3 = sp.csc_array(bci3.reshape(1,-1))
    
    #bc D3 (bord externe)
    bce3= np.zeros(N-2)
    bce3[-1] = 1 / (2*ds**3)
    bce3[-2] = 2 / (2*ds**3)
    bce3[-3] = -1 / (2*ds**3)
    bce3 = sp.csc_array(bce3.reshape(1,-1))

    D3 = sp.vstack([bci3, D3[1:-1,:], bce3], format = 'csc')        
    fact3 = sp.diags([(2 /s[1:-1] + beta) * 1j*Ek], offsets=[0], format = 'csc')
    D3 = fact3 @ D3
    

    #matrice derivee seconde
    ui2 = np.full(N-2, -2 / ds**2)
    uiv2 = np.full(N-3, 1 / ds**2)
    D2 = sp.diags([uiv2,ui2,uiv2],offsets=[-1,0,1], format = 'csc')

    #bc D2 int
    bci2 = np.zeros(N-2)
    bci2[0] = -2 / ds**2
    bci2[1] = 1 / ds**2
    bci2 = sp.csc_array(bci2.reshape(1,-1))
    
    #bc D2 ext
    bce2 = np.zeros(N-2)
    bce2[-1] = -2 / ds**2
    bce2[-2] = 1 / ds**2
    bce2 = sp.csc_array(bce2.reshape(1,-1))

    D2 = sp.vstack([bci2, D2[1:-1,:], bce2], format = 'csc')      
    fact2 = sp.diags([(1 /s[1:-1]**2 - 2*m**2/s[1:-1]**2 + 2*beta/s[1:-1] + dbeta) * 1j*Ek], offsets=[0], format = 'csc')
    D2 = fact2 @ D2    

    
    #matrice derivee premiere 
    uiv1 = np.full(N-3, 1 / (2*ds))
    D1 = sp.diags([-uiv1,uiv1], offsets = [-1,1], format = 'csc')

    #bc D1 int
    bci1 = np.zeros(N-2)
    bci1[1] = 1 / (2*ds)
    bci1 = sp.csc_array(bci1.reshape(1,-1))
    
    #bc D1 ext
    bce1 = np.zeros(N-2)
    bce1[-2] = -1 / (2*ds)
    bce1 = sp.csc_array(bce1.reshape(1,-1))

    D1 = sp.vstack([bci1, D1[1:-1,:], bce1], format = 'csc')
    fact1 = sp.diags([(-2*m**2/s[1:-1]**3  + ddbeta + 1/s[1:-1]*dbeta - beta*m**2/s[1:-1]**2 + beta/s[1:-1]**2 + dbeta/s[1:-1]) * 1j*Ek], offsets=[0], format = 'csc')
    D1 = fact1 @ D1
    

    #matrice non derivee
    ui = 1j*Ek * (m**4/s[1:-1]**4 - beta*m**2/s[1:-1]**3 + 2*ddbeta/s[1:-1] + dbeta/s[1:-1]**2*(1-m**2) + dddbeta) + 2/s[1:-1]*beta*m
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

def eigvalues_shift_invert(A, B, f_df, n, N, v0, k=2): #f_df = array d'eigenvalues obtenues sans amortissement 
    eigvals = np.zeros(n, dtype=complex) #5 items
    eigvecs = np.zeros((N, n), dtype=complex) #500 lignes et 5 colonnes 
    
    for i, f in enumerate(f_df):
        vals, vecs = eigs(A, k=k, M=B, sigma=f, which='LM') 
        idx_max = np.argmax(np.imag(vals)) 
        eigvals[i] = vals[idx_max]
        eigvecs[1:-1, i] = vecs[:, idx_max]
        
    return eigvals, eigvecs
