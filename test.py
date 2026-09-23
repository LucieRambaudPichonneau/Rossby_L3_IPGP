import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv, eigs
from scipy.sparse.linalg import spsolve

def A(s,N):
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

    D4 = sp.vstack([bci4, D4[1:-1,:], bce4], format = 'csc')
        

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
    fact3 = sp.diags([2 /s[1:-1]], offsets=[0], format = 'csc')
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
    fact2 = sp.diags([-1 /s[1:-1]**2], offsets=[0], format = 'csc')
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
    fact1 = sp.diags([1 /s[1:-1]**3], offsets=[0], format = 'csc')
    D1 = fact1 @ D1
    

    A = D4 + D3 + D2 + D1
    
    return A
    

s_i = 4  # rayon interne couche fluide
s_o = 5  # rayon externe couche fluide
N = 500  # resolution
s = np.linspace(s_i, s_o, N)  # definition grille

m_array = np.arange(1, 51, 1)
n = 5  # nombre onde radial (nombre eigenvectors/values)

A = A(s, N)


F_test = np.empty(len(s))
for i, e in enumerate(s):
    Fe = 24 + 24/e*(2*e-s_i-s_o) - 2/e**2*((e-s_o)**2+(e-s_i)**2+4*(e-s_i)*(e-s_o)) + 2/e**3*(e-s_i)*(e-s_o)*(2*e-s_o-s_i)
    F_test[i] = Fe
psi_test_num = spsolve(A,F_test[1:-1])

psi_test_num = np.concatenate(([0],psi_test_num,[0]))

psi_test_an = np.empty(len(s))
for i, e in enumerate(s):
    psie = (e-s_o)**2 * (e-s_i)**2
    psi_test_an[i] = psie
    

fig, ax = plt.subplots(1, 1, figsize = (13,7))
ax.plot(s, psi_test_num, color='g', linewidth=4, label = 'num')
ax.plot(s, psi_test_an, linestyle='dashed', color='y', linewidth = 2, label = 'an')
ax.legend()

plt.show()
