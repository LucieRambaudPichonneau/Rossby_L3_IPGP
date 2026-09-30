import matplotlib.pyplot as plt
import numpy as np
import visco_functions as f2
import Functions as f

s_i = 4  # rayon interne couche fluide
s_o = 5  # rayon externe couche fluide
N = 500  # resolution
s = np.linspace(s_i, s_o, N)  # definition grille
Ek = 10**(-4) #nombre d'Ekman

m_array = np.arange(1, 51, 1)
n = 5  # nombre onde radial (nombre eigenvectors/values)
v0 = np.ones(N-2, dtype=complex)

beta0 = f.beta0(s_i, s_o)

#beta lineaire, f(s)=pulsation avec et sans visco, pour n=1,...,5 (partie reelle)
fig, ax = plt.subplots(1, 1, figsize=(13, 7))
y = np.empty((len(m_array), n))
for i, m in enumerate(m_array):#sans visco
    A, B = f.AB3(s, m, N, beta=beta0 * s[1:-1])  # approx sur beta 
    eigenvalues, eigenvectors = f.eigvalues_reg(A, B, N, n)
    y[i, :] = np.abs(eigenvalues.real)

y2 = np.empty((len(m_array),n))
for i, m in enumerate(m_array): #avec visco_functions
    A = f2.A2(s, m, N, beta0, Ek)
    B = f2.B2(s, m, N)
    f_df = f.f_wo_visco(m,n,s,N,beta=beta0,geom='lin')
    eigenvalues2, eigenvectors2 = f2.eigvalues_shift_invert(A, B,f_df,n, N, v0, k=1)
    y2[i, :] = np.abs(eigenvalues2.real)

    
for k in range(1, n + 1):
    lin = ax.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$ sans visco'.format(k))
    color = lin.get_facecolor()[0]
    ax.scatter(m_array, abs(y2[:, k-1]), facecolors='none', color=color, label=r'$n={}$ Ek=10E-4'.format(k))
    
ax.set_title(r'Dispertion relation (linear $\beta$)')
ax.set_yscale('log')
ax.set_xlabel(r'Azimuthal wave number $m$')
ax.set_ylabel(r'Angular frequency f')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
fig.savefig("comparaison_beta_lineaire_avec-sans_visco.pdf", bbox_inches='tight')

#beta complet, f(s)=pulsation avec et sans visco, pour n=1,...,5 (partie reelle)
fig, ax2 = plt.subplots(1, 1, figsize=(10, 7))
y = np.zeros((len(m_array), n))
for i, m in enumerate(m_array):
    A, B = f.AB4(s, m, N, f.beta(s, s_o), f.dbeta_ds(s, s_o))  # sans l'approx sur beta
    eigenvalues, eigenvectors = f.eigvalues_reg(A, B, N, n)
    y[i, :] = np.abs(eigenvalues.real)

y2 = np.empty((len(m_array),n))
for i, m in enumerate(m_array): #avec visco_functions
    A = f2.A(f2.beta(s, s_o), f2.dbeta(s, s_o), f2.ddbeta(s,s_o), f2.dddbeta(s, s_o), m, s, N, Ek)
    B = f2.B(f2.beta(s, s_o), f2.dbeta(s, s_o), m, s, N)
    f_df = f.f_wo_visco(m,n,s,N,beta=f2.beta(s, s_o),geom='sph', dbeta = f2.beta(s, s_o))
    eigenvalues2, eigenvectors2 = f2.eigvalues_shift_invert(A, B,f_df,n, N, v0, k=1)
    y2[i, :] = np.abs(eigenvalues2.real)

for k in range(1, n + 1):
    sph = ax2.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$ sans visco'.format(k))
    color = sph.get_facecolor()[0]
    ax2.scatter(m_array, abs(y2[:, k-1]), facecolors='none', color=color, label=r'$n={}$ Ek=10E-4'.format(k))
    
ax2.set_title(r'Dispertion relation (spherical $\beta$)')
ax2.set_yscale('log')
ax2.set_xlabel(r'Azimuthal wave number $m$')
ax2.set_ylabel(r'Angular frequency f')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
fig.savefig("comparaison_beta_non_lineaire_avec-sans_visco.pdf", bbox_inches='tight')


#beta lineaire, f(s)=pulsation avec et sans visco, pour n=1,...,5 (partie imaginaire)
fig, ax = plt.subplots(1, 1, figsize=(13, 7))
y = np.empty((len(m_array),n))
for i, m in enumerate(m_array): #avec visco_functions
    A = f2.A2(s, m, N, beta0, Ek)
    B = f2.B2(s, m, N)
    f_df = f.f_wo_visco(m,n,s,N,beta=beta0,geom='lin')
    eigenvalues2, eigenvectors2 = f2.eigvalues_shift_invert(A, B,f_df,n, N, v0, k=1)
    y[i, :] = np.abs(eigenvalues2.imag)

for k in range(1, n + 1):
    ax.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$ sans visco'.format(k))
    
ax.set_title(r'Dispertion relation (linear $\beta$), imaginary part')
ax.set_yscale('log')
ax.set_xlabel(r'Azimuthal wave number $m$')
ax.set_ylabel(r'Angular frequency f')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
fig.savefig("beta_lineaire_avec_visco_Im.pdf", bbox_inches='tight')


#beta spherique, f(s)=pulsation avec et sans visco, pour n=1,...,5 (partie imaginaire)
fig, ax = plt.subplots(1, 1, figsize=(13, 7))
y = np.empty((len(m_array),n))
for i, m in enumerate(m_array): #avec visco_functions
    A = f2.A(f2.beta(s, s_o), f2.dbeta(s, s_o), f2.ddbeta(s,s_o), f2.dddbeta(s, s_o), m, s, N, Ek)
    B = f2.B(f2.beta(s, s_o), f2.dbeta(s, s_o), m, s, N)
    f_df = f.f_wo_visco(m,n,s,N,beta=f2.beta(s, s_o),geom='sph', dbeta = f2.beta(s, s_o))
    eigenvalues2, eigenvectors2 = f2.eigvalues_shift_invert(A, B,f_df,n, N, v0, k=1)
    y[i, :] = np.abs(eigenvalues2.imag)

for k in range(1, n + 1):
    ax.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$ sans visco'.format(k))
    
ax.set_title(r'Dispertion relation (spherical $\beta$), imaginary part')
ax.set_yscale('log')
ax.set_xlabel(r'Azimuthal wave number $m$')
ax.set_ylabel(r'Angular frequency f')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
fig.savefig("beta_spherique_avec_visco_Im.pdf", bbox_inches='tight')


'''
fig, ax = plt.subplots(1, 1, figsize=(10,7))
A = f.A(f.beta(s, s_o), f.dbeta(s, s_o), f.ddbeta(s,s_o), f.dddbeta(s, s_o), m, s, N, Ek)  # sans l'approx sur beta
B = f.B(f.beta(s, s_o), f.dbeta(s, s_o), m, s, N)
eigenvalues, eigenvectors = f.eigvalues_shift_invert(A, B,f_df,n, N, v0, k=2)

for k in range(1, n + 1):
    ax.plot(s,eigenvectors[:, k-1].real,label=r'$n={}$ numerical'.format(k))
ax.legend()

# Plot 2D pour m=1,10,20,30 et n=1 --> solution sphérique
N_phi = 512
phi = np.linspace(0, 2 * np.pi, N_phi)
s_grid, phi_grid = np.meshgrid(s, phi, indexing='ij')
x_grid = s_grid * np.cos(phi_grid)
y_grid = s_grid * np.sin(phi_grid)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'2D Eigenmodes (spherical $\beta$)', fontsize=18)
axflat = ax.flatten()

for i, m_val in enumerate(np.array([1, 10, 20, 30])):
    A = f.A(f.beta(s, s_o), f.dbeta(s, s_o), f.ddbeta(s,s_o), f.dddbeta(s, s_o), m_val, s, N, Ek)  # sans l'approx sur beta
    B = f.B(f.beta(s, s_o), f.dbeta(s, s_o), m_val, s, N)
    eigenvalues, eigenvectors = f.eigvalues_shift_invert(A, B,f_df,n, N,v0, k=2)
    psi =np.real(np.outer(eigenvectors[:, 0], np.exp(1j * m_val * phi))) 
    
    axflat[i].contourf(x_grid, y_grid, psi, levels=15, cmap='RdBu_r')
    axflat[i].set_title(r'$m={}$'.format(m_val))
    
    cercle_s_i = plt.Circle((0, 0), s_i, color='black', linestyle='-', fill=False)
    cercle_s_o = plt.Circle((0, 0), s_o, color='black', linestyle='-', fill=False)
    axflat[i].add_patch(cercle_s_i)
    axflat[i].add_patch(cercle_s_o)
    axflat[i].set_aspect('equal')
    axflat[i].set_xticks([])
    axflat[i].set_yticks([])

fig.tight_layout()
fig.savefig("donuts_spheriques.png", dpi=300, bbox_inches='tight')
'''
plt.show()
