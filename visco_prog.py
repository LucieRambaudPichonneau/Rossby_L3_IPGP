import matplotlib.pyplot as plt
import numpy as np
import visco_functions as f

s_i = 4  # rayon interne couche fluide
s_o = 5  # rayon externe couche fluide
N = 500  # resolution
s = np.linspace(s_i, s_o, N)  # definition grille
Ek = 10**(-6) #nombre d'Ekman

m_array = np.arange(1, 51, 1)
n = 5  # nombre onde radial (nombre eigenvectors/values)

'''
fig, ax = plt.subplots(1,1,figsize = (10, 7))
psi_test = np.zeros(len(s))
for i,sl in enumerate(s):
    psi = (sl-s_i)**2*(sl-s_o)**2
    psi_test[i]=psi

ax.plot(s,psi_test)
plt.show()
'''

fig, ax2 = plt.subplots(1, 1, figsize=(10, 7))
y = np.zeros((len(m_array), n))
for i, m in enumerate(m_array):
    A = f.A(f.beta(s, s_o), f.dbeta(s, s_o), f.ddbeta(s,s_o), f.dddbeta(s, s_o), m, s, N, Ek)  # sans l'approx sur beta
    B = f.B(f.beta(s, s_o), f.dbeta(s, s_o), m, s, N)
    eigenvalues, eigenvectors = f.eigvalues_reg(A, B, N, n)
    y[i, :] = np.abs(eigenvalues.real)
    
for k in range(1, n + 1):
    ax2.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$'.format(k))
    
ax2.set_title(r'Dispertion relation (spherical $\beta$)')
ax2.set_yscale('log')
ax2.set_xlabel(r'Azimuthal wave number $m$')
ax2.set_ylabel(r'Angular frequency f')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
#fig.savefig("comparaison_beta_non_lineaire_visco.pdf", bbox_inches='tight')

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
    A = f.A(f.beta(s, s_o), f.dbeta(s, s_o), f.ddbeta(s,s_o), f.dddbeta(s, s_o), m_val, s, N, Ek)
    B = f.B(f.beta(s, s_o), f.dbeta(s, s_o), m_val, s, N)
    eigenvalues, eigenvectors = f.eigvalues_reg(A, B, N, n)
    psi = np.real(np.outer(eigenvectors[:, 0], np.exp(1j * m_val * phi))) 
    
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
#fig.savefig("donuts_spheriques_visco.png", dpi=300, bbox_inches='tight')

plt.show()
