import matplotlib.pyplot as plt
import numpy as np
import visco_functions as f

s_i = 4  # rayon interne couche fluide
s_o = 5  # rayon externe couche fluide
N = 500  # resolution
s = np.linspace(s_i, s_o, N)  # definition grille
Ek = 10**(-6) #nombre d'Ekman
f_df = [0.0407,0.0138,0.0070,0.0042,0.0028] #les 5 premiers modes obtenus sans dissipation (m=1)

m_array = np.arange(1, 51, 1)
n = 5  # nombre onde radial (nombre eigenvectors/values)
m=1
v0 = np.ones(N-2, dtype=complex)
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

plt.show()
