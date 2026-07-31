import matplotlib.pyplot as plt
import numpy as np
import functions as f

plt.rcParams.update({
    'font.size': 18,          
    'axes.titlesize': 20,     
    'axes.labelsize': 18,     
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,    
    'legend.fontsize': 16     
})

s_i = 4  # rayon interne couche fluide
s_o = 5  # rayon externe couche fluide
N = 500  # resolution
s = np.linspace(s_i, s_o, N)  # definition grille

m_array = np.arange(1, 51, 1)
n = 5  # nombre onde radial (nombre eigenvectors/values)

beta0 = f.beta0(s_i, s_o)

# Bessels (solution analytique) vs. solution numerique avec beta lineaire 
puls_bessel = f.bessel_solve(s_i, s_o, beta0, m_array, n)

fig, ax = plt.subplots(1, 1, figsize=(13, 7))
y = np.empty((len(m_array), n))
for i, m in enumerate(m_array):
    A, B = f.AB3(s, m, N, beta=beta0 * s[1:-1])  # approx sur beta 
    eigenvalues, eigenvectors = f.eigvalues_reg(A, B, N, n)
    y[i, :] = np.abs(eigenvalues.real)
    
for k in range(1, n + 1):
    num = ax.scatter(m_array, y[:, k-1], marker='x', label=r'$n={}$ numerical'.format(k))
    color = num.get_facecolor()[0]
    ax.scatter(m_array, abs(puls_bessel[:, k-1]), facecolors='none', color=color, label=r'$n={}$ analytical'.format(k))
    
ax.set_title(r'Dispertion relation (linear $\beta$)')
ax.set_yscale('log')
ax.set_xlabel(r'Azimuthal wave number $m$')
ax.set_ylabel(r'Angular frequency f')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig.tight_layout()
fig.savefig("comparaison_analytique_beta_lineaire.pdf", bbox_inches='tight')


# Figure solution numerique sans approx sur beta
fig, ax2 = plt.subplots(1, 1, figsize=(10, 7))
y = np.zeros((len(m_array), n))
for i, m in enumerate(m_array):
    A, B = f.AB4(s, m, N, f.beta(s, s_o), f.dbeta_ds(s, s_o))  # sans l'approx sur beta
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
fig.savefig("comparaison_beta_non_lineaire.pdf", bbox_inches='tight')


# Erreur entre sol analytique et sol numerique pour deux couples
N_array11, eps11, slope11 = f.error(m=1, n=1, s_i=s_i, s_o=s_o, beta0=beta0, puls_bessel=puls_bessel)
N_array530, eps530, slope530 = f.error2(m=30, n=5, s_i=s_i, s_o=s_o, beta0=beta0, puls_bessel=puls_bessel)

label11 = rf"(m,n)=(1,1) : $\epsilon \propto N^{{{slope11:.2f}}}$"
label530 = rf"(m,n)=(30,5) : $\epsilon \propto N^{{{slope530:.2f}}}$"

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.loglog(N_array11, eps11, marker='o', linestyle='-', label=label11)
ax.loglog(N_array530, eps530, marker='o', linestyle='-', label=label530)
ax.set_xlabel('Resolution N')
ax.set_ylabel(r'Error $\epsilon$')
ax.set_title(r'Analytical verus numerical solution (linear $\beta$)')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend()

fig.tight_layout()
fig.savefig("erreur_convergence.pdf", bbox_inches='tight')


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
    A, B = f.AB4(s, m_val, N, f.beta(s, s_o), f.dbeta_ds(s, s_o))
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
fig.savefig("donuts_spheriques.png", dpi=300, bbox_inches='tight')


# Plot 2D pour m=1,2 et n=1 --> solution lineaire 
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(r'2D Eigenmodes (linear $\beta$)', fontsize=18)
axflat = ax.flatten()

for i, m_val in enumerate(np.array([1, 10, 20, 30])):
    A, B = f.AB3(s, m_val, N, beta0 * s[1:-1])
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
fig.savefig("donuts_lineaires.png", dpi=300, bbox_inches='tight')


# Comparaison omega vs n pour m=1 et m=30 avec mcmahon
n_array = np.arange(1, 51)

weak_m = 1
large_m = 30

Aw, Bw = f.AB3(s, weak_m, N, beta=s[1:-1] * beta0)
Al, Bl = f.AB3(s, large_m, N, beta=s[1:-1] * beta0)

eigenval_w = np.sort(np.abs(f.eigvalues_reg(Aw, Bw, N, n=50)[0]))[::-1]
eigenval_l = np.sort(np.abs(f.eigvalues_reg(Al, Bl, N, n=50)[0]))[::-1]

alpha1_w, alpha2_w = f.analytical_zeros(50, s_i, s_o, weak_m)
alpha1_l, alpha2_l = f.analytical_zeros(50, s_i, s_o, large_m)

# Conversion des alpha en omega
asympt1_w = np.abs((2 * beta0 * weak_m) / alpha1_w**2)
asympt2_w = np.abs((2 * beta0 * weak_m) / alpha2_w**2)
asympt1_l = np.abs((2 * beta0 * large_m) / alpha1_l**2)
asympt2_l = np.abs((2 * beta0 * large_m) / alpha2_l**2)

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

ax[0].plot(n_array, asympt1_w, label=r'$0^{th}$ order approximation', linewidth=1)
ax[0].plot(n_array, asympt2_w, label=r'$1^{st}$ order approximation', linewidth=0.9, linestyle='--')
ax[0].scatter(n_array, eigenval_w, label='Numerical solution', linewidths=0.5, marker='x', color='red')
ax[0].set_xlabel('Radial mode n')
ax[0].set_ylabel('Angular frequency f')
ax[0].set_title(r'$m=1$')
ax[0].set_yscale('log')
ax[0].grid(True, which="both", linestyle='--', alpha=0.5)
ax[0].legend()

ax[1].plot(n_array, asympt1_l, label=r'$0^{th}$ order approximation', linewidth=1)
ax[1].plot(n_array, asympt2_l, label=r'$1^{st}$ order approximation', linewidth=0.9, linestyle='--')
ax[1].scatter(n_array, eigenval_l, label='Numerical solution', linewidths=0.5, marker='x', color='red')
ax[1].set_xlabel('Radial mode n')
ax[1].set_ylabel(r'Angular frequency f')
ax[1].set_title(r'$m=30$')
ax[1].set_yscale('log')
ax[1].grid(True, which="both", linestyle='--', alpha=0.5)
ax[1].legend()

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle('Dispersion relation: McMahon asymptotic expansion', fontsize=18)
fig.savefig("comparaison_asymptotique.pdf", bbox_inches='tight')


# Plot Beta
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(s[1:-1], f.beta(s, s_o), label=r'Spherical shell $\beta$')
ax.plot(s[1:-1], beta0 * s[1:-1], label=r'Linear $\beta$')
ax.set_xlabel('s')
ax.set_ylabel(r'$\beta$')
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend()

fig.savefig('beta.pdf', bbox_inches='tight')
fig.suptitle(r"$\beta$(s)")

plt.show()
