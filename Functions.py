import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv, eigs
from scipy.special import jv, yn
from scipy.optimize import root
from scipy.integrate import trapezoid

#Définition des fonctions

def beta0(s_i, s_o): 
    # Quand beta est lineaire, dbeta_ds=beta0 
    # Calcul au rayon s moyen
    s_mean = (s_i + s_o) / 2
    beta0 = (-(s_o**2 + s_mean**2)) / (s_o**2 - s_mean**2)**2
    return beta0
    
def beta(s, s_o):
    s = s[1:-1]  # Suppression des bords pour éviter d'avoir des soucis de définition de beta 
    beta = -s / (s_o**2 - s**2)  # Obtenu avec la def de h(s) cf. rapport et la def de beta de Schaeffer
    return beta

def dbeta_ds(s, s_o):
    s = s[1:-1]
    dbeta_ds = (-(s_o**2 + s**2)) / (s_o**2 - s**2)**2
    return dbeta_ds

def AB3(s, m, N, beta): 
    # Version avec beta nul dans le lhs et lineaire dans le rhs
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

    diag_a = 2 / s[1:-1] * beta * m
    A = sp.diags([diag_a], offsets=[0], format='csc')  # Matrice A à utiliser pour solve le problème (rhs)

    return A, B

def AB4(s, m, N, beta, dbeta_ds): 
    # Version sans approximation sur beta et sa derivée
    ds = (s[-1] - s[0]) / (N - 1)
    
    ui2 = np.full(N - 2, -2) / ds**2
    uiv2 = np.ones(N - 3) / ds**2
    D2 = sp.diags([uiv2, ui2, uiv2], offsets=[-1, 0, 1], format='csc')  # Matrice derivée seconde

    uiv1 = np.ones(N - 3) / (2 * ds)  # Diag decrit les voisins du point consideré donc de 2 a n-2
    D1 = sp.diags([-uiv1, uiv1], offsets=[-1, 1], format='csc')  # Matrice derivée première
    facteur = sp.diags([1 / s[1:-1] + beta], offsets=[0], format='csc')
    D1 = facteur @ D1
        
    ui = beta / s[1:-1] + dbeta_ds - m**2 / s[1:-1]**2
    D = sp.diags([ui], offsets=[0], format='csc')  # Matrice derivée 0 
    
    B = D2 + D1 + D  # Matrice B à utiliser pour solve le problème (lhs)
    
    diag_a = 2 / s[1:-1] * beta * m
    A = sp.diags([diag_a], offsets=[0], format='csc')  # Matrice A à utiliser pour solve le problème (rhs)
            
    return A, B

def bessel_eq(alpha, s_i, s_o, m):
    # Equation donnée par les conditions aux limites 
    return yn(m, alpha * s_o) * jv(m, alpha * s_i) - jv(m, alpha * s_o) * yn(m, alpha * s_i) 

def bessel_solve(s_i, s_o, beta0, m_array, n): 
    # Retourne les pulsations de bessels 
    H = s_o - s_i
    puls_bessel = np.empty((len(m_array), n))
    oldalphas = None
    
    for i, m in enumerate(m_array):
        alphas = np.empty(n)
        for j in range(n):
            k = j + 1
            if i == 0:
                x0 = k * np.pi / H  # guess initial k*pi modulé par l'inverse de l'epaisseur de l'océan
            else:
                x0 = oldalphas[j]
            sol = root(bessel_eq, x0=x0, args=(s_i, s_o, m))
            alphas[k - 1] = sol.x[0]
            
        alphas = np.sort(alphas)
        oldalphas = alphas.copy()
        puls_bessel[i, :] = (2 * m * beta0) / alphas**2
        
    return puls_bessel

def eigenvect_analytic(s_i, beta0, m, n, puls_bessel_m, s):
    alphas = np.sqrt(np.abs(2 * beta0 * m / puls_bessel_m))
    eigenvect_bessel_norm = np.empty((len(s), n))
    
    for i in range(n):
        gamma = -jv(m, alphas[i] * s_i) / yn(m, alphas[i] * s_i)
        psi_non_norm = jv(m, alphas[i] * s) + gamma * yn(m, alphas[i] * s)
        eigenvect_bessel_norm[:, i] = psi_non_norm / (np.max(np.abs(psi_non_norm)))
        
    return eigenvect_bessel_norm 

def eigvalues_reg(A, B, N, n): 
    # Résolution du problème regulier 
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

def analytical_zeros(nroots, s_i, s_o, m):
    lbda = s_o / s_i
    mu = 4 * m**2
    alpha = (np.arange(nroots) + 1) * np.pi / (lbda - 1.)
    p = (mu - 1.) / (8 * lbda)

    roots_zeroth_order = alpha / s_i
    roots_first_order = (alpha + p / alpha) / s_i

    return roots_zeroth_order, roots_first_order


def error(m, n, s_i, s_o, beta0, puls_bessel, powers=range(3, 12)):
    N_array = 2**np.array(powers)
    eps = np.empty(len(N_array))
    
    puls_bessel_err = puls_bessel[m - 1, :n] 
    
    for i, N_err in enumerate(N_array):
        s_err = np.linspace(s_i, s_o, N_err)
        
        A_err, B_err = AB3(s_err, m, N_err, beta=beta0 * s_err[1:-1])
        eigenvalues_err, eigenvectors_err = eigvalues_reg(A_err, B_err, N_err, n=n)
        
        psi_num_err = eigenvectors_err[:, n - 1].flatten()

        psi_an_err_mtx = eigenvect_analytic(s_i, beta0, m, n=n, puls_bessel_m=puls_bessel_err, s=s_err)
        psi_an_err = psi_an_err_mtx[:, n - 1].flatten()

        prod_scal_cyl = trapezoid(psi_an_err * psi_num_err * s_err, x=s_err)
        if prod_scal_cyl < 0:
            psi_num_err = -psi_num_err
                
        norm_num = np.sqrt(trapezoid(psi_num_err**2 * s_err, x=s_err))
        norm_an = np.sqrt(trapezoid(psi_an_err**2 * s_err, x=s_err))

        if norm_num != 0: 
            psi_num_err /= norm_num
        if norm_an != 0:  
            psi_an_err /= norm_an

        err = (psi_an_err - psi_num_err)**2 * s_err
        eps[i] = np.sqrt(trapezoid(err, x=s_err))    
        
    slope = np.polyfit(np.log10(N_array), np.log10(eps), 1)[0]
    return N_array, eps, slope


def error2(m, n, s_i, s_o, beta0, puls_bessel, powers=range(3, 11)): 
    N_array_raw = 2**np.array(powers)
    
    # On va stocker les résultats uniquement pour les résolutions valides
    valid_N = []
    eps_list = []
    
    puls_bessel_err = puls_bessel[m - 1, :n] 
    
    for N_err in N_array_raw:
        s_err = np.linspace(s_i, s_o, N_err)
        
        # Si la grille est trop petite pour contenir le n-ième mode physique, passage à la résolution suivante sans calculer (il faut N_err - 2 > n)
        if (N_err - 2) < n:
            continue
            
        # Résolution numérique
        A_err, B_err = AB3(s_err, m, N_err, beta=beta0 * s_err[1:-1])
        eigenvalues_err, eigenvectors_err = eigvalues_reg(A_err, B_err, N_err, n=n)
        
        if eigenvectors_err.shape[1] < n:
            continue
            
        psi_num_err = eigenvectors_err[:, n - 1].flatten()

        psi_an_err_mtx = eigenvect_analytic(s_i, beta0, m, n=n, puls_bessel_m=puls_bessel_err, s=s_err)
        psi_an_err = psi_an_err_mtx[:, n - 1].flatten()

        produit_scalaire = trapezoid(psi_an_err * psi_num_err * s_err, x=s_err)
        if produit_scalaire < 0:
            psi_num_err = -psi_num_err
                
        norm_num = np.sqrt(trapezoid(psi_num_err**2 * s_err, x=s_err))
        norm_an = np.sqrt(trapezoid(psi_an_err**2 * s_err, x=s_err))

        if norm_num != 0: 
            psi_num_err /= norm_num
        if norm_an != 0:  
            psi_an_err /= norm_an

        err = (psi_an_err - psi_num_err)**2 * s_err
        
        valid_N.append(N_err)
        eps_list.append(np.sqrt(trapezoid(err, x=s_err)))
        
    N_array = np.array(valid_N)
    eps = np.array(eps_list)
    
    slope = np.polyfit(np.log10(N_array), np.log10(eps), 1)[0]
    return N_array, eps, slope
