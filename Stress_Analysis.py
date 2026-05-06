import numpy as np 
 
points_before = np.array([ 
    [5, 0, 20],     
    [30, 0, 20],    
    [30, 10, 20],   
    [0, 10, 20],    
    [0, 0, 0],      
    [30, 0, 0],     
    [30, 10, 0],    
    [0, 10, 0],     
    [30, 5, 10]     
], dtype=float) 
 
points_after = np.array([ 
    [5.003, 0.002, 19.004], 
    [30.003, 0.000, 19.995], 
    [29.995, 10.003, 19.996], 
    [0.000, 10.009, 19.995], 
    [0.000, 0.200, 0.000], 
    [30.004, 0.002, 0.004], 
    [29.996, 9.995, 0.300], 
    [0.0011, 9.995, 0.000], 
    [29.987, 5.005, 9.986] 
], dtype=float) 
 
displacements = points_after - points_before  
 
 
n = points_before.shape[0] 
X = np.hstack([np.ones((n, 1)), points_before])   
A_fit = np.linalg.lstsq(X, displacements, rcond=None)[0]   
 
grad_u = A_fit[1:, :].T   
 
 
strain = 0.5 * (grad_u + grad_u.T) 
 
 
def hooke_stress(E, nu, eps): 
    """Return stress tensor from strain tensor for isotropic 
material.""" 
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu)) 
    mu = E / (2 * (1 + nu)) 
    eps_vol = np.trace(eps) 
    return lam * eps_vol * np.eye(3) + 2 * mu * eps 
 
# Aluminum 
E_Al, nu_Al, sigY_Al = 69000.0, 0.33, 276.0 
sigma_Al = hooke_stress(E_Al, nu_Al, strain) 
 
# Steel 
E_St, nu_St, sigY_St = 210000.0, 0.30, 250.0 
sigma_St = hooke_stress(E_St, nu_St, strain) 
 
 
 
def stress_measures(sig): 
    eigvals, eigvecs = np.linalg.eigh(sig)   
     
    s_dev = sig - np.trace(sig)/3*np.eye(3) 
    von_mises = np.sqrt(1.5 * np.sum(s_dev**2)) 
 
    max_shear = 0.5 * (np.max(eigvals) - np.min(eigvals)) 
     
    sigma_oct = np.trace(sig) / 3 
    tau_oct = np.sqrt(2/3 * np.sum(s_dev**2)) 
    return eigvals, eigvecs, von_mises, max_shear, sigma_oct, 
tau_oct 
 
vals_Al = stress_measures(sigma_Al) 
vals_St = stress_measures(sigma_St) 
 
 
 
gamma_xz = 2 * strain[0, 2]   
angle_change_deg = np.degrees(gamma_xz) 
percent_change = 100 * angle_change_deg / 90.0 
 
 
np.set_printoptions(precision=6, suppress=True) 
 
print("\nDisplacement gradient (du_i/dx_j):\n", grad_u) 
print("\nSmall strain tensor:\n", strain) 
 
print("\n--- ALUMINUM ---") 
print("Stress tensor (MPa):\n", sigma_Al) 
print("Principal stresses (MPa):", vals_Al[0]) 
print("Von Mises =", vals_Al[2], "MPa") 
print("Max shear =", vals_Al[3], "MPa") 
print("Octahedral normal =", vals_Al[4], "MPa") 
print("Octahedral shear =", vals_Al[5], "MPa") 
 
print("\n--- STEEL ---") 
print("Stress tensor (MPa):\n", sigma_St) 
print("Principal stresses (MPa):", vals_St[0]) 
print("Von Mises =", vals_St[2], "MPa") 
print("Max shear =", vals_St[3], "MPa") 
print("Octahedral normal =", vals_St[4], "MPa") 
print("Octahedral shear =", vals_St[5], "MPa") 
 
print("\nAngle change at top-right front corner (deg):", 
angle_change_deg) 
print("Percent change relative to 90°:", percent_change, "%") 
 
 
print("\nYield check (Steel): Von Mises / Yield =", vals_St[2] / 
sigY_St) 
if vals_St[2] > sigY_St: 
    print("=> Steel has yielded at top corners.") 
else: 
    print("=> Steel has NOT yielded at top corners.")
