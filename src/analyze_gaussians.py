import torch
import pathlib as Path

# 1. Load the Checkpoint

# Path to the saved model checkpoint (contains optimized parameters like means, scales, etc.)
ckpt_path = Path("results/lantern_spectral/ckpts/train_step9999.pt")
ckpt = torch.load(ckpt_path)

# Extract the dictionary of Gaussian parameters ("splats") from the checkpoint
# This dictionary matches the state_dict of the ParameterDict used during training.
splats_state = ckpt["splats"]

# 2. Extract Parameters

# 'scales' are stored as log-scales during optimization for numerical stability.
# Shape: (N, 3) where N is the number of Gaussians.
scales_log: torch.Tensor = splats_state["scales"]   

# 'means' are the 3D world coordinates of the Gaussian centers.
# Shape: (N, 3).
means: torch.Tensor = splats_state["means"]         

N = scales_log.shape[0] # Total number of Gaussians in the scene
print(f"Loaded {N} Gaussians from {ckpt_path}")

# 3. Select Gaussians for Analysis

# Arbitrarily pick two Gaussians by index to compare their properties.
i = 0
j = 1

# Retrieve the specific parameters for the chosen Gaussians
s1_log = scales_log[i]          # Log-scales for Gaussian i (3,)
s2_log = scales_log[j]          # Log-scales for Gaussian j (3,)
m1 = means[i]                   # Position for Gaussian i
m2 = means[j]                   # Position for Gaussian j

# 4. Compute Covariance (Principal-Axis Frame)

def scales_to_cov(scales_log: torch.Tensor) -> torch.Tensor:
    """
    Constructs the covariance matrix in the local (principal-axis) frame.
    
    Note: Real world-space covariance would require rotation (quaternions).
    However, eigenvalues and spectral norms are rotation-invariant, so 
    analyzing the diagonal matrix of squared scales is sufficient for 
    understanding the shape/anisotropy.
    """
    # Convert log-scales back to linear scales (standard deviations)
    s = torch.exp(scales_log)       # (3,)
    
    # Square the scales to get variances (eigenvalues of the covariance matrix)
    s2 = s ** 2                     
    
    # Construct a 3x3 diagonal matrix. 
    # Since we are in the local frame, the covariance is diagonal.
    return torch.diag(s2)           # (3, 3)

Sigma1 = scales_to_cov(s1_log)
Sigma2 = scales_to_cov(s2_log)

# 5. Spectral Analysis (Eigenvalues & Condition Number)

def analyze_covariance(Sigma: torch.Tensor):
    """
    Performs spectral analysis on the covariance matrix.
    
    Returns:
        vals: Eigenvalues (variances along the principal axes).
        vecs: Eigenvectors (principal directions - aligned with axes here).
        spectral_norm: The largest eigenvalue (maximum variance).
        cond_number: The ratio of largest to smallest eigenvalue (measure of anisotropy).
                     Close to 1.0 means spherical; large values mean needle-like or flat.
    """
    # Compute eigenvalues and eigenvectors for a symmetric matrix (eigh)
    vals, vecs = torch.linalg.eigh(Sigma)  # 'vals' are returned in ascending order
    
    spectral_norm = vals.max().item()
    
    # Condition number = lambda_max / lambda_min
    # Indicates how stretched the Gaussian is.
    cond_number = (vals.max() / vals.min()).item()
    
    return vals, vecs, spectral_norm, cond_number

vals1, vecs1, spec1, cond1 = analyze_covariance(Sigma1)
vals2, vecs2, spec2, cond2 = analyze_covariance(Sigma2)

# 6. Report Results

def print_gaussian_report(idx, mean, vals, spec, cond):
    print(f"\n=== Gaussian {idx} ===")
    print(f"Mean (world coords): {mean.tolist()}")
    print(f"Eigenvalues (axis variances): {vals.tolist()}")
    print(f"Spectral norm (largest eigenvalue): {spec:.6f}")
    print(f"Condition number (lambda_max / lambda_min): {cond:.3f}")
    print("  -> Interpretation: " + ("isotropic" if cond < 2.0 else "anisotropic"))

print_gaussian_report(i, m1, vals1, spec1, cond1)
print_gaussian_report(j, m2, vals2, spec2, cond2)
