import torch
import math

def spectral_entropy_from_scales(scales_log, eps=1e-8):
    """
    scales_log: (N, 3) tensor of log-scales (like in gsplat: log of radii along x,y,z)
    returns: (N,) tensor with spectral entropy per Gaussian (normalized to [0, 1])
    """
    # 1) Go from log-scale to scale
    scales = torch.exp(scales_log)          # (N, 3), all positive

    # 2) Work with squared scales (these are proportional to eigenvalues s_i^2)
    s2 = scales ** 2                        # (N, 3)

    # 3) Normalize to get something like probabilities t_i = s_i^2 / sum_j s_j^2
    trace = s2.sum(dim=-1, keepdim=True)   # (N, 1)
    t = s2 / (trace + eps)                 # (N, 3), each row sums to ~1

    # 4) Spectral entropy H = -Σ t_i log t_i  
    H = -(t * (t + eps).log()).sum(dim=-1)  # (N,)

    # 5) Normalize by log(3) so H_norm is in [0, 1]
    H_norm = H / math.log(3.0)

    return H_norm

def spectral_anisotropy_loss(scales_log, eps=1e-8):
    """
    Returns a scalar loss that penalizes anisotropic (needle-like) Gaussians.
    0 when Gaussians are perfectly isotropic, up to ~1 for very anisotropic ones.
    """
    H_norm = spectral_entropy_from_scales(scales_log, eps=eps)  # (N,)

    # We want high entropy (H_norm → 1). Penalty = 1 - H_norm.
    anisotropy = 1.0 - H_norm             # (N,), in [0, 1]

    # Average over all Gaussians
    loss = anisotropy.mean()

    return loss, H_norm.mean()
