import torch

def StructuralSimilarityIndexMeasure(original, reconstruction, data_range=1.0, K1=0.01, K2=0.03, eps=1e-8):
    """
    Compute Structural Similarity Index Measure (SSIM) between two tensors.
    Works for grayscale or RGB images (normalized to [0,1]).

    Args:
        original (torch.Tensor): Ground truth tensor, shape (N, C, H, W)
        reconstruction (torch.Tensor): Reconstructed tensor, same shape
        data_range (float): Value range of the inputs (1.0 for normalized images)
        K1, K2: Stability constants
        eps: Small epsilon for numerical stability

    Returns:
        torch.Tensor: Scalar SSIM value
    """
    mu_x = original.mean()
    mu_y = reconstruction.mean()
    sigma_x = original.var()
    sigma_y = reconstruction.var()
    sigma_xy = ((original - mu_x) * (reconstruction - mu_y)).mean()

    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + eps)
    return ssim
