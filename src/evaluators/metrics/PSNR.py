import torch

def PeakSignalToNoiseRatio(original, reconstruction, data_range=1.0, eps=1e-8):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two tensors.
    
    Args:
        original (torch.Tensor): Ground truth image (or signal)
        reconstruction (torch.Tensor): Reconstructed image (or signal)
        data_range (float): Maximum possible value in the data (1.0 if normalized, 255 if 8-bit)
        eps (float): Small constant to avoid log of zero
    
    Returns:
        torch.Tensor: PSNR value in decibels (dB)
    """
    mse = torch.mean((original - reconstruction) ** 2)
    psnr = 10 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr
