import torch
import torch.nn as nn

class NegativeELBO(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' for manual reduction

    def forward(self, logits, labels, mu, log_sigma_2):
        # Reconstruction: sum over pixels (dim=1), then mean over batch
        reconstruction = self.bce(logits, labels).sum(dim=1).mean()
    
        # KL: sum over latent dims (dim=1), then mean over batch
        kl_divergence = (0.5 * (mu**2 + torch.exp(log_sigma_2) - log_sigma_2 - 1)).sum(dim=1).mean()
    
        neg_elbo = reconstruction + kl_divergence
        return neg_elbo, kl_divergence, reconstruction