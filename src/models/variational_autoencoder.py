import torch
import torch.nn as nn

from src.components.variational_autoencoder import Encoder, Decoder

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = Encoder(self.num_layers, self.input_size, self.hidden_size, self.latent_size)
        self.decoder = Decoder(self.num_layers, self.latent_size, self.hidden_size, self.input_size)

        self._init_params()

    def forward(self, X):
        # Encode
        mu, log_sigma_2 = self.encoder(X)
        # Sample Noise from fixed distribution
        epsilon = torch.randn(self.latent_size, device=X.device)
        sigma = torch.exp(0.5 * log_sigma_2)
        # Create latent code via latent transform
        z = mu + sigma*epsilon
        # Decode
        logits = self.decoder(z)

        return logits, mu, log_sigma_2

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)