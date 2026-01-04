import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, latent_size):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.layers = []
        
        # First layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        self.layers.append(nn.ReLU())

        # 500 hidden layers (as per VAE paper for MNIST)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.ReLU())

        #Final Layer
        self.layers.append(nn.Linear(self.hidden_size, 2*self.latent_size)) #2*latent_size: Because distribution will be per pixel in latent dimension (first `latent_size`: mean, next `latent_size`: log(variance))

        #Sequential combination of above
        self.network = nn.Sequential(*self.layers)

    def forward(self, X):
        mu, log_sigma_2 = torch.split(self.network(X), self.latent_size, dim=-1)
        return mu, log_sigma_2