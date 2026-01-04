import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_layers, latent_size, hidden_size, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = []

        #First layer
        self.layers.append(nn.Linear(self.latent_size, self.hidden_size))
        self.layers.append(nn.ReLU())

        # 500 hidden layers
        self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.ReLU())

        #Final layer
        self.layers.append(nn.Linear(self.hidden_size, self.output_size)) #Only `output_size` no. of outputs because it is binary data, so bernoulli distribution (only one output needed per pixel)

        #Combined Network
        self.network = nn.Sequential (*self.layers)

    def forward(self, X):
        y = self.network(X)
        return y