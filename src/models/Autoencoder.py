import torch 
import torch.nn as nn

from ..components.autoencoder import Encoder
from ..components.autoencoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, latent_space_size):
        super(Autoencoder, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.latent_space_size = latent_space_size
        
        self.encoder = Encoder(input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, latent_space_size)
        self.decoder = Decoder(latent_space_size, hidden4_size, hidden3_size, hidden2_size, hidden1_size, input_size)

    def forward(self, X):
        z = self.encoder(X)
        reconstruction = self.decoder(z)
        return reconstruction