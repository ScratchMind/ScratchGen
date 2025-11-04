import torch 
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_space_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, input_size):
        super(Decoder, self).__init__()
        
        #Initialize params
        self.latent_space_size = latent_space_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.input_size = input_size

        # Activation Function
        self.activation = nn.ReLU()
        
        self.hidden1 = nn.Linear(self.latent_space_size, self.hidden1_size)
        self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.hidden3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.hidden4 = nn.Linear(self.hidden3_size, self.hidden4_size)
        self.reconstruction = nn.Linear(self.hidden4_size, self.input_size)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.activation(X)
        X = self.hidden2(X)
        X = self.activation(X)
        X = self.hidden3(X)
        X = self.activation(X)
        X = self.hidden4(X)
        X = self.activation(X)
        X = self.reconstruction(X)
        return X
