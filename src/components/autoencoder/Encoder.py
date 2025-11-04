import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size):
        super(Encoder, self).__init__()
        
        #Initialize params
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.output_size = output_size

        # Activation Function
        self.activation = nn.ReLU()
        
        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.hidden3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.hidden4 = nn.Linear(self.hidden3_size, self.hidden4_size)
        self.latent_space = nn.Linear(self.hidden4_size, self.output_size)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.activation(X)
        X = self.hidden2(X)
        X = self.activation(X)
        X = self.hidden3(X)
        X = self.activation(X)
        X = self.hidden4(X)
        X = self.activation(X)
        X = self.latent_space(X)
        return X
