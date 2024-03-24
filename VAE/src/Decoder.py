# This code is contributed by Shreyas Dixit under the Paper Replication series
import torch
from torch import nn

class Decoder(nn.Module):
    """
    Decoder component of a Variational AutoEncoder (VAE).
    This class represents the decoder part of a VAE, responsible for reconstructing
    data from the latent space representation.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        hidden_dim (int): Dimensionality of the hidden layers.
        output_dim (int): Dimensionality of the output.

    Attributes:
        FC_hidden (torch.nn.Linear): Fully connected layer for the first hidden layer.
        FC_hidden2 (torch.nn.Linear): Fully connected layer for the second hidden layer.
        FC_output (torch.nn.Linear): Fully connected layer for the output.
        LeakyReLU (torch.nn.LeakyReLU): Leaky ReLU activation function with slope 0.2.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        z = self.LeakyReLU(self.FC_hidden(z))
        z = self.LeakyReLU(self.FC_hidden2(z))
        x_hat = torch.sigmoid(self.FC_output(z)) 
        return x_hat
