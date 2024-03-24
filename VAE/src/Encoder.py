import torch 
from torch import nn

#--------------------Encoder---------------#
class Encoder(nn.Module):
    """
    Encoder component of a Variational AutoEncoder (VAE).

    This class represents the encoder part of a VAE, responsible for encoding input data 
    into a latent space representation.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layers.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        fc_1 (torch.nn.Linear): Fully connected layer for the first hidden layer.
        fc_2 (torch.nn.Linear): Fully connected layer for the second hidden layer.
        _mu (torch.nn.Linear): Fully connected layer for calculating the mean of the latent space.
        sigma (torch.nn.Linear): Fully connected layer for calculating the standard deviation of the latent space.
        LeakyReLU (torch.nn.LeakyReLU): Leaky ReLU activation function with slope 0.2.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc_2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.LeakyReLU(self.fc_1(x))
        mean = self.fc_2_mean(x)
        logvar = self.fc_2_logvar(x)
        return mean, logvar