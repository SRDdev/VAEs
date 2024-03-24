import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    """
    Variational AutoEncoder (VAE) model.

    This class represents a Variational AutoEncoder, which consists of an encoder 
    and a decoder. It learns to reconstruct input data and generate new samples 
    from the learned latent space distribution.

    Args:
        Encoder (nn.Module): Encoder module.
        Decoder (nn.Module): Decoder module.

    Attributes:
        encoder (nn.Module): Instance of the Encoder module.
        decoder (nn.Module): Instance of the Decoder module.
    """
    def __init__(self, Encoder, Decoder):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def latent_space(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.rand_like(std)
        z_sampled = mu + (eps * std)
        return z_sampled
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.latent_space(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
