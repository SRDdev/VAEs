import torch
import matplotlib.pyplot as plt
from src.Decoder import Decoder
from src.Encoder import Encoder
from src.VAE import VariationalAutoEncoder
from src.utils import load_data

# Define hyperparameters
batch_size = 256
input_dim = 28*28
hidden_dim = 256
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_dataloader, test_dataloader = load_data(batch_size=batch_size, data="MNIST")

# Load the saved model
VAE_Encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
VAE_Decoder = Decoder(latent_dim, hidden_dim, input_dim).to(device)
VAE = VariationalAutoEncoder(VAE_Encoder, VAE_Decoder).to(device)
VAE.load_state_dict(torch.load('vae_model.pth'))
VAE.eval()

# Plot original and reconstructed images
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_dataloader):
        x = x.to(device)
        x_hat, _, _ = VAE(x)
        x = x.cpu().numpy().reshape(-1, 28, 28)  # Move tensor to CPU and then convert to numpy
        x_hat = torch.sigmoid(x_hat).cpu().numpy().reshape(-1, 28, 28)  # Move tensor to CPU and then convert to numpy
        break  # Remove this break statement if you want to iterate over the entire test dataset

plt.figure(figsize=(10, 4))
for i in range(5):  # Plotting first 5 images
    # Original Image
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i], cmap='gray')
    plt.title('Original')
    plt.axis('off')
    # Reconstructed Image
    plt.subplot(2, 5, i + 6)
    plt.imshow(x_hat[i], cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()
