import torch
from torch import nn
from tqdm import tqdm
from src.Decoder import Decoder
from src.Encoder import Encoder
from src.utils import *
from src.VAE import VariationalAutoEncoder
import shutil
import wandb
wandb.login(key="40531f3abc9f15fec5bbf4e37dc3148f1140e16c")
hyperparams = {
    "learning_rate": 1e-3,
    "batch_size": 256,
    "input_dim": 28 * 28,
    "hidden_dim": 256,
    "latent_dim": 128,
    "epochs": 100
}

wandb.init(project="VAE", name="MNIST", config=hyperparams)

device = check_gpu()
training_dataloader, test_dataloader = load_data(batch_size=hyperparams["batch_size"], data="MNIST")

# Move model and criterion to GPU
VAE_Encoder = Encoder(hyperparams["input_dim"], hyperparams["hidden_dim"], hyperparams["latent_dim"]).to(device)
VAE_Decoder = Decoder(hyperparams["latent_dim"], hyperparams["hidden_dim"], hyperparams["input_dim"]).to(device)
VAE = VariationalAutoEncoder(VAE_Encoder, VAE_Decoder).to(device)
optimizer = torch.optim.Adam(VAE.parameters(), lr=hyperparams["learning_rate"],weight_decay=0,eps=1e-6)

def criterion(x_hat, x, mu, logvar):
    RECON = nn.functional.binary_cross_entropy_with_logits(input=x_hat, target=x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = RECON + KLD
    return loss

terminal_width = shutil.get_terminal_size().columns
print("-" * terminal_width)
print("Variational AutoEncoder")
print("-" * terminal_width)

VAE.train()

for epoch in range(hyperparams["epochs"]):
    overall_loss = 0
    with tqdm(total=len(training_dataloader), desc=f'Epoch {epoch + 1}/{hyperparams["epochs"]}', unit='batch') as pbar:
        for batch_idx, (x, _) in enumerate(training_dataloader):
            x = x.view(hyperparams["batch_size"], hyperparams["input_dim"]).to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = VAE(x)
            loss = criterion(x_hat, x, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)
            wandb.log({'StepLoss': loss.item()})
    wandb.log({'Loss': overall_loss})
    print(f"\nEpoch {epoch + 1} complete! \t Average Loss: {overall_loss / (batch_idx* hyperparams['batch_size'])}")

print("-" * terminal_width)
print("Finish Variational AutoEncoder Training")
print("-" * terminal_width)

# Save the model
torch.save(VAE.state_dict(), 'vae_model.pth')