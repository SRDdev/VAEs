import torch
import torch.optim as optim
import torch.nn as nn
from model import ConvVAE
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
import shutil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae_model = ConvVAE().to(device)

lr = 0.001
epochs = 10
batch_size = 64
optimizer = optim.Adam(vae_model.parameters(), lr=lr,eps=1e-6)
criterion = nn.BCELoss(reduction='sum')  
grid_images = []
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),])

trainset = torchvision.datasets.FashionMNIST(root='./input', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='./input', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []
terminal_width = shutil.get_terminal_size().columns
print("-"*terminal_width)
print("Training ConvVAE")
print("-"*terminal_width)
for epoch in range(epochs):
    print("")
    print(f"Epoch {epoch+1} of {epochs}")
    print("-"*(terminal_width))
    train_epoch_loss = train(vae_model, trainloader, trainset, device, optimizer, criterion)
    valid_epoch_loss, recon_images = validate(vae_model, testloader, testset, device, criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    save_reconstructed_images(recon_images, epoch+1)
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print("-"* int(terminal_width/2))
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    print("-"* int(terminal_width/2))


image_to_vid(grid_images)
save_loss_plot(train_loss, valid_loss)
print("-" * terminal_width)
print("Finish Convolutional Variational AutoEncoder Training")
print("-" * terminal_width)
torch.save(vae_model.state_dict(), 'conv_vae_model.pth')
