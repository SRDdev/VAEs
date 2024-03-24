# This code is contributed by Shreyas Dixit under the Paper Replication series
from tqdm import tqdm
import torch 

def final_loss(bce_loss, mu, logvar):
    """
    Computes the final loss by combining the reconstruction loss (BCELoss)
    and the Kullback-Leibler Divergence (KL-Divergence).
    
    Parameters:
        bce_loss (torch.Tensor): Reconstruction loss.
        mu (torch.Tensor): Mean of the latent vector.
        logvar (torch.Tensor): Log variance of the latent vector.
    
    Returns:
        torch.Tensor: Final combined loss.
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    """
    Trains the VAE model.

    Parameters:
        model (torch.nn.Module): The VAE model.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        dataset (torch.utils.data.Dataset): Training dataset.
        device (torch.device): Device to perform operations on (CPU or GPU).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion: Loss criterion for calculating reconstruction loss.
        
    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    """
    Validates the VAE model.

    Parameters:
        model (torch.nn.Module): The VAE model.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        dataset (torch.utils.data.Dataset): Validation dataset.
        device (torch.device): Device to perform operations on (CPU or GPU).
        criterion: Loss criterion for calculating reconstruction loss.
        
    Returns:
        tuple: Validation loss, reconstructed images.
    """
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images
