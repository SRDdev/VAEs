# This code is contributed by Shreyas Dixit under the Paper Replication series
import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()

def image_to_vid(images):
    """
    Convert a list of images into a GIF.

    Parameters:
        images (list): List of images in PyTorch tensor format.
    """
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('./outputs/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch):
    """
    Save reconstructed images to disk.

    Parameters:
        recon_images (torch.Tensor): Reconstructed images.
        epoch (int): Epoch number for naming the saved file.
    """
    save_image(recon_images.cpu(), f"./outputs/output{epoch}.jpg")

def save_loss_plot(train_loss, valid_loss):
    """
    Plot and save the training and validation loss.

    Parameters:
        train_loss (list): List of training loss values.
        valid_loss (list): List of validation loss values.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss.jpg')
    plt.show()
