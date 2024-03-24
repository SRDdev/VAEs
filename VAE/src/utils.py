# This code is contributed by Shreyas Dixit under the Paper Replication series
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
def check_gpu():
    terminal_width = shutil.get_terminal_size().columns
    print("")
    print("Hardware Specifications")
    print("-"*terminal_width)
    print('PyTorch version:', torch.__version__)
    print('GPU name:', torch.cuda.get_device_name())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is:', device)
    print('Total number of GPUs:', torch.cuda.device_count())
    print("-"*terminal_width)
    return device


def load_data(batch_size=32,data="MNIST"):
    transform = transforms.Compose([
        transforms.ToTensor()])
    if data =="USPS":
        training_dataset = datasets.USPS('./data_src', train=True, download=True, transform=transform)
        test_dataset = datasets.USPS('./data_src', train=False, download=True, transform=transform)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    elif data=="MNIST":
        training_dataset = datasets.MNIST('./data_src', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data_src', train=False, download=True, transform=transform)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return training_dataloader, test_dataloader
