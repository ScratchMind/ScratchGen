import torchvision
from torchvision.transforms import transforms


def download_FashionMNIST():
    # Define data transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Mean and Std for Fashion-MNIST
    ])
    
    # Load training dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../data/train',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../data/test',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def download_MNIST():
    # Define data transforms for preprocessing
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std for MNIST
    ])
    
    # Load training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='../data/train',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='../data/test',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def download_CIFAR10():
    # Define data transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # Mean for RGB channels
                            (0.2023, 0.1994, 0.2010))  # Std for RGB channels
    ])
    
    # Load training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data/train',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data/test',
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset
