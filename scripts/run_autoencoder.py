
if __name__ == "__main__":
    import torch
    import os
    import yaml
    import torch.nn as nn
    import torch.optim as optim
    from src.models import Autoencoder
    from data.dowload_dataset import download_MNIST, download_CIFAR10, download_FashionMNIST
    from src.utils.loaders import create_DataLoaders
    from src.trainers import train_autoencoder

    # Load YAML config file
    with open('configs/autoencoder.yml', 'r') as file:
        config = yaml.safe_load(file)

    if config['dataset'] == 'MNIST':
        train_dataset, test_dataset = download_MNIST()
    elif config['dataset'] == 'FashionMNIST':
        train_dataset, test_dataset = download_FashionMNIST()
    else:
        train_dataset, test_dataset = download_CIFAR10()

    train_dataloader, test_dataloader = create_DataLoaders(
        train_dataset, 
        test_dataset, 
        config['batch_size'], 
        shuffle_train=config['shuffle_train'], 
        num_workers=config['num_workers']
    )

    model = Autoencoder(
        input_size= config['input_size'],
        hidden1_size= config['hidden1_size'],
        hidden2_size= config['hidden2_size'],
        hidden3_size=config['hidden3_size'],
        hidden4_size=config['hidden4_size'],
        latent_space_size=config['latent_space_size']
        ).to(config['device'])

    if config['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    if config['optimizer'] == 'Adam':
        weight_decay = float(config['weight_decay'])
        optimizer = optim.Adam(
            model.parameters(),
            lr = config['learning_rate'],
            weight_decay=weight_decay
        )

    train_losses, train_psnrs, train_ssims, test_losses, test_psnrs, test_ssims = train_autoencoder(
        num_epochs= config['num_epochs'],
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        target_psnr=config['target_psnr'],
        target_ssim=config['target_ssim'],
        device=config['device'],
        update_freq=config['update_freq']
    )

    # Save weights
    os.makedirs("experiments/autoencoders", exist_ok=True)
    torch.save(model.state_dict(), "experiments/autoencoders/ae_weights.pth")

    # Save full checkpoint
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_psnrs': train_psnrs,
        'train_ssims': train_ssims,
        'test_losses': test_losses,
        'test_psnrs': test_psnrs,
        'test_ssims': test_ssims,
    }, "experiments/autoencoders/ae_detailed.pth")
    
    from src.utils.visualization import plot_loss, plot_psnr, plot_ssim, plot_psnr_improvement, plot_ssim_improvement
    save_path = "experiments/autoencoders"
    plot_loss(train_losses, test_losses, save_path)
    plot_psnr(train_psnrs, test_psnrs, save_path)
    plot_ssim(train_ssims, test_ssims, save_path)
    plot_psnr_improvement(test_psnrs, save_path)
    plot_ssim_improvement(test_ssims, save_path)
