if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    from src.models import VariationalAutoencoder
    from data.dowload_dataset import download_MNIST_VAE
    from src.evaluators.metrics import NegativeELBO
    from src.utils.loaders import create_DataLoaders
    from src.trainers.variational_autoencoder import train_vae
    
    from src.utils.visualization import plot_loss, plot_reconstructions, plot_prior_samples, plot_latent_interpolation
    
    BATCH_SIZE = 100
    NUM_EPOCHS = 50
    NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset, test_dataset = download_MNIST_VAE()

    train_dataloader, test_dataloader = create_DataLoaders(train_dataset, test_dataset, batch_size=BATCH_SIZE, shuffle_train=True, num_workers=NUM_WORKERS)

    model = VariationalAutoencoder(num_layers=2, input_size=784, hidden_size=256, latent_size=32)
    model.to(DEVICE)

    criterion = NegativeELBO()

    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    train_losses, test_losses, train_kl_divergences, test_kl_divergences, train_reconstructions, test_reconstructions, all_reconstructions, all_originals = train_vae(num_epochs=NUM_EPOCHS, train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, criterion=criterion, optimizer=optimizer, device=DEVICE, update_freq=100)
    
    torch.save(model.state_dict(), "experiments/variational-autoencoders/vae_weights.pth")
    
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_kl_divergences': train_kl_divergences,
        'train_reconstructions': train_reconstructions,
        'test_losses': test_losses,
        'test_kl_divergences': test_kl_divergences,
        'test_reconstructions': test_reconstructions,
    }, "experiments/variational-autoencoders/vae_detailed.pth")
    
    plot_loss(train_losses, test_losses, "experiments/variational-autoencoders/neg-elbo")
    plot_loss(train_reconstructions, test_reconstructions, "experiments/variational-autoencoders/reconstructions")
    plot_loss(train_kl_divergences, test_kl_divergences, "experiments/variational-autoencoders/kl_divergence")
    plot_reconstructions(model, test_dataloader, DEVICE)
    plot_prior_samples(model, DEVICE)
    plot_latent_interpolation(model, test_dataloader, DEVICE)