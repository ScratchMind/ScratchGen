import torch

def plot_reconstructions(model, dataloader, device, num_images=8):
    import matplotlib.pyplot as plt
    model.eval()

    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x.to(device)[:num_images]
        x_flat = x.view(x.size(0), -1)

        mu, _ = model.encoder(x_flat)
        z = mu  # deterministic
        logits = model.decoder(z)
        recon = torch.sigmoid(logits)

    side = int(x_flat.size(1) ** 0.5)
    x = x_flat.view(num_images, side, side).cpu()
    recon = recon.view(num_images, side, side).cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    for i in range(num_images):
        axes[0, i].imshow(x[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i], cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
    plt.show()