import torch

def plot_latent_interpolation(model, dataloader, device, steps=10):
    import matplotlib.pyplot as plt
    model.eval()

    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x1 = x[0:1].to(device)
        x2 = x[1:2].to(device)

        x1 = x1.view(1, -1)
        x2 = x2.view(1, -1)

        mu1, _ = model.encoder(x1)
        mu2, _ = model.encoder(x2)

        alphas = torch.linspace(0, 1, steps).to(device)
        zs = torch.stack([(1-a)*mu1 + a*mu2 for a in alphas]).squeeze(1)

        logits = model.decoder(zs)
        recon = torch.sigmoid(logits)

    side = int(recon.size(1) ** 0.5)
    recon = recon.view(steps, side, side).cpu()

    fig, axes = plt.subplots(1, steps, figsize=(steps*2, 2))
    for i in range(steps):
        axes[i].imshow(recon[i], cmap='gray')
        axes[i].axis('off')

    plt.suptitle("Latent Interpolation", fontsize=14)
    plt.show()
