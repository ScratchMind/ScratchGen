import torch

def plot_prior_samples(model, device, num_samples=16):
    import matplotlib.pyplot as plt
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_size).to(device)
        logits = model.decoder(z)
        samples = torch.sigmoid(logits)

    side = int(samples.size(1) ** 0.5)
    samples = samples.view(num_samples, side, side).cpu()

    grid_size = int(num_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for i in range(num_samples):
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')

    plt.suptitle("Samples from p(z) ~ N(0, I)", fontsize=14)
    plt.show()
