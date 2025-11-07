import matplotlib.pyplot as plt

def plot_loss(train_losses, test_losses, save_path):
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs_range, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # dpi=300 gives high-quality image

    plt.show()