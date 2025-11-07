import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def to_numpy(arr):
    # Convert input to a numpy array safely
    if isinstance(arr, list):
        arr_np = np.array([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in arr])
    elif isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy()
    else:
        arr_np = np.array(arr)
    return arr_np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_psnr(train_accuracies, test_accuracies, save_dir=None, name=None):
    train_accuracies_np = to_numpy(train_accuracies)
    test_accuracies_np = to_numpy(test_accuracies)
    epochs_range = range(1, len(train_accuracies_np) + 1)
    plt.plot(epochs_range, train_accuracies_np, 'b-o', label='Training PSNR', linewidth=2, markersize=6)
    plt.plot(epochs_range, test_accuracies_np, 'r-s', label='Test PSNR', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Peak Signal to Noise Ratio')
    plt.title('Training and Test PSNR', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir and name:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{name}_psnr.png")
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_psnr_improvement(test_accuracies, save_dir=None, name=None):
    test_accuracies_np = to_numpy(test_accuracies)
    epochs_range = range(1, len(test_accuracies_np) + 1)
    improvement = test_accuracies_np - test_accuracies_np[0]
    plt.bar(epochs_range, improvement, alpha=0.7, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR Improvement')
    plt.title('Test PSNR Improvement', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir and name:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{name}_psnr_improvement.png")
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_ssim(train_accuracies, test_accuracies, save_dir=None, name=None):
    train_accuracies_np = to_numpy(train_accuracies)
    test_accuracies_np = to_numpy(test_accuracies)
    epochs_range = range(1, len(train_accuracies_np) + 1)
    plt.plot(epochs_range, train_accuracies_np, 'b-o', label='Training SSIM', linewidth=2, markersize=6)
    plt.plot(epochs_range, test_accuracies_np, 'r-s', label='Test SSIM', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Structural Similarity Index Measure')
    plt.title('Training and Test SSIM', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir and name:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{name}_ssim.png")
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_ssim_improvement(test_accuracies, save_dir=None, name=None):
    test_accuracies_np = to_numpy(test_accuracies)
    epochs_range = range(1, len(test_accuracies_np) + 1)
    improvement = test_accuracies_np - test_accuracies_np[0]
    plt.bar(epochs_range, improvement, alpha=0.7, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Improvement')
    plt.title('Test SSIM Improvement', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir and name:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"{name}_ssim_improvement.png")
        plt.savefig(save_path, dpi=300)

    plt.show()
