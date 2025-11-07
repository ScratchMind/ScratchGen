import torch
import torch.nn as nn
import time

from .train_autencoder_epoch import train_epoch
from ..evaluators import evaluate

def train_autoencoder(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, target_psnr, target_ssim, device, update_freq):
    train_losses = []
    train_psnrs = []
    train_ssims = []
    
    test_losses = []
    test_psnrs = []
    test_ssims = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        #Train
        print("INFO - Training...")
        train_loss, train_psnr, train_ssim = train_epoch(model, train_dataloader, criterion, optimizer, device, update_freq=update_freq)
        
        #Eval
        print("INFO - Evaluating...")
        test_loss, test_psnr, test_ssim, _, _ = evaluate(model, test_dataloader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)
        train_ssims.append(train_ssim)
        test_losses.append(test_loss)
        test_psnrs.append(test_psnr)
        test_ssims.append(test_ssim)
        
        epoch_time = time.time() - epoch_start_time
        
        if test_psnr > target_psnr and test_ssim > target_ssim:
            print(f"🎯 Targets reached! Stopping early.")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"🕒 Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"🎯 Final test peak signal to noise ratio: {test_psnrs[-1]:.2f}%")
    print(f"🎯 Final test structured similarity index measure: {test_ssims[-1]:.2f}%")
    
    return train_losses, train_psnrs, train_ssims, test_losses, test_psnrs, test_ssims
        