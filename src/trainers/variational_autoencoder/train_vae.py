import time

from src.trainers.variational_autoencoder.train_vae_epoch import train_epoch
from src.evaluators.evaluate_vae import evaluate

def train_vae(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device, update_freq):
    train_losses = []
    test_losses = []
    train_kl_divergences = []
    test_kl_divergences = []
    train_reconstructions = []
    test_reconstructions = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        #Train
        print("INFO - Training...")
        train_loss, train_kl_divergence, train_reconstruction = train_epoch(model, train_dataloader, criterion, optimizer, device, update_freq=update_freq)
        
        #Eval
        print("INFO - Evaluating...")
        test_loss, test_kl_divergence, test_reconstruction, all_reconstructions, all_originals = evaluate(model, test_dataloader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_kl_divergences.append(train_kl_divergence)
        test_kl_divergences.append(test_kl_divergence)
        train_reconstructions.append(train_reconstruction)
        test_reconstructions.append(test_reconstruction)
        
        epoch_time = time.time() - epoch_start_time
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print(f"🕒 Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    return train_losses, test_losses, train_kl_divergences, test_kl_divergences, train_reconstructions, test_reconstructions, all_reconstructions, all_originals