def train_epoch(model, dataloader, criterion, optimizer, device, update_freq):
    model.train()

    total_loss = 0.0
    total_kl_divergence = 0.0
    total_reconstruction = 0.0
    num_batches = 0

    for batch_idx, (features, _) in enumerate(dataloader):
        features = features.to(device)
        labels = features

        optimizer.zero_grad()

        features = features.view(features.size(0), -1)  # Flatten
        labels = labels.view(labels.size(0), -1)
        
        logits, mu, log_sigma_2 = model(features)

        loss, kl_divergence, reconstruction = criterion(logits, labels, mu, log_sigma_2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl_divergence += kl_divergence.item()
        total_reconstruction += reconstruction.item()
        num_batches +=1

        if(batch_idx+1)%update_freq == 0:
            print(f'   Batch [{batch_idx+1}/{len(dataloader)}] - 'f'Loss: {loss.item():.4f} ; KL-Divergence: {kl_divergence.item() : .4f} ; Reconstruction: {reconstruction.item() : .4f}')

    avg_loss = total_loss/num_batches
    avg_kl_divergence = total_loss/num_batches
    avg_reconstruction = total_reconstruction/num_batches

    return avg_loss, avg_kl_divergence, avg_reconstruction