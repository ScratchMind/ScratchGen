import torch

def evaluate(model, dataloader, criterion, device, use_mean=False):
    model.eval()
    total_loss = 0.0
    total_kl_divergence = 0.0
    total_reconstruction = 0.0
    num_batches = 0

    all_reconstructions = []
    all_originals = []

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            features = features.view(features.size(0), -1)
            logits, mu, log_sigma_2 = model(features)

            loss, kl_divergence, reconstruction = criterion(logits, features, mu, log_sigma_2)

            total_loss += loss.item()
            total_kl_divergence += kl_divergence.item()
            total_reconstruction += reconstruction.item()
            num_batches += 1

            #probabilites
            probs = torch.sigmoid(logits)

            all_reconstructions.append(probs.cpu())
            all_originals.append(features.cpu())

    avg_loss = total_loss / num_batches
    avg_kl_divergence = total_kl_divergence/num_batches
    avg_reconstruction = total_reconstruction/num_batches

    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    all_originals = torch.cat(all_originals, dim=0)

    return avg_loss, avg_kl_divergence, avg_reconstruction, all_reconstructions, all_originals