import torch 
import torch.nn as nn

from .metrics import StructuralSimilarityIndexMeasure, PeakSignalToNoiseRatio

def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    all_reconstructions = []
    all_originals = []
    
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            labels = features
            
            features = features.view(features.size(0), -1)  # flatten
            labels = labels.view(labels.size(0), -1)  
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            psnr = PeakSignalToNoiseRatio(original=labels, reconstruction=outputs)
            ssim = StructuralSimilarityIndexMeasure(original=labels, reconstruction=outputs)
            total_loss +=loss.item()
            total_psnr+=psnr
            total_ssim+=ssim
            num_batches+=1
            
            all_reconstructions.extend(outputs.cpu().numpy())
            all_originals.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / num_batches
    avg_pnsr = total_psnr / num_batches
    avg_ssim = total_ssim/num_batches

    return avg_loss, avg_pnsr, avg_ssim, all_reconstructions, all_originals