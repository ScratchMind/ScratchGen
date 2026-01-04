import torch
import torch.nn as nn
import torch.optim as optim

from ..evaluators import PeakSignalToNoiseRatio, StructuralSimilarityIndexMeasure

def train_epoch(model, dataloader, criterion, optimizer, device, update_freq=10):
    # train mode
    model.train()
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    for batch_idx, (features, _) in enumerate(dataloader):
        features = features.to(device)
        labels = features
        
        # Zero the gradients
        optimizer.zero_grad()
        
        #Forward Pass
        features = features.view(features.size(0), -1)  # Flatten
        labels = labels.view(labels.size(0), -1)        # flatten labels
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        
        #Calculate metrics
        psnr = PeakSignalToNoiseRatio(original=labels, reconstruction=outputs)
        ssim = StructuralSimilarityIndexMeasure(original=labels, reconstruction=outputs)
        total_psnr+=psnr 
        total_ssim += ssim 
        total_loss+=loss.item()
        num_batches+=1
        
        if(batch_idx+1)%update_freq == 0:
            print(f'   Batch [{batch_idx+1}/{len(dataloader)}] - 'f'Loss: {loss.item():.4f}, PSNR: {psnr:.2f}, SSIM: {ssim: .2f}')
            
    avg_loss = total_loss/num_batches
    avg_psnr = total_psnr/num_batches
    avg_ssim = total_ssim/num_batches

    return avg_loss, avg_psnr, avg_ssim