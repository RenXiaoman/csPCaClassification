import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # type: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join
import json
import re
import time

# MONAI imports
from monai.losses import DiceFocalLoss # type: ignore

#### Model
# from Model.as_unetr import ADSA_Net
from picai_baseline.unet.training_setup.neural_networks.unets import UNet

# Local imports
from Options.Options_UNet import Options_UNet
# Legacy loader: always reads all imagesTr/labelsTr for training and
# imagesTs/labelsTs for validation. It intentionally ignores --fold.
from dataset_nnunet import Lits_DataSet
# from dataset.dataset_nnunet_fivefold import FiveFoldLitsDataSet
from utils.nnunet_augmentation import nnunet_style_augmentation


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    Polynomial learning rate decay from nnUNet
    lr = initial_lr * (1 - epoch/max_epochs)^exponent
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent


def main():
    # Parse options
    opt_parser = Options_UNet()
    opt = opt_parser.parse()
    # Keep checkpoints and logs from different folds in separate directories.
    base_task_name = re.sub(r"_fold\d+$", "", opt.task_name)
    opt.task_name = f"{base_task_name}_fold{opt.fold}"
    
    # Set device
    if opt.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Initialize model    
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
        channels=[32, 64, 128, 256, 512, 1024],
        ).to(device)
    
    # Resume training from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    
    # Define optimizer (nnUNet style: SGD with momentum 0.99)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    
    if opt.resume:
        if Path(opt.resume).exists():
            print(f"Resuming training from: {opt.resume}")
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_val_dice = checkpoint['best_val_dice']
            best_val_miou = checkpoint['best_val_miou']
            best_loss_epoch = checkpoint['best_loss_epoch']
            best_dice_epoch = checkpoint['best_dice_epoch']
            best_miou_epoch = checkpoint['best_miou_epoch']
            
            # Restore learning rate from checkpoint
            if 'current_lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['current_lr']
                print(f"Resumed learning rate: {checkpoint['current_lr']:.6f}")
            
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {opt.resume} not found, starting from scratch")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function - adjusted for small lesion segmentation
    dice_focal_loss = DiceFocalLoss(include_background=False,
                                    softmax=True, 
                                    to_onehot_y=True,
                                    gamma=2.0,           # Lower gamma for less extreme focal weighting
                                    weight=[0.2, 0.8],    # Higher weight for lesion class
                                    # lambda_dice=0.5,      # Balanced dice and focal
                                    # lambda_focal=0.5
                                    ).to(device)
    
    # Metrics calculation functions
    def calculate_dice(preds, targets):
        """Calculate Dice coefficient"""
        smooth = 1e-6
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        pred_sum = preds_flat.sum()
        target_sum = targets_flat.sum()
        
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        return dice.item()
    
    def calculate_miou(preds, targets):
        """Calculate mean Intersection over Union"""
        smooth = 1e-6
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def save_model_checkpoint(filename):
        """Save model checkpoint with all training metrics"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': avg_train_dice,
            'val_dice': avg_val_dice,
            'train_miou': avg_train_miou,
            'val_miou': avg_val_miou,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'best_val_miou': best_val_miou,
            'best_loss_epoch': best_loss_epoch,
            'best_dice_epoch': best_dice_epoch,
            'best_miou_epoch': best_miou_epoch,
            'current_lr': new_lr
        }, results_dir / filename)
    
    def plot_result_mix(plot_data1, plot_data2, label1, label2, description, save_path, save_name, showCurrentBestLoss=None):
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data1, label=label1)
        plt.plot(plot_data2, label=label2)
        
        if showCurrentBestLoss is not None:
            plt.title(str(description)+f' ({showCurrentBestLoss})')
        else:
            plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(join(save_path, f'{save_name}.png'))
        plt.close()
    

    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Create results directory
    results_dir = Path(opt.checkpoints_dir) / opt.task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_mious = []
    val_mious = []
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    
    # Save training configuration
    config_file = results_dir / 'training_config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(opt), f, indent=4)
    
    # Five-fold loader: resolves paths in splits.json without copying the dataset.
    train_dataset = FiveFoldLitsDataSet(
        Path(opt.datapath),
        split='train',
        fold=opt.fold,
        enable_augmentation=True,
    )
    val_dataset = FiveFoldLitsDataSet(
        Path(opt.datapath),
        split='val',
        fold=opt.fold,
        enable_augmentation=False,
    )

    # Legacy fixed split (independent of --fold): imagesTr/labelsTr for training
    # and imagesTs/labelsTs for validation.
    # train_dataset = Lits_DataSet(
    #     Path(opt.datapath), 'imagesTr', 'labelsTr', enable_augmentation=True
    # )
    # val_dataset = Lits_DataSet(
    #     Path(opt.datapath), 'imagesTs', 'labelsTs', enable_augmentation=False
    # )
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=False)
    
    print(f'Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples')

    # Training loop
    for epoch in range(start_epoch, opt.epoch):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        all_train_dice_scores = []  # 存储每个样本的Dice分数
        all_train_miou_scores = []  # 存储每个样本的mIoU分数
        
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Train]')):
            ADC, DWI, T2W, labels, patient_names = batch_data
            inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 合并ADC和T2W和DWI作为输入
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                
                # Calculate loss
                loss = dice_focal_loss(outputs, labels)
                
                # Calculate metrics - apply softmax first since model outputs logits
                outputs_softmax = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 计算每个样本的Dice和mIoU（case级别）
            batch_size = preds.size(0)
            for i in range(batch_size):
                single_pred = preds[i:i+1]
                single_label = labels[i:i+1]
                dice_score = calculate_dice(single_pred, single_label)
                miou_score = calculate_miou(single_pred, single_label)
                all_train_dice_scores.append(dice_score)
                all_train_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_miou = 0
        all_val_dice_scores = []  # 存储每个样本的Dice分数
        all_val_miou_scores = []  # 存储每个样本的mIoU分数
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Val]')):
                ADC, DWI, T2W, labels, patient_names = batch_data
                inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 合并T2W和DWI作为输入
                labels = labels.to(device)

                with autocast('cuda'):
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = dice_focal_loss(outputs, labels)
                    
                    # Calculate metrics - apply softmax first since model outputs logits
                    outputs_softmax = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)

                val_loss += loss.item()
                
                # 计算每个样本的Dice和mIoU（case级别）
                batch_size = preds.size(0)
                for i in range(batch_size):
                    single_pred = preds[i:i+1]
                    single_label = labels[i:i+1]
                    dice_score = calculate_dice(single_pred, single_label)
                    miou_score = calculate_miou(single_pred, single_label)
                    all_val_dice_scores.append(dice_score)
                    all_val_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_dice = sum(all_train_dice_scores) / len(all_train_dice_scores)  # 样本级别平均
        avg_train_miou = sum(all_train_miou_scores) / len(all_train_miou_scores)  # 样本级别平均
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_dice = sum(all_val_dice_scores) / len(all_val_dice_scores)  # 真正的样本级别平均
        avg_val_miou = sum(all_val_miou_scores) / len(all_val_miou_scores)  # 真正的样本级别平均
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)
        
        # Update learning rate using poly_lr (nnUNet style)
        new_lr = poly_lr(epoch + 1, opt.epoch, opt.lr, 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{opt.epoch}], Time: {epoch_time:.2f}s, LR: {new_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train mIoU: {avg_train_miou:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val mIoU: {avg_val_miou:.4f}')
        
        # Save metrics to file
        metrics_file = results_dir / 'training_metrics.txt'
        with open(metrics_file, 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Train Dice: {avg_train_dice:.6f}, Train mIoU: {avg_train_miou:.6f}, Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.6f}, Val mIoU: {avg_val_miou:.6f}\n')
        
        # Update best validation metrics and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_epoch = epoch + 1
            save_model_checkpoint('best_loss_model.pth')
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_dice_epoch = epoch + 1
            save_model_checkpoint('best_dice_model.pth')
        
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_miou_epoch = epoch + 1
        
        # Always save latest model for resume training
        save_model_checkpoint('model_latest.pth')
        
        # Save current model
        if (epoch + 1) % opt.model_save_fre == 0:
            save_model_checkpoint(f'model_epoch_{epoch+1}.pth')
        
        # Plot results every epoch
        plot_result_mix(train_losses, val_losses, 'Train_Loss', 'Val_Loss', 'Training and Validation Loss', str(results_dir), 'Loss_Curve', f'Best Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch})')
        plot_result_mix(train_dices, val_dices, 'Train_Dice', 'Val_Dice', 'Training and Validation Dice', str(results_dir), 'Dice_Curve', f'Best Dice: {best_val_dice:.6f} (Epoch {best_dice_epoch})')
        plot_result_mix(train_mious, val_mious, 'Train_mIoU', 'Val_mIoU', 'Training and Validation mIoU', str(results_dir), 'mIoU_Curve', f'Best mIoU: {best_val_miou:.6f} (Epoch {best_miou_epoch})')
    
    print("Training completed!")


if __name__ == "__main__":
    main()
