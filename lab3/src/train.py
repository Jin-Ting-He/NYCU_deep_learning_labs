import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.unet import UNet
from models.resnet34_unet import UNetPlusResNet34
from oxford_pet import load_dataset
from utils import dice_score, plot_dice_score_curve
from importlib import import_module
from evaluate import evaluate
import torch.optim.lr_scheduler as lr_scheduler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class Optimizer:
    def __init__(self, target):
        # create optimizer
        trainable = target.parameters()
        optimizer_name = 'AdamW' 
        lr = 1e-3 
        weight_decay = 5e-3
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr, weight_decay=weight_decay)
        # create scheduler
        T_max = 50  
        eta_min = 1e-6  
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
            
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()

def dice_loss(preds, labels, smooth=1e-6):
    """Dice Loss"""
    preds = preds.contiguous().float()
    labels = labels.contiguous().float()

    intersection = (preds * labels).sum(dim=[2, 3])
    dice_coef = (2. * intersection + smooth) / (preds.sum(dim=[2, 3]) + labels.sum(dim=[2, 3]) + smooth)
    dice_loss = 1 - dice_coef

    return dice_loss.mean()

def train(args, model, model_save_path):
    train_dataset = load_dataset(data_path=args.data_path, mode = 'train')    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,num_workers=16, shuffle=True)
    valid_dataset = load_dataset(data_path=args.data_path, mode='valid') 
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    model = model.cuda()

    optimizer = Optimizer(model)

    training_dice_scores = []  
    validation_dice_scores = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(total=len(train_dataset), ncols=80)
        running_loss = 0.0
        total_dice = 0.0
        count = 0
        for sample in train_dataloader:
            image, mask = sample['image'].cuda(), sample['mask'].cuda()
            optimizer.zero_grad() 
            output = model(image)
            loss = loss_fn(output, mask) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            current_loss = running_loss / (pbar.n + 1)

            preds = torch.sigmoid(output) > 0.5 
            total_dice += dice_score(preds, mask)
            count += 1

            pbar.set_description(f"Epoch {epoch+1} Loss: {current_loss:.4f} lr: {optimizer.get_lr():.6f}")
            pbar.update(image.size(0))

        pbar.close()
        optimizer.lr_schedule()
        avg_training_dice = total_dice / count
        training_dice_scores.append(avg_training_dice.item())
        print(f'Epoch {epoch + 1}: Average Training Dice Score: {avg_training_dice.item():.4f}')

        # Validation
        avg_validation_dice = evaluate(model, valid_dataloader, len(valid_dataset))
        validation_dice_scores.append(avg_validation_dice.item())
        print(f'Epoch {epoch + 1}: Average Validation Dice Score: {avg_validation_dice.item():.4f}')

        torch.save(model.state_dict(), model_save_path)

    plot_dice_score_curve(training_dice_scores, validation_dice_scores, args.epochs, args.model_name)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_name', type=str, default=' ', help='model name')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()

    print("########### UNet Training ###########")
    args.model_name = 'UNet'
    model = UNet(3,1)
    model_save_path = "saved_models/DL_Lab3_UNet_312551065_何勁廷.pth"
    train(args, model, model_save_path)
    
    print("########### ResNet34+UNet Training ###########")
    args.model_name = 'ResNet34+UNet'
    model = UNetPlusResNet34(3,1)
    model_save_path = "saved_models/DL_Lab3_ResNet34_UNet_312551065_何勁廷.pth"
    train(args, model, model_save_path)