import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from dataloader import BufferflyMothLoader
from VGG19 import VGG19
import torch.optim as optim
import argparse
import sys
import numpy as np
import random
from tqdm import tqdm
import torch.nn.init as init
from pytorch_optimizer import create_optimizer
from optimizer import Optimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def save_learning_curve(vgg19_train_accs, vgg19_val_accs):
    vgg19_train_accs = [acc.cpu().item() for acc in vgg19_train_accs]
    vgg19_val_accs = [acc.cpu().item() for acc in vgg19_val_accs]
    plt.figure(figsize=(10, 5))
    plt.plot(vgg19_train_accs, label='Training Acc')
    plt.plot(vgg19_val_accs, label='Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('vgg19_accuracy_curve.png')
    plt.close()

if __name__ == "__main__":
    num_epochs = 200
    batch_size = 16
    model_save_path = 'vgg19.pth'
     
    model = VGG19(100).cuda()

    train_dataset = BufferflyMothLoader(root='dataset', mode='train')
    val_dataset = BufferflyMothLoader(root='dataset', mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,pin_memory=True,num_workers=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Optimizer(model)
    # optimizer = create_optimizer(
    #     model,
    #     'ranger21',
    #     lr=1e-3,
    #     weight_decay=1e-3,
    #     num_iterations=200
    # )

    grad_clip = 20

    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_corrects = 0
        total_samples = 0 
        running_loss = 0.0
        pbar = tqdm(total=len(train_dataset), ncols=80)
        for images, labels in train_dataloader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()  
            outputs = model(images)  
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)  
            loss.backward()  

            optimizer.step() 
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            running_loss += loss.item()
            current_loss = running_loss / (pbar.n + 1)
            pbar.set_description(f"Epoch {epoch+1} Loss: {current_loss:.4f} lr: {optimizer.get_lr():.6f}")
            pbar.update(images.size(0))
        pbar.close()
        optimizer.lr_schedule()
        train_acc = running_corrects.double() / total_samples
        train_accs.append(train_acc.cpu())
        print(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}")
        # val
        model.eval() 
        val_running_corrects = 0
        val_total_samples = 0
        with torch.no_grad(): 
            for images, labels in val_dataloader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += labels.size(0)

        val_acc = val_running_corrects.double() / val_total_samples
        val_accs.append(val_acc.cpu())
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")

        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        save_learning_curve(train_accs, val_accs)
        np.savez('vgg19_accuracy.npz', array1 = np.array(train_accs), array2 = np.array(val_accs))