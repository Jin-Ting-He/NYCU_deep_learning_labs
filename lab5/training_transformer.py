import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        # self.optim = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs(args.checkpoint_path, exist_ok=True)

    def train_one_epoch(self, train_loader):
        step = 0
        with tqdm(range(len(train_loader))) as pbar:
            for i, imgs in zip(pbar, train_loader):
                imgs = imgs.to(device=args.device)
                # print(imgs.shape)
                logits, target = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
                if step % args.accum_grad == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                step += 1
                pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                pbar.set_description(f"lr: {self.optim.param_groups[0]['lr']:.6f}")
                pbar.update(0)
            self.scheduler.step()

    def eval_one_epoch(self, val_dataset):
        total_loss = 0.0
        total_samples = 0
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient computation during evaluation
            with tqdm(val_dataset, desc="Validating", leave=False) as pbar:
                for imgs in pbar:
                    imgs = imgs.to(device=args.device)
                    logits, target = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    total_loss += loss.item() * imgs.size(0)
                    total_samples += imgs.size(0)
                    pbar.set_postfix(Validation_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
        avg_loss = total_loss / total_samples
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.transformer.parameters(), lr=args.learning_rate, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        return optimizer,scheduler
        # return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/v5', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    best_val_loss = 10
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch {epoch}:")
        train_transformer.train_one_epoch(train_loader=train_loader)
        if epoch % args.ckpt_interval == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.checkpoint_path, f"transformer_epoch_{epoch}.pt"))
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.checkpoint_path, "transformer_current.pt"))
        val_loss = train_transformer.eval_one_epoch(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.checkpoint_path, "transformer_best.pt"))
        print(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")