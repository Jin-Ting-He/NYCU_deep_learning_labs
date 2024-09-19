import os
import torch
import torchvision
import argparse
import sys
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dataloader import iclevrDataset
from model import NoisePredictor
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluator import evaluation_model

os.chdir(sys.path[0])

class Trainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.train_loader = DataLoader(iclevrDataset(mode = 'train'), batch_size = args.batch_size, shuffle=True)
        self.test_loader = DataLoader(iclevrDataset(mode = 'test', test_file = args.test_file), batch_size = args.test_batch_size, num_workers=8)
        self.new_test_loader = DataLoader(iclevrDataset(mode = 'test', test_file = 'new_test.json'), batch_size = args.test_batch_size, num_workers=8)
        self.model = NoisePredictor(args)
        self.noise_scheduler = DDPMScheduler(args.timesteps)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.best_acc_test = 0
        self.best_acc_new_test = 0

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, self.args.ckpt)))

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.args.epochs):
            self.model.train()

            # training
            self.train_one_epoch(epoch)
            
            # testing 
            acc_list = self.test(epoch)
            print(f'test acc: {acc_list[0]:.2f}\n')
            print(f'new_test acc: {acc_list[1]:.2f}\n')

            # save model
            if(acc_list[0] >= 0.8) and (acc_list[1] >= 0.8):
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "best_net.pth"))
                print("------ save best net ------")
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, f"epoch_{epoch}_"+self.args.ckpt))

    def train_one_epoch(self, epoch):
        total_loss = 0
        num_samples = 0
        pbar = tqdm(total=len(self.train_loader), ncols=80)
        for idx, (img, label) in enumerate(self.train_loader):
            img, label = img.to(self.device), label.to(self.device)
            
            noise = torch.randn_like(img)
            timesteps = torch.randint(0, args.timesteps - 1, (img.shape[0],)).long().to(self.device)
            noisy_img = self.noise_scheduler.add_noise(img, noise, timesteps)
            
            # Get prediction
            pred = self.model(noisy_img, timesteps, label)
            
            # Calculate loss
            loss = self.criterion(pred, noise)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.detach().item()
            num_samples += 1
            avg_loss = total_loss / num_samples

            pbar.set_description(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            pbar.update(1)

        pbar.close()    
        print(f'Train Epoch_{epoch} loss: {total_loss / len(self.train_loader) :.5f}\n')

    def test(self, epoch = None):
        self.model.to(self.device)
        self.model.eval()
        
        score = evaluation_model()
        gen_imgs = []
        eval_acc_list = []

        for data in [self.test_loader, self.new_test_loader]:
            with torch.no_grad():
                for label in data:
                    label = label.to(self.device)
                    noise = torch.randn(label.shape[0], 3, 64, 64).to(self.device)
                    print(label.shape)
                    for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                        # Get model predict
                        residual = self.model(noise, t, label)
                        
                        # Update sample with step
                        noise = self.noise_scheduler.step(residual, t, noise).prev_sample
                    
                    gen_image = noise
                    
                eval_acc = score.eval(gen_image, label)
                gen_imgs.append(gen_image)
                eval_acc_list.append(eval_acc)

        if epoch != None:
            save_image(make_grid(gen_imgs[0], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'test_{epoch}_{eval_acc_list[0]:.2f}.png'))
            save_image(make_grid(gen_imgs[1], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'new_test_{epoch}_{eval_acc_list[1]:.2f}.png'))
        else:
            save_image(make_grid(gen_imgs[0], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'test_{eval_acc_list[0]:.2f}_final.png'))
            save_image(make_grid(gen_imgs[1], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'new_test_{eval_acc_list[1]:.2f}_final.png'))
        
        return eval_acc_list
    
    def demo_sample(self):
        self.model.to(self.device)
        self.model.eval()
        label = torch.zeros(1, 24).to(self.device)
        noise = torch.randn(1, 3, 64, 64).to(self.device)
        label[0][7] = 1
        label[0][9] = 1
        label[0][22] = 1
        gen_imgs = []
        with torch.no_grad():
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                # Get model predict
                residual = self.model(noise, t, label)
                
                # Update sample with step
                noise = self.noise_scheduler.step(residual, t, noise).prev_sample
                
                if (i+1 >= 700) and ((i+1)%30 == 0):
                    # print(noise.shape)
                    gen_imgs.append(noise[0])
            gen_imgs.append(noise[0])
        gen_imgs = torch.stack(gen_imgs)

        last_img = gen_imgs[-1]
        min_val = last_img.min()
        max_val = last_img.max()

        # Normalize all images using min and max from the last image
        normalized_imgs = (gen_imgs - min_val) / (max_val - min_val)

        # Create a grid of images
        grid = make_grid(normalized_imgs, nrow=len(gen_imgs))

        # Save the grid as an image file
        save_image(grid, os.path.join(self.args.figure_file, 'denoising_process.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', default=28, help='Size of batches')
    parser.add_argument('--lr', default=0.0002, help='Learning rate')
    parser.add_argument('--c_dim', default=4, help='Condition dimension')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_path', default='ckpt/DDPM', help='Path to save model checkpoint')
    parser.add_argument('--timesteps', default=1000, help='Time step for diffusion')
    parser.add_argument('--test_file', default='test.json', help='Test file')
    parser.add_argument('--test_batch_size', default=32, help='Test batch size')
    parser.add_argument('--figure_file', default='figure/DDPM', help='Figure file')
    parser.add_argument('--resume', default=False, help='Continue for training')
    parser.add_argument('--ckpt', default='best_net.pth', help='Checkpoint for network')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(args=args, device=device)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.figure_file, exist_ok=True)
    
    if args.test_only:
        trainer.load_model()
        trainer.demo_sample()
        acc = trainer.test()
        print(f'test acc: {acc[0]} new_test acc:{acc[1]}')
    
    else:
        trainer.train()