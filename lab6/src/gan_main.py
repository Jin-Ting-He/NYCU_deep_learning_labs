import os
import torch
import torchvision
import argparse
import sys
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dataloader import iclevrDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluator import evaluation_model
from model import W_Generator, W_Discriminator

os.chdir(sys.path[0])

class Trainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.G = W_Generator(args.batch_size,args.im_size,args.z_size,args.g_conv_dim,args.num_cond,args.c_size)
        self.D = W_Discriminator(args.batch_size,args.im_size,args.d_conv_dim,args.num_cond)
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr = args.g_lr, betas = (0, 0.9))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr = args.d_lr, betas = (0, 0.9))

        self.train_loader = DataLoader(iclevrDataset(mode = 'train'), batch_size = args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        self.test_loader = DataLoader(iclevrDataset(mode = 'test', test_file = args.test_file), batch_size = args.test_batch_size, num_workers=8)
        self.new_test_loader = DataLoader(iclevrDataset(mode = 'test', test_file = 'new_test.json'), batch_size = args.test_batch_size, num_workers=8)

        self.best_acc_test = 0
        self.best_acc_new_test = 0

    def load_model(self):
        self.G.load_state_dict(torch.load(os.path.join(self.args.model_path, self.args.ckpt)))

    def train(self):
        for epoch in range(self.args.epochs):
            self.G.train()
            self.D.train()

            # training
            self.train_one_epoch(epoch)
            
            # testing 
            acc_list = self.test(epoch)
            print(f'test acc: {acc_list[0]:.2f}\n')
            print(f'new_test acc: {acc_list[1]:.2f}\n')

            # save model
            if(acc_list[0] >= 0.8) and (acc_list[1] >= 0.8):
                torch.save(self.G.state_dict(), os.path.join(self.args.model_path, "best_net.pth"))
                print("------ save best net ------")
            if (epoch+1) % 10 == 0:
                torch.save(self.G.state_dict(), os.path.join(self.args.model_path, f"epoch_{epoch}_"+self.args.ckpt))

    def train_one_epoch(self, epoch):
        total_loss_G = 0
        total_loss_D = 0
        num_samples = 0
        pbar = tqdm(total=len(self.train_loader), ncols=80)
        for idx, (img, label) in enumerate(self.train_loader):
            img, label = img.to(self.device), label.to(self.device)

            # ========== Train Discriminator =========== #
            d_out_real,dr1,dr2 = self.D(img, label)
            d_loss_real = -torch.mean(d_out_real)

            # apply Gumbel softmax
            z = self.tensor2var(torch.randn(self.args.batch_size, 128), self.device)
            fake_images,gf1,gf2 = self.G(z, label)
            d_out_fake,df1,df2 = self.D(fake_images, label)
            d_loss_fake = torch.mean(d_out_fake)

            # backward + optimize
            d_loss = d_loss_real + d_loss_fake

            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            d_loss_item = d_loss.item()

            # Compute gradient penalty
            alpha = torch.rand(self.args.batch_size, 1, 1, 1).to(self.device).expand_as(img)
            interpolated = Variable(alpha * img.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out,_,_ = self.D(interpolated,label)
            grad = torch.autograd.grad( outputs=out,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones(out.size()).to(self.device),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            d_loss_gp =  self.args.lambda_gp * d_loss_gp
            self.optimizer_D.zero_grad()
            d_loss_gp.backward()
            self.optimizer_D.step()
            d_loss_item += d_loss_gp.item()

            # ========== Train generator and gumbel ========== #
            # create random noise
            for _ in range(2):
                z = self.tensor2var(torch.randn(self.args.batch_size, self.args.z_size), self.device)
                fake_images,_,_ = self.G(z, label)
            
                # compute loss with fake images
                g_out_fake,_,_= self.D(fake_images, label)
                g_loss = -torch.mean(g_out_fake)
                
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()
                
            total_loss_G += g_loss.item()
            total_loss_D += d_loss_item  

            num_samples += 1

            avg_loss_G = total_loss_G / num_samples
            avg_loss_D = total_loss_D / num_samples
            pbar.set_description(f"Epoch {epoch+1} G Loss: {avg_loss_G:.4f} D Loss: {avg_loss_D:.4f}")
            pbar.update(1)
        
        pbar.close()    
        print(f'Train Epoch_{epoch} G Loss: {total_loss_G / len(self.train_loader) :.5f} D Loss: {total_loss_D / len(self.train_loader)}\n')

    def test(self, epoch=None):
        self.G.to(self.device)
        self.G.eval()

        eval_model = evaluation_model()
        gen_imgs = []
        eval_acc_list = []
        for data in [self.test_loader, self.new_test_loader]:
            gen_image=None
            eval_acc = 0
            with torch.no_grad():
                for idx,conds in enumerate(data):
                    conds=conds.to(self.device)
                    z = self.tensor2var(torch.randn(32, self.args.z_size), self.device)

                    fake_images,_,_=self.G(z,conds)
                    gen_image=fake_images
                    acc = eval_model.eval(fake_images,conds)
                    eval_acc+=acc*conds.shape[0]

            eval_acc/=len(data.dataset)
            
            gen_imgs.append(gen_image)
            eval_acc_list.append(eval_acc)
        
        if epoch != None:
            save_image(make_grid(gen_imgs[0], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'test_{epoch}_{eval_acc_list[0]:.2f}.png'))
            save_image(make_grid(gen_imgs[1], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'new_test_{epoch}_{eval_acc_list[1]:.2f}.png'))
        else:
            save_image(make_grid(gen_imgs[0], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'test_{eval_acc_list[0]:.2f}_final.png'))
            save_image(make_grid(gen_imgs[1], nrow = 8, normalize = True), os.path.join(self.args.figure_file, f'new_test_{eval_acc_list[1]:.2f}_final.png'))

        return eval_acc_list

    def tensor2var(self, x, device, grad=False):
        x=x.to(device)
        x=Variable(x,requires_grad=grad)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=800, help='Number of training epochs')
    parser.add_argument('--im-size',type=int,default=64,help="image size")
    parser.add_argument('--z-size',type=int,default=128,help="latent size")
    parser.add_argument('--g-conv-dim',type=int,default=300,help="generator convolution size")
    parser.add_argument('--d-conv-dim',type=int,default=100,help="discriminator convolution size")
    parser.add_argument('--g-lr',type=float,default=0.0001,help='initial generator learing rate')
    parser.add_argument('--d-lr',type=float,default=0.0004,help='initial discriminator learning rate')
    parser.add_argument('--c-size',type=int,default=100)
    parser.add_argument('--num-cond',type=int,default=24,help='number of conditions')
    parser.add_argument('--batch_size', default=64, help='Size of batches')
    parser.add_argument('--c_dim', default=4, help='Condition dimension')
    parser.add_argument('--lambda-gp',type=float,default=10)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--model_path', default='ckpt/GAN', help='Path to save model checkpoint')
    parser.add_argument('--test_file', default='test.json', help='Test file')
    parser.add_argument('--test_batch_size', default=32, help='Test batch size')
    parser.add_argument('--figure_file', default='figure/GAN', help='Figure file')
    parser.add_argument('--resume', default=False, help='Continue for training')
    parser.add_argument('--ckpt', default='best_net.pth', help='Checkpoint for network')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(args=args, device=device)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.figure_file, exist_ok=True)
    
    if args.test_only:
        print("------- Testing -------")
        trainer.load_model()
        acc = trainer.test()
        print(f'test acc: {acc[0]} new_test acc:{acc[1]}')
        
    else:
        trainer.train()