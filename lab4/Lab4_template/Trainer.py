import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def plot_loss_curve(train_loss, val_loss, epochs, kl_name):
    plt.plot(range(1, epochs + 1), train_loss, label='train loss')
    plt.plot(range(1, epochs + 1), val_loss, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(kl_name + ' Loss Curve')
    plt.legend()
    plt.savefig(kl_name+'/loss_curve.png')
    plt.close()

def plot_teacher_forcing_ratio(data, epochs, kl_name):
    plt.plot(range(1, epochs+1), data, label='teacher forcing ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.title('Teacher Forcing Ratio')
    plt.legend()
    plt.savefig(kl_name+"/teacher_forcing_ratio.png")
    plt.close()

def plot_val_psnr(data, frame_num, kl_name):
    plt.plot(range(1, frame_num), data, label='Val PSNR')
    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.title('Val PSNR Per Frame')
    plt.legend()
    plt.savefig(kl_name+"/val_psnr_per_frame.png")
    plt.close()

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.current_epoch = current_epoch
        self.n_iter = args.num_epoch
        self.start = 0.0
        self.stop = 1.0
        self.n_cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.type = args.kl_anneal_type
    def update(self):
        self.current_epoch += 1
        
    def get_beta(self):
        if self.type == 'Monotonic':
            return self.monotonic_kl_annealing(args, 0)
        elif self.type == 'Cyclical':
            return self.cyclical_kl_annealing(args, 0)
        else:
            return 1

    def monotonic_kl_annealing(self, args, start_epoch):
        return min(1, 0.1 * self.current_epoch)

    
    def cyclical_kl_annealing(self, args, start_epoch):
        weight = 1.0 / (self.n_cycle * self.ratio)
        return min(1, weight * ((self.current_epoch - start_epoch) % (self.n_cycle+1)))
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch, eta_min=1e-6)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.best_psnr = 0

        self.train_losses = []
        self.val_losses = []
        self.val_psnr = []
        self.teacher_forcing_ratios = []

        self.warm_up_flag = True

    def forward(self, img, label, mode, target_img = None):
        feature_img = self.frame_transformation(img)
        feature_label = self.label_transformation(label)
        if mode == "train":
            feature_target_img = self.frame_transformation(target_img)
            z, mu, logvar = self.Gaussian_Predictor(feature_target_img, feature_label)
            decoded = self.Decoder_Fusion(feature_img, feature_label, z)
            output = self.Generator(decoded)
            return output, mu, logvar
        else:
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W, device='cuda')
            decoded = self.Decoder_Fusion(feature_img, feature_label, z)
            output = self.Generator(decoded)
            return output
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            # warm up on first epoch
            # if(i==0) and self.warm_up_flag:
            #     for param_group in self.optim.param_groups:
            #         param_group['lr'] = 1e-4
            # elif self.warm_up_flag:
            #     for param_group in self.optim.param_groups:
            #         param_group['lr'] = self.args.lr
            #     self.warm_up_flag = False

            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            total_loss = 0
            idx = 0
            for (img, label) in (pbar := tqdm(train_loader)):
                idx +=1
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()

                total_loss += loss.detach().cpu()

                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, round(beta,4)), pbar, total_loss/idx, lr=self.optim.param_groups[0]['lr'])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, round(beta,4)), pbar, total_loss/idx, lr=self.optim.param_groups[0]['lr'])


            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            self.train_losses.append(total_loss/len(train_loader))
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        
        plot_loss_curve(self.train_losses, self.val_losses, self.args.num_epoch, self.args.kl_anneal_type)
        plot_teacher_forcing_ratio(self.teacher_forcing_ratios, self.args.num_epoch, self.args.kl_anneal_type)
        np.savez(self.args.kl_anneal_type+'/train_val_loss.npz', array1 = np.array(self.train_losses), array2 = np.array(self.val_losses))
    @torch.no_grad()
    def eval(self):
        val_totoal_loss = 0
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0], psnr=psnr.detach())

            if psnr > self.best_psnr:
                self.best_psnr = psnr.detach()
                self.save(os.path.join(self.args.save_root, f"best.ckpt"))

            val_totoal_loss += loss.detach().cpu()
            plot_val_psnr(self.val_psnr, self.args.val_vi_len, self.args.kl_anneal_type)
            self.val_psnr = []

        self.val_losses.append(val_totoal_loss/len(val_loader))

    def training_one_step(self, imgs, labels, adapt_TeacherForcing):
        self.optim.zero_grad()
        batch_size, seq_len, c, h, w = imgs.shape
        
        train_total_loss = 0
        output = imgs[:, 0, :, :, :]
        
        for t in range(1, seq_len):
            current_input = imgs[:, t-1, :, :, :]*0.3 + output*0.7 if adapt_TeacherForcing else output
            output, mu, logvar = self.forward(current_input, labels[:, t, :, :, :], "train", target_img = imgs[:, t, :, :, :])
            kl_loss = kl_criterion(mu, logvar, batch_size) * self.kl_annealing.get_beta()
            reconstruction_loss = self.mse_criterion(output, imgs[:, t, :, :, :])
            loss = kl_loss + reconstruction_loss
            train_total_loss += loss

        train_total_loss.backward()
        self.optimizer_step()
        
        avg_train_loss = train_total_loss.detach() / (seq_len-1)
        return avg_train_loss
    
    def val_one_step(self, imgs, labels):
        batch_size, seq_len, c, h, w = imgs.shape
        total_loss = 0
        total_psnr = 0

        # Initialize the current input with the first frame of the sequence
        current_input = imgs[:, 0, :, :, :]

        for t in range(1, seq_len):
            output = self.forward(current_input, labels[:, t, :, :, :], "val")

            reconstruction_loss = self.mse_criterion(output, imgs[:, t, :, :, :])
            loss = reconstruction_loss
            total_loss += loss

            psnr = Generate_PSNR(output, imgs[:, t, :, :, :])
            total_psnr += psnr

            # Update current_input to the last generated output for the next time step
            # Make sure to detach the output from the graph to prevent gradients from accumulating
            current_input = output.detach()

            self.val_psnr.append(psnr.cpu())

        average_loss = total_loss.detach() / seq_len
        average_psnr = total_psnr.detach() / seq_len

        # Return both the average loss and average PSNR for the validation step
        return average_loss, average_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr -= self.args.tfr_d_step
            self.tfr = max(self.tfr, 0)  
        self.teacher_forcing_ratios.append(self.tfr)

    def tqdm_bar(self, mode, pbar, loss, lr, psnr=0):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}, PSNR:{round(float(psnr),4)}, Best:{round(float(self.best_psnr),4)}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=1e-4,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=16)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="Monotonic or Cyclical")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=5,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
