from diffusers import DDPMScheduler, UNet2DModel
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.nn import functional as F 

class NoisePredictor(nn.Module):
    def __init__(self, args, num_class = 24, embed_size = 512):
        super().__init__() 
        
        self.model = UNet2DModel(
            sample_size = 64,
            in_channels =  3,
            out_channels = 3,
            layers_per_block = 2,
            class_embed_type = None,
            block_out_channels = (128, 256, 256, 512, 512, 1024),
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",    
                "DownBlock2D",    
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D", 
            ), 
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",         
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        self.model.class_embedding = nn.Linear(num_class, embed_size)
    
    def forward(self, x, t, y):
        return self.model(x, t, class_labels = y).sample 
    

class W_Generator(nn.Module):
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64,num_cond=24,c_size=100):
        super(W_Generator, self).__init__()
        self.imsize = image_size
        self.c_size=c_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,c_size),
            nn.ReLU(inplace=True)
        )
        repeat_num = int(np.log2(self.imsize)) - 3  # 6-3=3
        mult = 2 ** repeat_num # 8
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim+c_size, conv_dim * mult, kernel_size=4),   # 1 -> 4
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(),
        )
        curr_dim = conv_dim * mult
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),    # 4 -> 8(6-2+3+0+1)
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
         )
        curr_dim = int(curr_dim / 2)
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
        )
        curr_dim = int(curr_dim / 2)
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(),
        )
        curr_dim = int(curr_dim / 2)
        self.last = nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1)# 32 -> 64
    
    def forward(self, z,c):
        p1,p2 = 2,2
        z = z.view(z.size(0), z.size(1), 1, 1) 
        c_embd=self.embed_c(c).reshape(-1,self.c_size,1,1) 
        z=torch.cat((z,c_embd),dim=1) 
        out=self.l1(z)  
        out=self.l2(out)    
        out=self.l3(out)    
        out=self.l4(out)      
        out=self.last(out) 
        return out, p1, p2


class W_Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64,num_cond=24):
        super(W_Discriminator, self).__init__()
        self.imsize = image_size
        self.embed_c=nn.Sequential(
            nn.Linear(num_cond,self.imsize*self.imsize),
            nn.ReLU(inplace=True)
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(4, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )
        curr_dim = curr_dim * 2
        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )
        curr_dim = curr_dim * 2
        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )
        curr_dim = curr_dim*2
        self.last = nn.Conv2d(curr_dim, 1, 4)

    def forward(self, x,c):

        p1, p2 = 1,1
        c_embd=self.embed_c(c).reshape(-1,1,self.imsize,self.imsize)
        x=torch.cat((x,c_embd),dim=1)   
        out = self.l1(x)   
        out = self.l2(out) 
        out = self.l3(out)  
        out=self.l4(out)    
        out=self.last(out)  
        return out.squeeze(), p1, p2