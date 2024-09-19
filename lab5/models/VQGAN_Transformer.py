import torch 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        # loaded_state_dict = torch.load(load_ckpt_path)
        # print("Loaded keys:", loaded_state_dict.keys())
        # print("Model keys:", self.transformer.state_dict().keys())
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def generate_random_mask(self, z_indices):
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)

        for i in range(z_indices.shape[0]):
            r = math.floor(np.random.uniform() * z_indices.shape[1])
            
            sample = torch.rand(z_indices.shape[1], device=z_indices.device).topk(r).indices
            
            mask[i].scatter_(dim=0, index=sample, value=True)
        
        return mask

##TODO2 step1-3:            
    def forward(self, x):
        
        # z_indices=None #ground truth
        # logits = None  #transformer predict the probability of tokens
        # raise Exception('TODO2 step1-3!')
        # return logits, z_indices
        _, z_indices = self.encode_to_z(x)

        r = math.floor(np.random.uniform() * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)
        # mask = self.generate_random_mask(z_indices)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = (~mask) * z_indices + (mask) * masked_indices

        target = z_indices
        logits = self.transformer(a_indices)
        return logits, target

##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, ratio):

        masked_indices = torch.where(mask_b, 
                                 torch.full_like(z_indices, self.mask_token_id),
                                 z_indices)

        logits = self.transformer(masked_indices)

        logits = F.softmax(logits, dim=-1)
        z_indices_predict = torch.argmax(logits, dim=-1)
        z_indices_predict = torch.where(mask_b, z_indices_predict, z_indices)

        max_probabilities, _ = torch.max(logits, dim=-1)

        g = -torch.empty_like(max_probabilities).exponential_().log()  # Gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = max_probabilities + temperature * g

        inf_mask = torch.full_like(confidence, float('inf'))  # Create a tensor of infinities
        confidence = torch.where(mask_b, confidence, inf_mask)
        
        n_mask = math.ceil(self.gamma(ratio) * mask_b.sum())

        flat_confidence = confidence.flatten()
        _, indices_to_mask = torch.topk(flat_confidence, n_mask, largest=False, sorted=True)
        new_mask_b = torch.zeros_like(mask_b, dtype=torch.bool)
        new_mask_b.view(-1)[indices_to_mask] = True
        return z_indices_predict, new_mask_b
        raise Exception('TODO3 step1-1!')
        logits = self.transformer(None)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = None

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = None

        ratio=None 
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = None  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_bc=None
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
