import torch.nn as nn
import torch
import math

#TODO1
class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(Attention, self).__init__()
        self.d = dim // num_heads
        self.q, self.k, self.v = nn.Linear(dim, self.d), nn.Linear(dim, self.d), nn.Linear(dim, self.d)
        self.dropout = nn.Dropout(p=attn_drop)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.d**0.5, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention_list = nn.ModuleList([
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop),
            Attention(dim, num_heads, attn_drop)])
        self.out_layer = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.attention_list[0](x)
        out = torch.cat((out, self.attention_list[1](x)), axis=-1)
        out = torch.cat((out, self.attention_list[2](x)), axis=-1)
        out = torch.cat((out, self.attention_list[3](x)), axis=-1)
        out = torch.cat((out, self.attention_list[4](x)), axis=-1)
        out = torch.cat((out, self.attention_list[5](x)), axis=-1)
        out = torch.cat((out, self.attention_list[6](x)), axis=-1)
        out = torch.cat((out, self.attention_list[7](x)), axis=-1)
        out = torch.cat((out, self.attention_list[8](x)), axis=-1)
        out = torch.cat((out, self.attention_list[9](x)), axis=-1)
        out = torch.cat((out, self.attention_list[10](x)), axis=-1)
        out = torch.cat((out, self.attention_list[11](x)), axis=-1)
        out = torch.cat((out, self.attention_list[12](x)), axis=-1)
        out = torch.cat((out, self.attention_list[13](x)), axis=-1)
        out = torch.cat((out, self.attention_list[14](x)), axis=-1)
        out = torch.cat((out, self.attention_list[15](x)), axis=-1)
        out = self.out_layer(out)
        return out

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    