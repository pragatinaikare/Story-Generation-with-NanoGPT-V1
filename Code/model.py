import torch 
import torch.nn as nn
from torch.nn import functional as F
import random 
import os
from torch.utils.data import DataLoader,Dataset 
import numpy as np
import tiktoken 
from torch.nn.parallel import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"]= "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class Embedding(nn.Module):
    def __init__(self, emb_size, vocab_size, context_length, device):
        super().__init__()
        self.emb_size=emb_size
        self.vocab_size=vocab_size
        self.context_length= context_length
        self.token_embd= nn.Embedding(vocab_size, emb_size)
        self.pos_embd= nn.Embedding(context_length, emb_size)
        self.to(device)
        self.device = device
        
    def forward(self, x):
        # print(f"Input shape is {x.shape}")
        x = self.token_embd(x)
        # print(f"Token embedding shape is {x.shape}")
        pos = torch.arange(x.shape[-2]).unsqueeze(0).to(device)
        # print(f"Positional Input is {pos.shape}")
        # print(f"Positional {pos}")

        pos_embd = self.pos_embd(pos)
        # print(f"Positional Output is {pos_embd.shape}")
        x= pos_embd+x
        # print(f"Final embd output is {x.shape}")
        
        return x 


import math

class SingleHead(nn.Module):
    def __init__(self, emb_size, head_dim,context_length,dropout, device):
        super().__init__()
        
        self.emb_size=emb_size
        self.head_dim=head_dim
        
        self.Wq = nn.Linear(emb_size, head_dim,bias=False)
        self.Wk = nn.Linear(emb_size, head_dim,bias=False)
        self.Wv = nn.Linear(emb_size, head_dim,bias=False)
        
        self.mask=torch.tril(torch.ones(context_length,context_length)).view(1,context_length,context_length).to(device)
        self.drop = nn.Dropout(dropout)
        self.to(device)
                                
    def forward(self,x):
        # print(x.shape)
        T= x.shape[-2]  
        q = self.Wq(x)
        k =  self.Wk(x)
        v=  self.Wv(x)
                                
        att_weights = q @ k.transpose(1,2) / math.sqrt(self.head_dim)        
        att_weights = att_weights.masked_fill( self.mask[:,:T,:T]==0,-float("inf"))
        att_weights = F.softmax(att_weights,dim = -1)
        att_weights = self.drop(att_weights)
        # print(f"Attention Weights -  {att_weights.shape}")
        out = att_weights @ v
        # print(f"Output {out.shape}")
        return out
    


class MultiHead(nn.Module):
    def __init__(self,emb_size,num_heads,context_length,dropout, device):
        super().__init__() 
        self.heads = nn.ModuleList([SingleHead(emb_size,emb_size//num_heads,context_length,dropout=dropout, device=device) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)
        self.to(device)
        
    def forward(self,x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # print(x.shape)
        x = self.proj(x) 
        x = self.drop(x)
        
        return  x
        


class FeedForward(nn.Module):
    def __init__(self,emb_size,dff, dropout, device):
        super().__init__() 
        
        self.fc1 = nn.Linear(emb_size, dff)
        self.fc2 = nn.Linear(dff, emb_size)
        self.drop = nn.Dropout(dropout)
        self.to(device)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x
    


class Decoder(nn.Module):
    def __init__(self,emb_size,dff,num_heads,context_length,dropout, device):
        super().__init__() 
        
        self.multihead = MultiHead(emb_size,num_heads,context_length,dropout, device)
        self.feedforward = FeedForward(emb_size,dff, dropout, device)      
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.to(device)
        
    def forward(self,x):
        
        out = self.multihead(x)
        out = F.relu(self.ln1(out)+x)
        out1 = self.feedforward(out)
        out1 = F.relu(self.ln2(out1)+out)
        return out1
    


class DecoderStack(nn.Module):
    def __init__(self,emb_size,dff,num_heads,context_length,N, dropout, device):
        super().__init__() 
        
        self.decstack = nn.ModuleList([Decoder(emb_size,dff,num_heads,context_length,dropout, device) for _ in range(N)])
        self.to(device)
        
    def forward(self,x):
        for decoder in self.decstack:
            x= decoder(x)
        
        return x
    
class GPT(nn.Module):
    def __init__(self,emb_size,dff,num_heads,context_length,N, vocab_size, dropout, device):
        super().__init__() 
        
        self.embd =  Embedding(emb_size,vocab_size,context_length, device)
        self.decoderstack = DecoderStack(emb_size,dff,num_heads,context_length,N, dropout, device)
        self.out = nn.Linear(emb_size,vocab_size, device)
        self.to(device)
        
    def forward(self,x):
        x = self.embd(x)
        x= self.decoderstack(x)
        x = F.relu(x)
        x = self.out(x)
        return x 
