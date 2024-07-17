#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
from torch.nn import functional as F
import random 
import os
from torch.utils.data import DataLoader,Dataset 
import numpy as np
import tiktoken 
from torch.nn.parallel import DataParallel
from model import GPT

os.environ["CUDA_VISIBLE_DEVICES"]= "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2") 
context_length = 512 
batch_size =16
emb_size= 256
num_heads = 8
dff= 512
decoder_blocks = N = 4
vocab_size = tokenizer.n_vocab
lr = 3e-4
dropout = 0.2
    

gpt =  GPT(emb_size,dff,num_heads,context_length,N, vocab_size, dropout, device)  

gpt.load_state_dict(torch.load("gpt_model_24.pth", map_location=device))

# Move the model to the appropriate device (CPU or GPU)
gpt.to(device)

num_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)

num_params_million = num_params / 1e6

print(f"Number of parameters in the model: {num_params_million:.2f} million")


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(gpt.parameters(),lr=lr)



def next_word_pred(start, gpt):
    start = "\n<|startofstory|>\n"+start
    start_tokens = tokenizer.encode(start)
    inp = torch.tensor(start_tokens).unsqueeze(0).to(device)
    while inp[0][-9:].tolist() != [198, 27, 91, 437, 1659, 13571, 91, 29, 198]:
        with torch.inference_mode():
            prob = gpt(inp[:,-context_length:].to(device))
        probs = F.softmax(prob[:,-1,:],dim=-1)
        next_token = torch.multinomial(probs,num_samples=1)
        inp = torch.cat([inp, next_token], dim=-1)
        tokenizer.decode(inp.tolist()[0])
        decoded_string = tokenizer.decode(inp.tolist()[0])
        decoded_string = decoded_string.replace('\n<|startofstory|>\n', '')
        decoded_string = decoded_string.replace('\n<|endofstory|>\n', '')
    return decoded_string


# In[7]:


user_input = "Once upon a time, there was a little boy named Tim"


# In[8]:


out = next_word_pred(user_input, gpt)
print(out)


# In[ ]:




