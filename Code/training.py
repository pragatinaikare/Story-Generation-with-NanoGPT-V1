


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

os.environ["CUDA_VISIBLE_DEVICES"]= "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")

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


class CustomDataset(Dataset):
    def __init__(self, dataset,context_length,num_of_stories,batch_size,n_itr, start='\n<|startofstory|>\n', end='\n<|endofstory|>\n'):
        self.dataset = dataset 
        self.context_length = context_length  
        self.num_of_stories = num_of_stories  
        self.batch_size = batch_size 
        self.length = len(dataset)  
        self.start = start 
        self.end = end 
        self.n_itr = n_itr
        self.tokenizer = tiktoken.get_encoding("gpt2") 

    def __len__(self):
        return self.batch_size*self.n_itr
    
    def __getitem__(self, idx):
        n = list(np.random.randint(0,self.length,size= self.num_of_stories))
        text = ''
        
        for i in n:
            
            text += self.start + self.dataset[i] + self.end 
        
            
        tokens = self.tokenizer.encode(text)
        while len(tokens) < 513:
            print("")
            text += text
        tokens = self.tokenizer.encode(text)
            
        start = random.randint(0, len(tokens) - self.context_length-1) 
        inputs = tokens[start: start+self.context_length]
        outputs = tokens[start+1: start+self.context_length+1]
        inputs = torch.tensor(inputs, dtype = torch.long)
        outputs = torch.tensor(outputs, dtype = torch.long)
        assert inputs.shape[-1] == self.context_length, f"Input's last dimension must be {self.context_length}, but got {inputs.shape[-1]}"
        assert outputs.shape[-1] == self.context_length, f"Output's last dimension must be {self.context_length}, but got {outputs.shape[-1]}"

        return inputs,outputs
    
train_dataset =CustomDataset(ds['train']['text'], context_length=context_length ,num_of_stories=200 ,batch_size = batch_size, n_itr =3000)

val_dataset = CustomDataset(ds['validation']['text'], context_length=context_length ,num_of_stories=200 ,batch_size = batch_size, n_itr=200)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size)



gpt =  GPT(emb_size,dff,num_heads,context_length,N, vocab_size, dropout, device)  


num_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)

num_params_million = num_params / 1e6

print(f"Number of parameters in the model: {num_params_million:.2f} million")


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(gpt.parameters(),lr=lr)

from tqdm import tqdm

def calculate_loss(dataloader):
    val_loss = 0
    gpt.eval() 
    with torch.inference_mode():
        for inputs, outputs in val_dataloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            pred = gpt(inputs)
            loss = loss_fn(pred.view(-1, vocab_size), outputs.view(-1))
            val_loss += loss.item()
    gpt.train() 
    return val_loss / len(dataloader)


def training(n_epochs):
    total_loss = 0
    losses = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        for inputs, outputs in train_dataloader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            
            pred = gpt(inputs)
            loss = loss_fn(pred.view(-1, vocab_size), outputs.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Avg training loss for epoch {epoch + 1}: {avg_train_loss}")
        total_loss = 0
        
        val_loss = calculate_loss(val_dataloader)
        print(f"Avg validation loss after epoch {epoch + 1}: {val_loss}")
        model_save_path = f'gpt_model_{epoch}.pth'

        torch.save(gpt.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return losses

print('Training..............')
losses = training(n_epochs=25)



