import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from transformer import Decoder
import io

torch.manual_seed(1337)

batch_size = 64
block_size = 256
epoches = 5000
eval_epoche = 500
estimate_eval_epoches = 200
lr = 3e-4 
n_embd = 384
n_heads = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_name = 'input/input.txt'
with open(file_name, 'r', encoding= 'utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

itoc = {i: c for i, c in enumerate(chars)}
ctoi = {c: i for i, c in enumerate(chars)}
encode = lambda l: [ctoi[i] for i in l] #take in a string
decode = lambda l: "".join([itoc[i] for i in l]) # take in a list of intiegers

m = Decoder(vocab_size, n_embd, block_size, n_heads, n_layer, dropout, device)
with open('output/model.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())
state_dict = torch.load(buffer, map_location=torch.device('cpu'))
m.load_state_dict(state_dict)
res = decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=10000, block_size=block_size).tolist()[0])
print(res)


