import torch
import torch.nn as nn
from torch.nn import functional as F
import time
torch.manual_seed(1337)

def get_batch(split):
    d = train if split == 'train' else val
    rand = torch.randint(len(d) - block_size, (batch_size,), device = device)
    x = torch.stack([d[i:i+block_size] for i in rand])
    y = torch.stack([d[i+1:i+block_size+1] for i in rand])
    return x, y

def print_batch(x, y):
    for j, b in enumerate(x):
        print(f"===")
        print(f"Batch Number: {j}")
    for i in range(block_size):
        print(f'When input is {x[j][:i+1]}, output is {y[j][i]}')

@torch.no_grad
def estimate_eval():
  m.eval()
  loss_dict = {'train': 0, 'val': 0}
  for split in ['train', 'val']:
    loss_sum = 0
    for i in range(estimate_eval_epoches):
        xa, xb = get_batch(split)
        _, loss = m(xa, xb)
        loss_sum += loss.item()
    avg_loss = loss_sum / estimate_eval_epoches
    loss_dict[split] = avg_loss
  print(f"Step {steps}: train loss = {loss_dict['train']}, val loss = {loss_dict['val']}")
  m.train()

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size, n_embd
        self.pos_embedding_table = nn.Embedding(block_size, n_embd) # block_size, n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)], 
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # n_embd, vocab_size
  
    def forward(self, idx, y=None): # idx = B, T; y = B, T
        B, T = idx.shape # B, T
        tok_embd = self.token_embedding_table(idx) # B, T, n_embd
        pos_embd = self.pos_embedding_table(torch.arange(T, device=device)) # T, n_embd
        x = tok_embd + pos_embd # B, T, n_embd
        sa_block = self.blocks(x)
        logits = self.lm_head(sa_block) # B, T, vocab_size

        if y == None: 
            loss = None

        else:
            B, T, C = logits.shape # B, T, vocab_size
            logits = logits.view(B*T, C) # BxT, vocab_size
            y = y.view(B*T,) # BxT,
            loss = F.cross_entropy(logits, y) # BxT, vocab_size

        return logits, loss # B, T, vocab_size; BxT, vocab_size
  
    def generate(self, idx, max_tokens):
        for i in range(max_tokens):
            idx_cond = idx[:, -block_size:] # B, T (max block_size)
            logits, _ = self(idx_cond) # B, T, vocab_size; BxT, vocab_size
            probs = logits[:,-1,:] # B, vocab_size
            probs = F.softmax(probs, dim=1) # B, vocab_size
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx
  
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # n_embd, head_size
        self.query = nn.Linear(n_embd, head_size, bias=False) # n_embd, head_size
        self.value = nn.Linear(n_embd, head_size, bias=False) # n_embd, head_size
        self.register_buffer('tril', torch.tril(torch.tril(torch.ones(block_size, block_size)))) # block_size, block_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T, C = idx.shape # B, T, n_embd
        k = self.key(idx) # B, T, head_size
        q = self.query(idx) # B, T, head_size
        v = self.value(idx) # B, T, head_size
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)
        out = wei @ v # B, T, head_size
        return out # B, T, head_size

class MultiheadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for i in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
  
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ma = MultiheadAttention(n_embd, n_heads)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = self.ma(self.ln1(x)) + x
        out = self.ffw(self.ln2(x)) + x
        return out
      

# Create Bigram Model

# Process Input
if __name__ == "__main__":
    file_name = 'input/input.txt'
    with open(file_name, 'r', encoding= 'utf-8') as f:
        text = f.read()

    # Hyperparameters
    batch_size = 64
    block_size = 256
    epoches = 5000
    eval_epoche = 500
    estimate_eval_epoches = 200
    lr = 3e-4 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 384
    n_heads = 6
    n_layer = 6
    dropout = 0.2
    assert n_embd % n_heads == 0

    # Tokenisation
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    itoc = {i: c for i, c in enumerate(chars)}
    ctoi = {c: i for i, c in enumerate(chars)}
    encode = lambda l: [ctoi[i] for i in l] #take in a string
    decode = lambda l: "".join([itoc[i] for i in l]) # take in a list of intiegers
    data = torch.tensor(encode(text), dtype=torch.long, device = device)

    # Train Val Split
    train = data[:int(len(data)*0.9)]
    val = data[int(len(data)*0.9):]
    m = Decoder()
    m.to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr) # Typical good learning rate is 3e-4, but small models can use higher learning rate

    m.train()
    for steps in range(epoches):
        xa, xb = get_batch('train')
        logits, loss = m(xa, xb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (steps % eval_epoche == 0):
            estimate_eval()

    res = decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_tokens=1000).tolist()[0])
    print(res)