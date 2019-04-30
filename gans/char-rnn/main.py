import json
import random

import numpy as np
import numpy.random as rd

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def stream(data, window=128, mix=100000):
    while True:
        random.shuffle(leads)
        test = [c for lead in data for c in lead]
        for _ in range(mix):
            i = random.randint(0, len(test)-window-1)
            yield test[i:i+window], test[i+1:i+1+window]

def sample(net, chars, length=50, power=2.0, starter=None):
    characters = []
    exponent = 3
    inp = torch.tensor([[chids[" STOP "]]]).to(DEVICE)
    h   = net.create_hidden(1)
    for i in range(length):
        # exponent = min(exponent+power/100, power)
        inp, h = net(inp, h)
        p = inp.exp() / inp.exp().sum()
        p = (p ** exponent) / (p ** exponent).sum()
        p = p[0]
        inp    = rd.choice(len(chars), p=p.cpu().detach().numpy())
        characters.append(chars[inp])
        inp    = torch.tensor([[inp]]).to(DEVICE)
    return "".join(characters)



class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cells   = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size,)

    def forward(self, inp, hidden):
        inp = self.encoder(inp)
        output, hidden = self.cells(inp, hidden)
        output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return output, hidden

    def create_hidden(self, batch_size):
        zero = lambda: torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return [zero().to(DEVICE) for i in range(self.n_layers)]

with open("nice-char.json") as f:
    data = json.load(f)
chars = sorted(set(c for lead in data for c in lead)) + [" STOP "]
chids = {c:i for i,c in enumerate(chars)} 
leads = [[chids[c] for c in lead]+[chids[" STOP "]] for lead in data]
# ------ 
window   = 1000
iterator = stream(leads, window)
# net    = LSTM(len(chars), 512, len(chars)).to(DEVICE)
net    = (torch.load("net-weights-2.model")
        if torch.cuda.is_available()
        else torch.load("net-weights-2.model", map_location="cpu")).to(DEVICE)
loss   = nn.CrossEntropyLoss(reduction="none")
opt    = optim.Adam(net.parameters())
hid    = net.create_hidden(1)
wloss  = torch.tensor(np.linspace(0, 1, window, dtype=np.float32)**0.2).to(DEVICE)
wloss /= wloss.sum()
print("device         :", DEVICE)
print("num characters :", len(chars))
# for i in range(10000):
#     inp, tar = next(iterator)
#     inp = torch.tensor([[i] for i in inp]).to(DEVICE)
#     tar = torch.tensor(tar).to(DEVICE)
#     opt.zero_grad()
#     out, _ = net(inp, hid)
#     cost = loss(out, tar).dot(wloss)
#     cost.backward()
#     opt.step()
#     if i % 50 == 0:
#         print("%d %.2f" %(i, float(cost)))
#         print(sample(net, chars, 800))
#         print("---------") 
#     if (i+1) % 500 == 0:
#         torch.save(net, "net-weights.model")
for i in range(100):
    print(sample(net, chars, 2000))
    print("------")