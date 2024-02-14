"""
NOTES

The feature vector has shape = m.

In the mlp described in the paper, there is a matrix (or lookup table) C with
shape (17000, 30) or (|V|, m) where |V| is the size of the vocabulary.

The x input is the concatenation of the n feature vectors where n depends on the size
of our context (how many words we take into account to predict the next one).

Hidden layer shape is (h, n*m). h is a free variable up to us to decide.

If we remember linear algebra, we know that matmul of (h, n*m) matrix and (n*m, 1) vector
will produce a (h, 1) vector. Thus, the weights on the output layer must be (|V|, h).
With the hidden layer output having shape (h, 1) and the output layer having shape (|V|, h),
matmul(output, hidden) will have shape (|V|, 1).

To this output vector we apply an elementwise softmax function.
On the hidden layer output we apply a elementwise tanh or relu activation function.

When actually defining each of our matrices and vectors, we transpose all of them, so
the dimensions are reversed. This is cus of how arrays are indexed in computers, its less
of a hassle to think in column major.

Vlen = len(stoi)
m = 30 # embedding dimension
n = 3 # context
h = 200 # neurons in hidden layer
"""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open('../data/names.txt', 'r') as f:
    V = f.read().splitlines()

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
#chars = sorted(set(''.join(V)))
#stoi = {s:i+1 for i,s in enumerate(chars)}
#stoi['.'] = 0
#itos = {i:s for s,i in stoi.items()}
#print(itos)

def build_dataset(words, n):  
    X, Y = [], []
    for w in words:
        context = [0] * n
        for c in w + '.':
            ix = stoi[c]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

class MLP:
    def __init__(self, Vlen, m=80, n=4, h=200, generator=None):
        self.Vlen = Vlen
        self.m = m
        self.n = n
        self.h = h
        self.C = torch.randn((Vlen, m),     generator=generator) # embeddings
        # NOTE making hidden params small prevents tanh from saturating and gives the network more room to learn
        self.H = torch.randn((n*m, h),      generator=generator) * ((5/3) / ((n*m)**0.5)) # hidden layer
        self.d = torch.randn(h,             generator=generator) * 0.0 # hidden bias
        # NOTE making the output layer params tiny removes hocky stick shape from loss
        self.U = torch.randn((h, Vlen),     generator=generator) * 0.01 # output
        self.b = torch.randn(Vlen,          generator=generator) * 0.0 # output bias
        self.bngain = torch.ones((1, h))
        self.bnbias = torch.zeros((1, h))
        # NOTE you need these for forwardence
        self.bnmean_running = torch.zeros((1, h))
        self.bnstd_running = torch.ones((1, h))
        self.params = [self.C, self.H, self.d, self.U, self.b, self.bngain, self.bnbias]
        for p in self.params:
            p.requires_grad = True
    def forward(self, X):
        emb = self.C[X]
        prehidden = emb.view(-1, self.H.shape[0]) @ self.H + self.d
        bnmeani = self.bnmean_running
        bnstdi = self.bnstd_running
        prehidden = self.bngain*((prehidden-bnmeani)/bnstdi) + self.bnbias
        hidden = torch.tanh(prehidden)
        logits = hidden @ self.U + self.b
        probs = F.softmax(logits, dim=1)
        return probs
    def train(self, X, Y):
        # forward pass
        emb = self.C[X]
        concat = emb.view(-1, self.H.shape[0])
        prehidden = concat @ self.H + self.d
        # batch norm
        bnmeani = prehidden.mean(0,keepdim=True)
        bnstdi = prehidden.std(0,keepdim=True)
        prehidden = self.bngain*((prehidden-bnmeani)/bnstdi) + self.bnbias
        with torch.no_grad():
            self.bnmean_running = 0.999*self.bnmean_running + 0.001*bnmeani
            self.bnstd_running = 0.999*self.bnstd_running + 0.001*bnstdi
        hidden = torch.tanh(prehidden)
        logits = hidden @ self.U + self.b
        loss = F.cross_entropy(logits, Y)
        # backward pass
        for p in self.params:
            p.grad = None
        loss.backward()
        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in self.params:
            p.data += -lr * p.grad
        return loss


mlp = MLP(len(stoi), generator=torch.Generator().manual_seed(2147483647))

random.seed(22)
random.shuffle(V)
n1 = int(0.8*len(V))
n2 = int(0.9*len(V))

Xtrain, Ytrain = build_dataset(V[:n1], mlp.n)
Xvalidate, Yvalidate = build_dataset(V[n1:n2], mlp.n)
Xtest, Ytest = build_dataset(V[n2:], mlp.n)

batch_size = 32
assert batch_size <= Xtrain.shape[0]

stepi = []
lossi = []

# training

for i in range(10000):
    # minibatch construct
    ix = torch.randint(0, Xtrain.shape[0], (batch_size,))
    loss = mlp.train(Xtrain[ix], Ytrain[ix])
    # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())

plt.plot(stepi, lossi)
plt.savefig('loss.png')

# sampling

g = torch.Generator()
mlp.generator = g

for _ in range(20):
    out = []
    context = [0] * mlp.n # initialize with all ...
    while True:
        probs = mlp.forward(torch.tensor([context]))
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
