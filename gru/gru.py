import os
import numpy as np
import tinygrad as tg
from tinygrad.tensor import Tensor
import matplotlib.pyplot as plt
import random

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['\n'] = 0
itos = {v:k for k,v in stoi.items()}

with open('../data/names.txt', 'r') as f:
    data = f.readlines()

Tensor.manual_seed(42)
np.random.seed(42)

# hyperparams
vocab_len = len(stoi)
hidden_dim = 150
context_len = 8
batch_size = 8

Wr = Tensor.randn(vocab_len, hidden_dim) * 0.01
Ur = Tensor.randn(hidden_dim, hidden_dim) * 0.01
br = Tensor.zeros(hidden_dim)

Wz = Tensor.randn(vocab_len, hidden_dim) * 0.01
Uz = Tensor.randn(hidden_dim, hidden_dim) * 0.01
bz = Tensor.zeros(hidden_dim)

Wa = Tensor.randn(vocab_len, hidden_dim) * 0.01
Ua = Tensor.randn(hidden_dim, hidden_dim) * 0.01
ba = Tensor.zeros(hidden_dim)

Wy = Tensor.randn(hidden_dim, vocab_len) * 0.01

params = [
        Wr, Ur, br,
        Wz, Uz, bz,
        Wa, Ua, ba,
        Wy
        ]
for param in params: param.requires_grad = True

def build_dataset(data):
    X, Y = [], []
    for w in data:
        ixs = [stoi[c] for c in w[:context_len]] + [0] * (context_len - len(w))
        pad = [0] * (context_len - 1)
        for t in range(1, context_len):
            X.append(ixs[:t] + pad)
            Y.append(ixs[1:t+1] + pad)
            pad = pad[:-1]
    return np.array(X), np.array(Y)

random.seed(42)
random.shuffle(data)
n1 = int(.8*len(data))
n2 = int(.9*len(data))
Xtrain, Ytrain = build_dataset(data[:n1])
Xval, Yval = build_dataset(data[n1:n2])
Xtest, Ytest = build_dataset(data[:n2])

def one_hot(ix, num_classes):
    t = np.zeros(num_classes)
    t[ix] = 1
    return Tensor(t)

def forward(x, hprev):
    pre_r = Wr[x] + hprev @ Ur + br
    r = pre_r.sigmoid()
    pre_z = Wz[x] + hprev @ Uz + bz
    pre_a = Wa[x] + (r * hprev) @ Ua + ba
    z = pre_z.sigmoid()
    a = pre_a.tanh()
    h = (1 - z) * hprev + z * a
    y = h @ Wy
    return y, h

def sample(seed):
    Tensor.no_grad = True
    assert type(seed) == int
    out = [seed]
    h = Tensor.zeros(hidden_dim)
    ix = seed
    for t in range(context_len):
        y, h = forward(ix, h)
        ix = y.softmax().multinomial().item()
        if ix == 0: break
        out.append(ix)
    Tensor.no_grad = False
    return ''.join(itos[i] for i in out)
 
def evaluate(inputs, targets, hprev):
    Tensor.no_grad = True
    loss = 0
    h = hprev.detach()
    for t in range(context_len):
        y, h = forward(inputs[:,t], hprev)
        target_one_hot = one_hot(targets[:,t], num_classes=vocab_len).float()
        loss += target_one_hot.binary_crossentropy(y)
    Tensor.no_grad = False
    return loss, h

def train_auto(inputs, targets, hprev, lr):
    loss = 0
    h = hprev.detach()
    for t in range(context_len):
        y, h = forward(inputs[:,t], hprev)
        target_one_hot = one_hot(targets[:,t],num_classes=vocab_len).float()
        loss += F.cross_entropy(y, target_one_hot)
        loss += target_one_hot.binary_crossentropy(y)
    for param in params: param.grad = None
    loss.backward()
    for param in params: param.data += -lr * param.grad
    return loss.detach(), h.detach()

train_steps = 90000
evaluate_steps = 30000
test_steps = 10000

train = train_auto

lossi = []
epochs = 2
for ep in range(epochs):
    hprev = Tensor.randn(hidden_dim)
    for i in range(train_steps):
        ix = np.random.randint(0, Xtrain.shape[0], size=batch_size)
        X, Y = Tensor(Xtrain[ix], requires_grad=False), Tensor(Ytrain[ix], requires_grad=False)
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, hprev = train(X, Y, hprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
            print(f'sample: {sample(X[0][0].item())}\n')
        lossi.append(loss.log10())
    evaluate_loss = 0
    for i in range(evaluate_steps):
        ix = np.random.randint(0, Xval.shape[0], size=batch_size)
        X, Y = Tensor(Xval[ix], requires_grad=False), Tensor(Yval[ix], requires_grad=False)
        loss, hprev = evaluate(X, Y, hprev)
        if i % 1000 == 0:
            print(f'evaluate step {i}/{evaluate_steps}: {loss.item():.4f}')
        evaluate_loss += loss
    avg_evaluate_loss = evaluate_loss/evaluate_steps
    print(f'average validation loss: {avg_evaluate_loss.item():.4f}')
    if avg_evaluate_loss <= 3.0:
        break

plt.plot(lossi)
plt.savefig('loss.png')

test_loss = 0
hprev = Tensor.rand(hidden_dim, generator=rng)
for i in range(test_steps):
    ix = np.random.randint(0, Xtest.shape[0], size=batch_size)
    X, Y = Tensor(Xtest[ix], requires_grad=False), Tensor(Ytest[ix], requires_grad=False)
    loss, hprev = evaluate(X, Y, hprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')

for _ in range(30):
    seed = np.random.randint(1, vocab_len, size=1).item()
    print(sample(seed))

