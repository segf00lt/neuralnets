import time
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['\n'] = 0
itos = {i:s for s,i in stoi.items()}

with open('names.txt', 'r') as f:
    data = f.readlines()

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['\n'] = 0
itos = {i:s for s,i in stoi.items()}

# hyper params
rng = torch.random.manual_seed(42)
vocab_len = len(stoi)
hidden_dim = 200
context_len = 8

# model params
U = torch.rand((vocab_len, hidden_dim), generator=rng) * 0.01
W = torch.rand((hidden_dim, hidden_dim), generator=rng) * 0.01
V = torch.rand((hidden_dim, vocab_len), generator=rng) * 0.01
b = torch.zeros(hidden_dim) # hidden bias
c = torch.zeros(vocab_len) # output bias
params = [U,W,V,b,c]
for param in params:
    param.requires_grad = True

def inference(inputs, targets, hprev):
    tmax = len(inputs)
    assert tmax == len(targets)
    h = torch.clone(hprev)
    loss = 0
    # forward pass
    with torch.no_grad():
        for t in range(tmax):
            x = F.one_hot(torch.tensor(inputs[t]), num_classes=vocab_len).float()
            h = F.sigmoid(x @ U + h @ W + b)
            q = h @ V + c
            loss += F.cross_entropy(q, F.one_hot(torch.tensor(targets[t]), num_classes=vocab_len).float())
    return loss, h

def sample(seed, hinit):
    h = torch.clone(hinit)
    outs = [seed]
    ix = seed
    with torch.no_grad():
        for t in range(context_len):
            x = F.one_hot(torch.tensor(ix), num_classes=vocab_len).float()
            h = F.sigmoid(x @ U + h @ W + b)
            q = h @ V + c
            p = F.softmax(q, dim=-1)
            ix = torch.multinomial(p, num_samples=1, generator=rng).item()
            if ix == 0: break
            outs.append(ix)
    return ''.join(itos[i] for i in outs)

def train(inputs, targets, hprev, lr):
    tmax = len(inputs)
    assert tmax == len(targets)
    h = torch.clone(hprev)
    loss = 0
    # forward pass
    for t in range(tmax):
        x = F.one_hot(torch.tensor(inputs[t]), num_classes=vocab_len).float()
        h = F.sigmoid(x @ U + h @ W + b)
        q = h @ V + c
        target_one_hot = F.one_hot(torch.tensor(targets[t]), num_classes=vocab_len).float()
        # NOTE F.cross_entropy expects unnormalized inputs, it applies a softmax() internally
        loss += F.cross_entropy(q, target_one_hot)
    # backward pass 
    for param in params: param.grad = None
    loss.backward()
    for param in params:
        torch.clamp(param.grad, -5, 5, out=param.grad) # mitigate exploding or vanishing grad
        param.data += -lr * param.grad
    return loss, h.detach()

def build_dataset(words, n):  
    X, Y = [], []
    for w in words:
        ixs = [stoi[c] for c in w.lower()]
        tmax = len(w)
        for t in range(1,tmax):
            x = ixs[:t]
            y = ixs[1:t+1]
            X.append(x)
            Y.append(y)
    return X, Y

random.seed(42)
random.shuffle(data)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))

Xtrain, Ytrain = build_dataset(data[:n1], context_len)
Xvalidate, Yvalidate = build_dataset(data[n1:n2], context_len)
Xtest, Ytest = build_dataset(data[n2:], context_len)

train_steps = 50000
validate_steps = 25000
test_steps = 10000

lossi = []
epochs = 5
for ep in range(epochs):
    hprev = torch.rand(hidden_dim, generator=rng)
    for i in range(train_steps):
        ix = torch.randint(0, len(Xtrain), (1,), generator=rng)
        X, Y = Xtrain[ix], Ytrain[ix]
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, hprev = train(X, Y, hprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
            print(f'sample: {sample(X[0], hprev)}')
        with torch.no_grad(): lossi.append(torch.log10(loss.detach()))
    validate_loss = 0
    for i in range(validate_steps):
        ix = torch.randint(0, len(Xvalidate), (1,), generator=rng)
        X, Y = Xvalidate[ix], Yvalidate[ix]
        loss, hprev = inference(X, Y, hprev)
        if i % 1000 == 0:
            print(f'validate step {i}/{validate_steps}: {loss.item():.4f}')
        validate_loss += loss
    avg_validate_loss = validate_loss/validate_steps
    print(f'average validation loss: {avg_validate_loss.item():.4f}')
    if avg_validate_loss <= 3.0:
        break

plt.plot(lossi)
plt.savefig('loss.png')

test_loss = 0
hprev = torch.rand(hidden_dim, generator=rng)
for i in range(test_steps):
    ix = torch.randint(0, len(Xtest), (1,), generator=rng)
    X, Y = Xtest[ix], Ytest[ix]
    loss, hprev = inference(X, Y, hprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')

for _ in range(30):
    seed = torch.randint(1, vocab_len, size=(1,), generator=rng)
    print(sample(seed, torch.zeros(hidden_dim)))
