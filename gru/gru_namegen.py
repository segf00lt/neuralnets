# GRU

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['\n'] = 0
itos = {v:k for k,v in stoi.items()}

with open('names.txt', 'r') as f:
    data = f.readlines()

rng = torch.random.manual_seed(42)

# hyperparams
vocab_len = len(stoi)
hidden_dim = 150
context_len = 8
batch_size = 8

Wr = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Ur = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
br = torch.zeros(hidden_dim)

Wz = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Uz = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
bz = torch.zeros(hidden_dim)

Wa = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Ua = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
ba = torch.zeros(hidden_dim)

Wy = torch.randn((hidden_dim, vocab_len), generator=rng) * 0.01

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
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(data)
n1 = int(.8*len(data))
n2 = int(.9*len(data))
Xtrain, Ytrain = build_dataset(data[:n1])
Xval, Yval = build_dataset(data[n1:n2])
Xtest, Ytest = build_dataset(data[:n2])

def forward(x, hprev):
    pre_r = Wr[x] + hprev @ Ur + br
    r = F.sigmoid(pre_r)
    pre_z = Wz[x] + hprev @ Uz + bz
    pre_a = Wa[x] + (r * hprev) @ Ua + ba
    z = F.sigmoid(pre_z)
    a = torch.tanh(pre_a)
    h = (1 - z) * hprev + z * a
    y = h @ Wy
    return y, h

def sample(seed):
    with torch.no_grad():
        assert type(seed) == int
        out = [seed]
        h = torch.zeros(hidden_dim)
        ix = seed
        for t in range(context_len):
            y, h = forward(ix, h)
            ix = torch.multinomial(F.softmax(y, dim=-1), 1, generator=rng).item()
            if ix == 0: break
            out.append(ix)
    return ''.join(itos[i] for i in out)
 
def evaluate(inputs, targets, hprev):
    with torch.no_grad():
        loss = 0
        h = hprev.clone()
        for t in range(context_len):
            y, h = forward(inputs[:,t], hprev)
            target_one_hot = F.one_hot(targets[:,t], num_classes=vocab_len).float()
            loss += F.cross_entropy(y, target_one_hot)
    return loss, h

def train_torch(inputs, targets, hprev, lr):
    loss = 0
    h = hprev.clone()
    for t in range(context_len):
        y, h = forward(inputs[:,t], hprev)
        target_one_hot = F.one_hot(targets[:,t],num_classes=vocab_len).float()
        loss += F.cross_entropy(y, target_one_hot)
    for param in params: param.grad = None
    loss.backward()
    for param in params: param.data += -lr * param.grad
    return loss.detach(), h.detach()

def train_manual(inputs, targets, hprev, lr):
    with torch.no_grad():
        loss = 0
        pre_r, pre_z, pre_a = {},{},{}
        x, r, z, a, h, y = {},{},{},{},{},{}
        h[-1] = hprev.clone()

        # forward pass
        for t in range(context_len):
            x[t] = F.one_hot(inputs[:,t],num_classes=vocab_len).float()
            pre_r[t] = x[t] @ Wr + h[t-1] @ Ur + br
            r[t] = F.sigmoid(pre_r[t])
            pre_z[t] = x[t] @ Wz + h[t-1] @ Uz + bz
            pre_a[t] = x[t] @ Wa + (r[t] * h[t-1]) @ Ua + ba
            z[t] = F.sigmoid(pre_z[t])
            a[t] = torch.tanh(pre_a[t])
            h[t] = (1 - z[t]) * h[t-1] + z[t] * a[t]
            y[t] = h[t] @ Wy
            target_one_hot = F.one_hot(targets[:,t],num_classes=vocab_len).float()
            loss += F.cross_entropy(y[t], target_one_hot)

        def sigmoid_grad(z):
            z -= z.max(-1,keepdims=True).values[0]
            e_nz = torch.exp(-z)
            return e_nz / (1 + e_nz)**2
        
        def tanh_grad(x):
            # NOTE this function assumes tanh'd input
            return 1 - x**2

        dpre_r_wrt_Wr = torch.zeros_like(Wr)
        dpre_r_wrt_Ur = torch.zeros_like(Ur)
        dpre_r_wrt_br = torch.zeros_like(br)
        dpre_z_wrt_Wz = torch.zeros_like(Wr)
        dpre_z_wrt_Uz = torch.zeros_like(Ur)
        dpre_z_wrt_bz = torch.zeros_like(br)
        dpre_a_wrt_Wa = torch.zeros_like(Wr)
        dpre_a_wrt_Ua = torch.zeros_like(Ur)
        dpre_a_wrt_ba = torch.zeros_like(br)
        dy_wrt_Wy = torch.zeros_like(Wy)
        dhnext = torch.zeros_like(h[0])
        grads = [
                dpre_r_wrt_Wr,
                dpre_r_wrt_Ur,
                dpre_r_wrt_br,
                dpre_z_wrt_Wz,
                dpre_z_wrt_Uz,
                dpre_z_wrt_bz,
                dpre_a_wrt_Wa,
                dpre_a_wrt_Ua,
                dpre_a_wrt_ba,
                dy_wrt_Wy
                ]

        # backward pass
        for t in reversed(range(tmax)):
            dloss_wrt_y = F.softmax(y[t], dim=-1)
            dloss_wrt_y[targets[:,t]] -= 1
            
            dy_wrt_Wy += h[t].view((-1,1)) @ dloss_wrt_y.view((1,-1))
            dy_wrt_h = dloss_wrt_y @ Wy.T + dhnext
            
            dh_wrt_a = z[t] * dy_wrt_h
            dh_wrt_z = dy_wrt_h * a[t] # I think (1 - z[t]) goes to 0, so it wont influence the gradient
            
            da_wrt_pre_a = tanh_grad(a[t]) * dh_wrt_a # tanh_grad expects tanh'd input
            dz_wrt_pre_z = sigmoid_grad(pre_z[t]) * dh_wrt_z
            
            dpre_a_wrt_ba += da_wrt_pre_a
            dpre_a_wrt_Ua += (r[t] * h[t-1]).view((-1,1)) @ da_wrt_pre_a.view((1,-1))
            dpre_a_wrt_Wa += x[t].view((-1,1)) @ da_wrt_pre_a.view((1,-1))
            dpre_a_wrt_r = (da_wrt_pre_a * h[t-1]) @ Ua.T # ???
            
            dpre_z_wrt_bz = dz_wrt_pre_z
            dpre_z_wrt_Uz = h[t-1].view((-1,1)) @ dz_wrt_pre_z.view((1,-1))
            dpre_z_wrt_Wz = x[t].view((-1,1)) @ dz_wrt_pre_z.view((1,-1))
            
            dr_wrt_pre_r = sigmoid_grad(pre_r[t]) * dpre_a_wrt_r
            
            dpre_r_wrt_br += dr_wrt_pre_r
            dpre_r_wrt_Ur += h[t-1].view((-1,1)) @ dr_wrt_pre_r.view((1,-1))
            dpre_r_wrt_Wr += x[t].view((-1,1)) @ dr_wrt_pre_r.view((1,-1))
            
            dhnext = (1 - z[t]) * dy_wrt_h
            dhnext += (r[t] * da_wrt_pre_a) @ Ua.T
            dhnext += dz_wrt_pre_z @ Uz.T
            dhnext += dr_wrt_pre_r @ Ur.T

        for param, grad in zip(params, grads):
            param.data += -lr * grad

    return loss, h[tmax-1]
    
train_steps = 90000
evaluate_steps = 30000
test_steps = 10000

train = train_torch

if os.getenv('MANUAL'):
    print('training GRU model with manual backprop\n')
    train = train_manual
elif os.getenv('TORCH'):
    print('training GRU model with torch backprop\n')
    train = train_torch

lossi = []
epochs = 2
for ep in range(epochs):
    hprev = torch.rand(hidden_dim, generator=rng)
    for i in range(train_steps):
        ix = torch.randint(0, len(Xtrain), (batch_size,), generator=rng)
        X, Y = Xtrain[ix], Ytrain[ix]
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, hprev = train(X, Y, hprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
            print(f'sample: {sample(X[0][0].item())}\n')
        lossi.append(torch.log10(loss))
    evaluate_loss = 0
    for i in range(evaluate_steps):
        ix = torch.randint(0, len(Xval), (batch_size,), generator=rng)
        X, Y = Xval[ix], Yval[ix]
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
hprev = torch.rand(hidden_dim, generator=rng)
for i in range(test_steps):
    ix = torch.randint(0, len(Xtest), (batch_size,), generator=rng)
    X, Y = Xtest[ix], Ytest[ix]
    loss, hprev = evaluate(X, Y, hprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')

for _ in range(30):
    seed = torch.randint(1, vocab_len, (1,), generator=rng).item()
    print(sample(seed))

