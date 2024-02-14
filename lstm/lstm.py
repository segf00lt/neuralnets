import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os

# LSTM

stoi = {chr(ord('a')+i): i+1 for i in range(26)}
stoi['\n'] = 0
itos = {v:k for k,v in stoi.items()}

with open('../data/names.txt', 'r') as f:
    data = f.readlines()

def build_dataset(data):
    X, Y = [], []
    for w in data:
        ixs = [stoi[c] for c in w]
        for t in range(1, len(w)):
            X.append(ixs[:t])
            Y.append(ixs[1:t+1])
    return X, Y

random.seed(42)
random.shuffle(data)
n1 = int(.8*len(data))
n2 = int(.9*len(data))
Xtrain, Ytrain = build_dataset(data[:n1])
Xval, Yval = build_dataset(data[n1:n2])
Xtest, Ytest = build_dataset(data[:n2])

# hyperparams
vocab_len = len(stoi)
hidden_dim = 150
context_len = 8

rng = torch.random.manual_seed(42)

Wf = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Wi = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Wo = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01
Wa = torch.randn((vocab_len, hidden_dim), generator=rng) * 0.01

Uf = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
Ui = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
Uo = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
Ua = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01

bf = torch.zeros(hidden_dim)
bi = torch.zeros(hidden_dim)
bo = torch.zeros(hidden_dim)
ba = torch.zeros(hidden_dim)

D = torch.randn((hidden_dim, vocab_len), generator=rng) * 0.01 # D for de-embed

params = [
        Wf, Wi, Wo, Wa,
        Uf, Ui, Uo, Ua,
        bf, bi, bo, ba,
        D
        ]
for param in params: param.requires_grad = True

def forward(x, hprev, cprev):
    pre_f = x @ Wf + hprev @ Uf + bf
    pre_i = x @ Wi + hprev @ Ui + bi
    pre_o = x @ Wo + hprev @ Uo + bo
    pre_a = x @ Wa + hprev @ Ua + ba

    f = F.sigmoid(pre_f)
    i = F.sigmoid(pre_i)
    o = F.sigmoid(pre_o)
    a = torch.tanh(pre_a)

    c = f * cprev + i * a
    h = o * torch.tanh(c)

    y = h @ D

    return y, h, c

def sample(seed, hprev, cprev):
    with torch.no_grad():
        out = [seed]
        h = hprev.clone()
        c = cprev.clone()
        ix = seed
        for t in range(context_len):
            input_one_hot = F.one_hot(torch.tensor(ix), num_classes=vocab_len).float()
            y, h, c = forward(input_one_hot, h, c)
            ix = torch.multinomial(F.softmax(y, dim=-1), 1, generator=rng).item()
            if ix == 0: break
            out.append(ix)
    return ''.join(itos[i] for i in out)

def evaluate(inputs, targets, hprev, cprev):
    with torch.no_grad():
        tmax = len(inputs)
        loss = 0
        h = hprev.clone()
        c = cprev.clone()
        for t in range(tmax):
            input_one_hot = F.one_hot(torch.tensor(inputs[t]), num_classes=vocab_len).float()
            y, h, c = forward(input_one_hot, hprev, cprev)
            target_one_hot = F.one_hot(torch.tensor(targets[t]), num_classes=vocab_len).float()
            loss += F.cross_entropy(y, target_one_hot)
    return loss, h, c

def train_torch(inputs, targets, hprev, cprev, lr):
    tmax = len(inputs)
    loss = 0
    h = hprev.clone()
    c = cprev.clone()
    for t in range(tmax):
        input_one_hot = F.one_hot(torch.tensor(inputs[t]), num_classes=vocab_len).float()
        y, h, c = forward(input_one_hot, h, c)
        target_one_hot = F.one_hot(torch.tensor(targets[t]), num_classes=vocab_len).float()
        loss += F.cross_entropy(y, target_one_hot)

    for param in params: param.grad = None
    loss.backward()

    for param in params:
        param.data += -lr * param.grad

    return loss.detach(), h.detach(), c.detach()

def train_manual(inputs, targets, hprev, cprev, lr):
    with torch.no_grad():
        tmax = len(inputs)
        loss = 0
        pre_f, pre_i, pre_o, pre_a, = {},{},{},{}
        x, f, i, o, a, c, e, h, y, p = {},{},{},{},{},{},{},{},{},{}
        
        h[-1] = hprev.clone()
        c[-1] = cprev.clone()
        
        for t in range(tmax):
            x[t] = F.one_hot(torch.tensor(inputs[t]), num_classes=vocab_len).float()
            pre_f[t] = x[t] @ Wf + h[t-1] @ Uf + bf
            pre_i[t] = x[t] @ Wi + h[t-1] @ Ui + bi
            pre_o[t] = x[t] @ Wo + h[t-1] @ Uo + bo
            pre_a[t] = x[t] @ Wa + h[t-1] @ Ua + ba
            f[t] = F.sigmoid(pre_f[t])
            i[t] = F.sigmoid(pre_i[t])
            o[t] = F.sigmoid(pre_o[t])
            a[t] = torch.tanh(pre_a[t])
            c[t] = f[t] * c[t-1] + i[t] * a[t]
            e[t] = torch.tanh(c[t])
            h[t] = o[t] * e[t]
            y[t] = h[t] @ D
            target_one_hot = F.one_hot(torch.tensor(targets[t]), num_classes=vocab_len).float()
            loss += F.cross_entropy(y[t], target_one_hot)
        
        def sigmoid_grad(z):
            z -= z.max(-1,keepdims=True).values[0]
            e_nz = torch.exp(-z)
            return e_nz / (1 + e_nz)**2
        
        def tanh_grad(x):
            # NOTE this function assumes tanh'd input
            return 1 - x**2

        dpre_f_wrt_Wf = torch.zeros_like(Wf)
        dpre_i_wrt_Wi = torch.zeros_like(Wi)
        dpre_o_wrt_Wo = torch.zeros_like(Wo)
        dpre_a_wrt_Wa = torch.zeros_like(Wa)
        dpre_f_wrt_Uf = torch.zeros_like(Uf)
        dpre_i_wrt_Ui = torch.zeros_like(Ui)
        dpre_o_wrt_Uo = torch.zeros_like(Uo)
        dpre_a_wrt_Ua = torch.zeros_like(Ua)
        dpre_f_wrt_bf = torch.zeros_like(bf)
        dpre_i_wrt_bi = torch.zeros_like(bi)
        dpre_o_wrt_bo = torch.zeros_like(bo)
        dpre_a_wrt_ba = torch.zeros_like(ba)
        dy_wrt_D = torch.zeros_like(D)
        dcnext = torch.zeros_like(c[0])
        dhnext = torch.zeros_like(h[0])

        grads = [
                dpre_f_wrt_Wf, dpre_i_wrt_Wi, dpre_o_wrt_Wo, dpre_a_wrt_Wa,
                dpre_f_wrt_Uf, dpre_i_wrt_Ui, dpre_o_wrt_Uo, dpre_a_wrt_Ua,
                dpre_f_wrt_bf, dpre_i_wrt_bi, dpre_o_wrt_bo, dpre_a_wrt_ba,
                dy_wrt_D,
                ]
        for grad in grads: grad.requires_grad = False

        for t in reversed(range(tmax)):
            # NOTE this optimization of cross_entropy_grad assumes softmax'd input
            dloss_wrt_y = F.softmax(y[t], dim=-1)
            dloss_wrt_y[targets[t]] -= 1

            dy_wrt_D += h[t].view((-1,1)) @ dloss_wrt_y.view((1,-1))
            dy_wrt_h = dloss_wrt_y @ D.T + dhnext

            dh_wrt_o = dy_wrt_h * e[t]
            dh_wrt_e = o[t] * dy_wrt_h

            do_wrt_pre_o = dh_wrt_o * sigmoid_grad(pre_o[t])

            dpre_o_wrt_bo += do_wrt_pre_o
            dpre_o_wrt_Uo += h[t-1].view((-1,1)) @ do_wrt_pre_o.view((1,-1))
            dpre_o_wrt_Wo += x[t].view((-1,1)) @ do_wrt_pre_o.view((1,-1))

            de_wrt_c = dh_wrt_e * tanh_grad(e[t]) + dcnext # tanh_grad expects tanh'd input

            dcnext = f[t] * de_wrt_c

            dc_wrt_f = de_wrt_c * c[t-1]
            dc_wrt_i = de_wrt_c * a[t]
            dc_wrt_a = i[t] * de_wrt_c

            df_wrt_pre_f = dc_wrt_f * sigmoid_grad(pre_f[t])
            di_wrt_pre_i = dc_wrt_i * sigmoid_grad(pre_i[t])
            da_wrt_pre_a = dc_wrt_a * tanh_grad(a[t]) # tanh_grad expects tanh'd input

            dpre_f_wrt_bf += df_wrt_pre_f
            dpre_f_wrt_Uf += h[t-1].view((-1,1)) @ df_wrt_pre_f.view((1,-1))
            dpre_f_wrt_Wf += x[t].view((-1,1)) @ df_wrt_pre_f.view((1,-1))

            dpre_i_wrt_bi += di_wrt_pre_i
            dpre_i_wrt_Ui += h[t-1].view((-1,1)) @ di_wrt_pre_i.view((1,-1))
            dpre_i_wrt_Wi += x[t].view((-1,1)) @ di_wrt_pre_i.view((1,-1))

            dpre_a_wrt_ba += da_wrt_pre_a
            dpre_a_wrt_Ua += h[t-1].view((-1,1)) @ da_wrt_pre_a.view((1,-1))
            dpre_a_wrt_Wa += x[t].view((-1,1)) @ da_wrt_pre_a.view((1,-1))

            dhnext = df_wrt_pre_f @ Uf.T
            dhnext += di_wrt_pre_i @ Ui.T
            dhnext += do_wrt_pre_o @ Uo.T
            dhnext += da_wrt_pre_a @ Ua.T
            
        for param,grad in zip(params,grads):
            param.data += -lr * grad
        
    return loss, h[tmax-1], c[tmax-1]


train_steps = 90000
evaluate_steps = 30000
test_steps = 10000

train = train_torch

if os.getenv('MANUAL'):
    print('training LSTM model with manual backprop\n')
    train = train_manual
elif os.getenv('TORCH'):
    print('training LSTM model with torch backprop\n')
    train = train_torch

lossi = []
epochs = 2
for ep in range(epochs):
    hprev = torch.rand(hidden_dim, generator=rng)
    cprev = torch.rand(hidden_dim, generator=rng)
    for i in range(train_steps):
        ix = torch.randint(0, len(Xtrain), (1,), generator=rng)
        X, Y = Xtrain[ix], Ytrain[ix]
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, hprev, cprev = train(X, Y, hprev, cprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
            print(f'sample: {sample(X[0], hprev, cprev)}\n')
        lossi.append(torch.log10(loss))
    evaluate_loss = 0
    for i in range(evaluate_steps):
        ix = torch.randint(0, len(Xval), (1,), generator=rng)
        X, Y = Xval[ix], Yval[ix]
        loss, hprev, cprev = evaluate(X, Y, hprev, cprev)
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
cprev = torch.rand(hidden_dim, generator=rng)
for i in range(test_steps):
    ix = torch.randint(0, len(Xtest), (1,), generator=rng)
    X, Y = Xtest[ix], Ytest[ix]
    loss, hprev, cprev = evaluate(X, Y, hprev, cprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')

for _ in range(30):
    seed = torch.randint(1, vocab_len, (1,), generator=rng).item()
    print(sample(seed, torch.zeros(hidden_dim), torch.zeros(hidden_dim)))
