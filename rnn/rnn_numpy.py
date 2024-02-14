import numpy as np
import random
import matplotlib.pyplot as plt

with open('../data/names.txt', 'r') as f:
    data = f.readlines()

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['\n'] = 0
itos = {i:s for s,i in stoi.items()}

# hyper params
rng=np.random.default_rng(seed=42)
vocab_len = len(stoi)
hidden_dim = 200
context_len = 8

# model params
U = rng.random((vocab_len, hidden_dim)) * 0.01
W = rng.random((hidden_dim, hidden_dim)) * 0.01
V = rng.random((hidden_dim, vocab_len)) * 0.01
b = np.zeros(hidden_dim) # hidden bias
c = np.zeros(vocab_len) # output bias
params = [U,W,V,b,c]

def sigmoid(z):
    z -= z.max(-1,keepdims=True)
    return (1 + np.exp(-z))**-1

def sigmoid_grad(z):
    z -= z.max(-1,keepdims=True)
    e_nz = np.exp(-z)
    return e_nz / (1 + e_nz)**2

def softmax(y):
    e_y = np.exp(y - y.max(-1,keepdims=True))
    return e_y * (e_y.sum(-1,keepdims=True)**-1)

def cross_entropy(y_pred, y_true):
    return -np.log(y_pred)[y_true].mean()

def cross_entropy_grad(y_pred, y_true):
    grad = np.copy(y_pred)
    grad[y_true] -= 1
    return grad

def inference(inputs, targets, hprev):
    tmax = len(inputs)
    assert tmax == len(targets)
    xs, hs, qs, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(tmax):
        xs[t] = np.zeros(vocab_len)
        xs[t][inputs[t]] = 1
        hs[t] = sigmoid(xs[t] @ U + hs[t-1] @ W + b)
        qs[t] = hs[t] @ V + c
        ps[t] = softmax(qs[t])
        loss += cross_entropy(ps[t], targets[t])
    return loss, hs[tmax-1]

def sample(seed, hinit):
    h = np.copy(hinit)
    outs = [seed]
    x = np.zeros(vocab_len)
    x[seed] = 1
    for t in range(context_len):
        h = sigmoid(x @ U + h @ W + b)
        q = h @ V + c
        p = softmax(q)
        assert p.shape == (vocab_len,)
        ix = rng.multinomial(1, p, size=1).argmax()
        x.fill(0)
        x[ix] = 1
        if ix == 0:
            break
        outs.append(ix)
    return ''.join(itos[ix] for ix in outs)

def train(inputs, targets, hprev, lr):
    tmax = len(inputs)
    assert tmax == len(targets)
    xs, hs, qs, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(tmax):
        xs[t] = np.zeros(vocab_len)
        xs[t][inputs[t]] = 1
        hs[t] = sigmoid(xs[t] @ U + hs[t-1] @ W + b)
        qs[t] = hs[t] @ V + c
        ps[t] = softmax(qs[t])
        loss += cross_entropy(ps[t], targets[t])
    # backward pass 
    dh_wrt_U, dh_wrt_W, dq_wrt_V = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    dh_wrt_b, dq_wrt_c = np.zeros_like(b), np.zeros_like(c)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(tmax)):
        dloss_wrt_q = cross_entropy_grad(ps[t], targets[t])
        dq_wrt_V += hs[t].reshape((-1,1)) @ dloss_wrt_q.reshape((1,-1))
        dq_wrt_c += dloss_wrt_q
        dq_wrt_h = dloss_wrt_q @ V.T + dhnext # what does dhnext do?
        dh_wrt_preact = sigmoid_grad(hs[t]) * dq_wrt_h
        dh_wrt_b += dh_wrt_preact
        dh_wrt_W += hs[t-1].T @ dh_wrt_preact
        dh_wrt_U += xs[t].reshape((-1,1)) @ dh_wrt_preact.reshape((1,-1))
        dhnext = dh_wrt_preact @ W.T
    grads = [dh_wrt_U, dh_wrt_W, dq_wrt_V, dh_wrt_b, dq_wrt_c]
    for param, grad in zip(params, grads):
        np.clip(grad, -5, 5, out=grad) # mitigate exploding or vanishing grad
        param += -lr * grad
    return (loss, grads, hs[tmax-1])

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

rng.shuffle(data)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))

Xtrain, Ytrain = build_dataset(data[:n1], context_len)
Xvalidate, Yvalidate = build_dataset(data[n1:n2], context_len)
Xtest, Ytest = build_dataset(data[n2:], context_len)

train_steps = 60000
validate_steps = 30000
test_steps = 10000

lossi = []
epochs = 5
hprev = np.zeros(hidden_dim)
for ep in range(epochs):
    for i in range(train_steps):
        ix = rng.integers(0, len(Xtrain), size=1)[0]
        X, Y = Xtrain[ix], Ytrain[ix]
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, grads, hprev = train(X, Y, hprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss:.4f}')
            print(f'sample: {sample(X[0], hprev)}')
        lossi.append(np.log10(loss))
    validate_loss = 0
    for i in range(validate_steps):
        ix = rng.integers(0, len(Xvalidate), size=1)[0]
        X, Y = Xvalidate[ix], Yvalidate[ix]
        loss, hprev = inference(X, Y, hprev)
        if i % 1000 == 0:
            print(f'validate step {i}/{validate_steps}: {loss:.4f}')
        validate_loss += loss
    avg_validate_loss = validate_loss/validate_steps
    print(f'average validation loss: {avg_validate_loss:.4f}')
    hprev.fill(0)
    if avg_validate_loss <= 3.0:
        break

test_loss = 0
for i in range(test_steps):
    ix = rng.integers(0, len(Xtest), size=1)[0]
    X, Y = Xtest[ix], Ytest[ix]
    loss, hprev = inference(X, Y, hprev)
    test_loss += loss

print(f'average test loss: {test_loss/test_steps:.4f}')

plt.plot(lossi)
plt.savefig('loss_numpy.png')

for _ in range(30):
    seed = rng.integers(1, vocab_len, size=1)[0]
    print(sample(seed, np.zeros(hidden_dim)))
