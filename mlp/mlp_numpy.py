#!/usr/bin/env python3

import numpy as np
import random
import matplotlib.pyplot as plt

with open('names.txt', 'r') as f:
    data = f.read().splitlines()

stoi = {chr(ord('a') + i):i+1 for i in range(26)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

rng=np.random.default_rng()
batch_size = 32
vocab_len = len(stoi)
embed_dim = 30
context_len = 3
hidden_dim = 200
C = rng.random((vocab_len, embed_dim)) * 0.01
H = rng.random((context_len * embed_dim, hidden_dim)) * 0.01
d = np.zeros(hidden_dim)
U = rng.random((hidden_dim, vocab_len)) * 0.01
b = np.zeros(vocab_len)
params = [C, H, d, U, b]

def softmax(y):
    e_y = np.exp(y - y.max(1,keepdims=True))
    return e_y * (e_y.sum(1,keepdims=True)**-1)

def cross_entropy(y_pred, y_true):
    return -np.log(y_pred)[range(batch_size),y_true].mean()

def cross_entropy_grad(y_pred, y_true):
    grad = y_pred
    # NOTE this indexing operation does the same as the following loop
    # for i,j in zip(range(batch_size), y_true): grad[i][j] -= 1
    # it subtracts 1 from the gradient corresponding to the correct output entry in the ith batch
    grad[range(batch_size), y_true] -= 1
    grad /= batch_size
    return grad

def forward(X):
    embed = C[X]
    concat = embed.reshape((-1, H.shape[0]))
    hidden = np.tanh(concat @ H + d)
    logits = hidden @ U + b
    probs = softmax(logits)
    return probs

def train(X, Y, lr):
    # forward pass
    global C, H, d, U, b
    embed = C[X]
    concat = embed.reshape((-1, H.shape[0]))
    prehidden = concat @ H + d
    hidden = np.tanh(prehidden)
    logits = hidden @ U + b
    s = softmax(logits)
    loss = cross_entropy(s, Y)
    # backward pass
    dloss_wrt_logits = cross_entropy_grad(s, Y)
    dlogits_wrt_b = dloss_wrt_logits.sum(0)
    dlogits_wrt_U = hidden.T @ dloss_wrt_logits
    dlogits_wrt_hidden = dloss_wrt_logits @ U.T
    dhidden_wrt_prehidden = (1 - hidden**2) * dlogits_wrt_hidden
    dprehidden_wrt_d = dhidden_wrt_prehidden.sum(0)
    dprehidden_wrt_H = concat.T @ dhidden_wrt_prehidden
    dprehidden_wrt_concat = dhidden_wrt_prehidden @ H.T
    dconcat_wrt_embed = dprehidden_wrt_concat.reshape(embed.shape)
    dembed_wrt_C = np.zeros(C.shape)
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            ix = X[k,j]
            dembed_wrt_C[ix] += dconcat_wrt_embed[k,j]
    # update
    grads = [dembed_wrt_C, dprehidden_wrt_H, dprehidden_wrt_d, dlogits_wrt_U, dlogits_wrt_b]
    for p, g in zip(params, grads):
        p -= g * lr
    return (loss, grads)

def build_dataset(words, n):  
    X, Y = [], []
    for w in words:
        context = [0] * n
        for c in w + '.':
            ix = stoi[c]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

rng.shuffle(data)
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))

Xtrain, Ytrain = build_dataset(data[:n1], context_len)
Xvalidate, Yvalidate = build_dataset(data[n1:n2], context_len)
Xtest, Ytest = build_dataset(data[n2:], context_len)

assert batch_size <= Xtrain.shape[0]

MAX_STEPS = 200000

lossi = []
for i in range(MAX_STEPS):
    ix = rng.integers(0, Xtrain.shape[0], (batch_size,))
    Xbatch, Ybatch = Xtrain[ix], Ytrain[ix] # batch X,Y
    loss, grads = train(Xbatch, Ybatch, 0.1 if i < (MAX_STEPS>>1) else 0.001)
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{MAX_STEPS:7d}: {loss:.4f}')
    lossi.append(np.log10(loss))

plt.plot(lossi)
plt.savefig('loss_numpy.png')

for _ in range(20):
    out = []
    context = [0] * context_len
    while True:
        probs = forward(context)
        ix = rng.multinomial(1, probs, size=1).argmax()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
