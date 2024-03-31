# implementation of a translation model similar to the paper
# "Learning Phrase Representations using RNN Encoder–Decoder
# for Statistical Machine Translation"

import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
import pickle
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

vocab_decode_path = 'de_en_vocab_decode.pkl'
vocab_encode_path = 'de_en_vocab_encode.pkl'
tokenized_dataset_path = 'de_en_tokenized_dataset.pkl'

assert os.path.exists(vocab_decode_path)
assert os.path.exists(vocab_encode_path)
assert os.path.exists(tokenized_dataset_path)

with open(tokenized_dataset_path, 'rb') as f: tokenized_dataset = pickle.load(f)
with open(vocab_decode_path, 'rb') as f: vocab_decode = pickle.load(f)
with open(vocab_encode_path, 'rb') as f: vocab_encode = pickle.load(f)

assert len(tokenized_dataset['de']) == len(tokenized_dataset['en'])
num_sentences = len(tokenized_dataset['de'])

print('german vocabulary',len(vocab_decode['de']))
print('english vocabulary',len(vocab_decode['en']))

rng = torch.random.manual_seed(42)
np.random.seed(42)

# hyper parameters

input_vocab_len = len(vocab_decode['de'])
target_vocab_len = len(vocab_decode['en'])
embed_dim = 80
hidden_dim = 500

# model parameters

Ei = torch.randn((input_vocab_len, embed_dim), generator=rng, device=device) * 0.01
Et = torch.randn((target_vocab_len, embed_dim), generator=rng, device=device) * 0.01

Wr = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Ur = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
br = torch.zeros(hidden_dim, device=device)

Qr = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Vr = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
cr = torch.zeros(hidden_dim, device=device)

Wz = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Uz = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
bz = torch.zeros(hidden_dim, device=device)

Qz = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Vz = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
cz = torch.zeros(hidden_dim, device=device)

Wa = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Ua = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
ba = torch.zeros(hidden_dim, device=device)

Qa = torch.randn((embed_dim, hidden_dim), generator=rng, device=device) * ((5/3)/(embed_dim**0.5))
Va = torch.randn((hidden_dim, hidden_dim), generator=rng, device=device) * 0.01
ca = torch.zeros(hidden_dim, device=device)

Vy = torch.randn((hidden_dim, target_vocab_len), generator=rng, device=device) * 0.01

params = [
        Ei, Et,
        Wr, Ur, br, Vr,
        Wz, Uz, bz, Vz,
        Wa, Ua, ba, Va,
        Vy
        ]
for param in params: param.requires_grad = True


def encoder_forward(x, hprev):
    emb = F.dropout(Ei[x])
    pre_r = emb @ Wr + hprev @ Ur + br
    r = F.sigmoid(pre_r)
    pre_z = emb @ Wz + hprev @ Uz + bz
    pre_a = emb @ Wa + (r * hprev) @ Ua + ba
    z = F.sigmoid(pre_z)
    a = torch.tanh(pre_a)
    h = ((1 - z) * hprev + z * a)
    return h

def decoder_forward(x, hprev):
    emb = F.relu(Et[x])
    pre_r = emb @ Qr + hprev @ Vr + cr
    r = F.sigmoid(pre_r)
    pre_z = emb @ Qz + hprev @ Vz + cz
    pre_a = emb @ Qa + (r * hprev) @ Va + ca
    z = F.sigmoid(pre_z)
    a = torch.tanh(pre_a)
    h = torch.tanh((1 - z) * hprev + z * a)
    y = h @ Vy
    return y, h

def compute_loss(inputs, targets, hprev, teacher_force=True):
    h = hprev.clone()
    input_tmax = len(inputs)
    targets = targets.to_list()
    targets.insert(0,0) # <SOS>
    targets.append(1) # <EOS>
    target_tmax = len(targets)
    loss = 0.0
    for t in reversed(range(input_tmax)):
        h = encoder_forward(inputs[t], h)
    if teacher_force:
        for t in range(target_tmax-1):
            y, h = decoder_forward(targets[t], h)
            target_one_hot = F.one_hot(torch.tensor(targets[t+1]),target_vocab_len).float()
            loss += F.cross_entropy(y, target_one_hot)
    else:
        y = targets[0]
        for t in range(target_tmax-1):
            y, h = decoder_forward(y, h)
            target_one_hot = F.one_hot(torch.tensor(targets[t+1]),target_vocab_len).float()
            loss += F.cross_entropy(y, target_one_hot)
            _, topi = y.topk(1)
            y = topi.squeeze().detach()
    return loss, h

def sample(inputs):
    h = torch.zeros(hidden_dim)
    input_tmax = len(inputs)
    for t in reversed(range(input_tmax)):
        h = encoder_forward(inputs[t], h)
    y = 0
    trans = ''
    while True:
        y, h = decoder_forward(y, h)
        y = int(F.log_softmax(y,dim=-1).topk(1)[1])
        out = vocab_decode['en'][y]
        if out == '<EOS>':
            break
        trans += out + ' '
    return trans
        

def evaluate(inputs, targets, hprev):
    with torch.no_grad(): loss,h = compute_loss(inputs, targets, hprev)
    return loss, h

def train(inputs, targets, hprev, lr):
    loss, h = compute_loss(inputs, targets, hprev)
    for param in params: param.grad = None
    loss.backward()
    for param in params:
        torch.clamp(param.grad, -5, 5, out=param.grad)
        param.data += -lr * param.grad
    return loss.detach(), h.detach()

n1 = int(.8*num_sentences)
n2 = int(.9*num_sentences)
train_range = (0, n1)
val_range = (n1, n2)
test_range = (n2, num_sentences)

# training loop

epochs = 10
train_steps = 2000
evaluate_steps = 1000
test_steps = 2000
lossi = []
batch_size = 1

for ep in range(epochs):
    for i in range(train_steps):
        hprev = torch.zeros(hidden_dim)
        ix = np.random.randint(train_range[0], train_range[1], (batch_size,))
        X, Y = tokenized_dataset[ix]
        if len(X) < 1 or len(Y) < 1: continue
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, _ = train(X[0], Y[0], hprev, lr)
        if i % 100 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
        lossi.append(torch.log10(loss))
    evaluate_loss = 0
    for i in range(evaluate_steps):
        hprev = torch.zeros(hidden_dim)
        ix = np.random.randint(train_range[0], train_range[1], (batch_size,))
        X, Y = tokenized_dataset[ix]
        if len(X) < 1 or len(Y) < 1: continue
        loss, _ = evaluate(X[0], Y[0], hprev)
        if i % 100 == 0:
            print(f'evaluate step {i}/{evaluate_steps}: {loss.item():.4f}')
            print(f"sample translation of `{' '.join(vocab_decode['de'][w] for w in X[0])}`: {sample(X[0])}")
        evaluate_loss += loss
    avg_evaluate_loss = evaluate_loss/evaluate_steps
    print(f'average validation loss: {avg_evaluate_loss.item():.4f}')
    if avg_evaluate_loss <= 3.0:
        break

test_loss = 0
hprev = torch.rand(hidden_dim, generator=rng)
for i in range(test_steps):
    ix = np.random.randint(train_range[0], train_range[1], (batch_size,))
    X, Y = tokenized_dataset[ix]
    if len(X) < 1 or len(Y) < 1: continue
    loss, hprev = evaluate(X[0], Y[0], hprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')
