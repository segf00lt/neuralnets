# implementation of a translation model similar to the paper
# "Learning Phrase Representations using RNN Encoder–Decoder
# for Statistical Machine Translation"

import torch
import torch.nn.functional as F
import re
import string
import pickle
import os
import random
from unidecode import unidecode

file_names = [
        'de_sent.pkl',
        'en_sent.pkl',
        'de_vocab.pkl',
        'en_vocab.pkl',
        ]

EOS = '<<<EOS>>>'

if all([os.path.exists(f) for f in file_names]):
    files = [open(f,'rb') for f in file_names]
    de_sent = pickle.load(files[0])
    en_sent = pickle.load(files[1])
    de_vocab = pickle.load(files[2])
    en_vocab = pickle.load(files[3])
    for f in files: f.close()
else:
    def process_file(name):
        mywhitespace = ' \t\r\x0b\x0c'
        with open(name, 'r') as f:
            pat = re.compile(f"[{re.escape(string.punctuation)}{mywhitespace}]")
            text = re.sub(pat, ' ', unidecode(f.read()).lower())
            sentences = list(map(lambda s: s.split() + [EOS], text.splitlines()))
            vocab = set(text.split())
            vocab.discard('')
            vocab.add(EOS)
            vocab = list(vocab)
            vocab = dict(zip(vocab,range(len(vocab))))
        return sentences,vocab
    de_sent, de_vocab = process_file('de-en/de_short')
    en_sent, en_vocab = process_file('de-en/en_short')
    for v,name in zip([de_sent,en_sent,de_vocab,en_vocab],file_names):
        f = open(name,'wb')
        pickle.dump(v,f)
        f.close()

print(f'german vocab size: {len(de_vocab)}\nenglish vocab size: {len(en_vocab)}')

rng = torch.random.manual_seed(42)
random.seed(42)

# hyper parameters

input_vocab_len = len(en_vocab)
output_vocab_len = len(de_vocab)
hidden_dim = 200
context_len = 30

# model parameters

Wr = torch.randn((input_vocab_len, hidden_dim), generator=rng) * 0.01
Ur = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
br = torch.zeros(hidden_dim)
Vr = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
cr = torch.zeros(hidden_dim)

Wz = torch.randn((input_vocab_len, hidden_dim), generator=rng) * 0.01
Uz = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
bz = torch.zeros(hidden_dim)
Vz = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
cz = torch.zeros(hidden_dim)

Wa = torch.randn((input_vocab_len, hidden_dim), generator=rng) * 0.01
Ua = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
ba = torch.zeros(hidden_dim)
Va = torch.randn((hidden_dim, hidden_dim), generator=rng) * 0.01
ca = torch.zeros(hidden_dim)

Vy = torch.randn((hidden_dim, output_vocab_len), generator=rng) * 0.01

params = [
        Wr, Ur, br, Vr,
        Wz, Uz, bz, Vz,
        Wa, Ua, ba, Va,
        Vy
        ]
for param in params: param.requires_grad = True


def encoder_forward(x, hprev):
    pre_r = x @ Wr + hprev @ Ur + br
    r = F.sigmoid(pre_r)
    pre_z = x @ Wz + hprev @ Uz + bz
    pre_a = x @ Wa + (r * hprev) @ Ua + ba
    z = F.sigmoid(pre_z)
    a = torch.tanh(pre_a)
    h = (1 - z) * hprev + z * a
    return h

def decoder_forward(hprev):
    pre_r = hprev @ Vr + cr
    r = F.sigmoid(pre_r)
    pre_z = hprev @ Vz + cz
    pre_a = (r * hprev) @ Va + ca
    z = F.sigmoid(pre_z)
    a = torch.tanh(pre_a)
    h = (1 - z) * hprev + z * a
    y = h @ Vy
    return y, h

def compute_loss(inputs, targets, hprev):
    h = hprev.clone()
    input_tmax = len(inputs)
    output_tmax = len(targets)
    loss = 0
    for t in range(input_tmax):
        input_one_hot = F.one_hot(torch.tensor(inputs[t]),input_vocab_len).float()
        h = encoder_forward(input_one_hot, h)
    for t in range(output_tmax):
        y, h = decoder_forward(h)
        target_one_hot = F.one_hot(torch.tensor(targets[t]),output_vocab_len).float()
        loss += F.cross_entropy(y, target_one_hot)
    return loss, h

def evaluate(inputs, targets, hprev):
    with torch.no_grad(): loss,h = compute_loss(inputs, targets, hprev)
    return loss, h

def train(inputs, targets, hprev, lr):
    loss, h = compute_loss(inputs, targets, hprev)
    for param in params: param.grad = None
    loss.backward()
    for param in params: param.data += -lr * param.grad
    return loss.detach(), h.detach()

pair_sent = list(filter(lambda t: len(t[0])<=context_len and len(t[1])<=context_len, zip(en_sent,de_sent)))
random.shuffle(pair_sent)
Xsent, Ysent = list(zip(*pair_sent))
Xsent = list(Xsent)
Ysent = list(Ysent)
n1 = int(.8*len(pair_sent))
n2 = int(.9*len(pair_sent))
Xtrain, Ytrain = Xsent[:n1], Ysent[:n1]
Xval, Yval = Xsent[n1:n2], Ysent[n1:n2]
Xtest, Ytest = Xsent[n2:], Ysent[n2:]
en_longest = sorted(Xsent,key=lambda s:len(s))[-1]
de_longest = sorted(Ysent,key=lambda s:len(s))[-1]

def sentence_to_tokens(sent, vocab):
    return list(map(lambda w: vocab[w], sent))

# training loop

epochs = 3
train_steps = 90000
evaluate_steps = 30000
test_steps = 10000
lossi = []

for ep in range(epochs):
    hprev = torch.rand(hidden_dim, generator=rng)
    for i in range(train_steps):
        ix = torch.randint(0, len(Xtrain), (1,), generator=rng)
        X = sentence_to_tokens(Xtrain[ix], en_vocab)
        Y = sentence_to_tokens(Ytrain[ix], de_vocab)
        lr = 0.1 if i < (train_steps>>1) else 0.001
        loss, hprev = train(X, Y, hprev, lr)
        if i % 10000 == 0: # print every once in a while
            print(f'train step {i}/{train_steps}: {loss.item():.4f}')
        lossi.append(torch.log10(loss))
    evaluate_loss = 0
    for i in range(evaluate_steps):
        ix = torch.randint(0, len(Xval), (1,), generator=rng)
        X = sentence_to_tokens(Xval[ix], en_vocab)
        Y = sentence_to_tokens(Yval[ix], de_vocab)
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
    ix = torch.randint(0, len(Xtest), (1,), generator=rng)
    X = sentence_to_tokens(Xtest[ix], en_vocab)
    Y = sentence_to_tokens(Ytest[ix], de_vocab)
    loss, hprev = evaluate(X, Y, hprev)
    test_loss += loss

print(f'average test loss: {test_loss.item()/test_steps:.4f}')
