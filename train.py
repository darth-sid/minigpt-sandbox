from data import load_data, get_batch, decode
from model import Naive, SingleHeadCausalAttention
import torch
from tqdm import tqdm, trange


def train_model(data, model, iters, batch_size, block_size, lr, print_freq=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in trange(iters):
        x, y = get_batch(data, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_freq is not None and i % print_freq == 0:
            tqdm.write(str(loss.item()))


train, test, vocab = load_data("input.txt")

models = [Naive(len(vocab)), SingleHeadCausalAttention(len(vocab), 32, 8)]

for m in models:
    train_model(train, m, 5000, 32, 8, 1e-2, print_freq=500)
    print(decode(m.predict(torch.zeros((1, 1), dtype=torch.long))[0].tolist(), vocab))
