# pyright: reportPrivateImportUsage=false
from data import load_data, get_batch, decode
from model import MultiHeadCausalAttentionLM, NaiveLM, SingleHeadCausalAttentionLM
import torch
from tqdm import tqdm, trange


def train_model(train, test, model, iters, batch_size, block_size, lr, print_freq=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in trange(iters):
        x, y = get_batch(train, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_freq is not None and i % print_freq == 0:
            tqdm.write(str(loss.item()))
    for name, data in [("train", train), ("test", test)]:
        print(
            f"{name}:{model.evaluate(data, get_batch, batch_size, block_size, block_size)}"
        )


train, test, vocab = load_data("input.txt")

models = [
    NaiveLM(len(vocab)),
    SingleHeadCausalAttentionLM(len(vocab), 32, 8),
    MultiHeadCausalAttentionLM(len(vocab), 32, 4, 8),
]

for m in models:
    train_model(train, test, m, 10000, 32, 8, 1e-2, print_freq=500)
    print(decode(m.predict(torch.zeros((1, 1), dtype=torch.long))[0].tolist(), vocab))
