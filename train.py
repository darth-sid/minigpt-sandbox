# pyright: reportPrivateImportUsage=false
from data import load_data, get_batch, decode
from model import (
    MultiHeadCausalAttentionLM,
    NaiveLM,
    SingleHeadCausalAttentionLM,
    GPTLM,
)
import torch
from tqdm import tqdm, trange

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


def train_model(train, test, model, iters, batch_size, lr):
    model = model.to(device)
    block_size = model._context_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_fn(data, bs, bl):
        x, y = get_batch(data, bs, bl)
        return x.to(device), y.to(device)

    pbar = trange(iters, desc=model.name, miniters=100)

    for _ in pbar:
        x, y = batch_fn(train, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.2f}", refresh=False)

    for name, data in [("train", train), ("test", test)]:
        print(
            f"[{name}] {model.evaluate(data, batch_fn, batch_size, block_size, block_size)}",
            end=" ",
        )
    print()


train, test, vocab = load_data("input.txt")

models = [
    # NaiveLM(len(vocab)),
    # SingleHeadCausalAttentionLM(len(vocab), 32, 8),
    # MultiHeadCausalAttentionLM(len(vocab), 32, 4, 8),
    GPTLM(len(vocab), 256, 4, 2, 256),
]

for m in models:
    train_model(train, test, m, 5000, 32, 1e-3)
    print("=" * 50)
    print(
        decode(
            m.predict(torch.zeros((1, 1), dtype=torch.long, device=device), 500)[
                0
            ].tolist(),
            vocab,
        )
    )
    print("=" * 50)
