import torch


class Vocab:
    def __init__(self, data: str):
        self.itos = sorted(set(data))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __getitem__(self, k: str | int) -> int | str:
        if isinstance(k, str):
            return self.stoi[k]
        elif isinstance(k, int):
            return self.itos[k]
        else:
            raise TypeError("Vocab keys must be either str or int")

    def __len__(self) -> int:
        return len(self.itos)


torch.manual_seed(1234)


def encode(raw: str, vocab: Vocab) -> list[int]:
    return [vocab[c] for c in raw]


def decode(encoded: list[int], vocab: Vocab) -> str:
    return "".join([vocab[i] for i in encoded])


def train_test_split(data: list[int], p=0.9) -> tuple[list[int], list[int]]:
    split_pt = int(len(data) * p)
    return data[:split_pt], data[split_pt:]


def get_batch(
    data: list[int], batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def load_data(fname: str) -> tuple[torch.Tensor, torch.Tensor, Vocab]:
    with open(fname, "r") as fdata:
        data = fdata.read()

    vocab = Vocab(data)

    encoded = encode(data, vocab)
    decoded = decode(encoded, vocab)

    assert decoded == data
    train, test = train_test_split(torch.tensor(encoded, dtype=torch.long))
    return train, test, vocab


train, test, vocab = load_data("input.txt")
