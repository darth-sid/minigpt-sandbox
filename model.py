import torch
from torch.nn.functional import cross_entropy, softmax


class Naive(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self._table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        pred = self._table(inputs)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss

    @torch.no_grad()
    def predict(self, inputs, max_new_tokens=100):
        for _ in range(max_new_tokens):
            prediction, _ = self.forward(inputs)
            next_token = torch.multinomial(
                prediction[:, -1, :].softmax(dim=-1), num_samples=1
            )
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
