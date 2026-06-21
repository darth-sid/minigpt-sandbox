import torch
from torch.nn.functional import cross_entropy, softmax


class LM(torch.nn.Module):
    def __init__(self, context_size: int):
        super().__init__()
        self._context_size = context_size

    @torch.no_grad()
    def predict(self, inputs, max_new_tokens=100):
        # (B, ?, vocab)
        for _ in range(max_new_tokens):
            next_pred = self.forward(inputs[:, -self._context_size :])[0][:, -1, :]
            next_token = torch.multinomial(next_pred.softmax(dim=-1), num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs


class Naive(LM):
    def __init__(self, vocab_size: int):
        super().__init__(1)
        self._table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        pred = self._table(inputs)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss


class AttentionHead(torch.nn.Module):
    def __init__(self, d_embed: int, d_head: int):
        super().__init__()
        self._Wq = torch.nn.Linear(d_embed, d_head, bias=False)
        self._Wk = torch.nn.Linear(d_embed, d_head, bias=False)
        self._Wv = torch.nn.Linear(d_embed, d_head, bias=False)
        self._dk = d_head

    def forward(self, inputs, mask=None):  # inputs: (B, T, d_embed)
        Q, K, V = self._Wq(inputs), self._Wk(inputs), self._Wv(inputs)  # (B, T, d_head)
        QKT = Q @ K.transpose(-2, -1) / (self._dk**0.5)  # (B, T, T)
        if mask is not None:
            QKT = QKT.masked_fill(mask == 0, float("-inf"))
        return softmax(QKT, dim=-1) @ V  # (B, T, d_head)


class SingleHeadCausalAttention(LM):
    def __init__(self, vocab_size: int, d_embed: int, context_size: int):
        super().__init__(context_size)
        self._embed = torch.nn.Embedding(vocab_size, d_embed)
        self._attention = AttentionHead(d_embed, d_embed)
        self._unembed = torch.nn.Linear(d_embed, vocab_size)

    def forward(self, inputs, targets=None):  # (B, T), (B, T)
        _, T = inputs.shape
        mask = torch.tril(torch.ones(T, T, device=inputs.device))
        embeddings = self._embed(inputs)  # (B, T, d_embed)
        att_out = self._attention(embeddings, mask)  # (B, T, d_embed)
        pred = self._unembed(att_out)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss
