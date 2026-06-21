# pyright: reportPrivateImportUsage=false
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

    @torch.no_grad()
    def evaluate(model, data, get_batch, batch_size, context_size, eval_iters=200):
        was_training = model.training
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, model._context_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        if was_training:
            model.train()
        return losses.mean().item()


class NaiveLM(LM):
    def __init__(self, vocab_size: int):
        super().__init__(1)
        self._table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        pred = self._table(inputs)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss


def _causal_mask(inputs):
    _, T = inputs.shape
    return torch.tril(torch.ones(T, T, device=inputs.device))


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


class SingleHeadCausalAttentionLM(LM):
    def __init__(self, vocab_size: int, d_embed: int, context_size: int):
        super().__init__(context_size)
        self._embed = torch.nn.Embedding(vocab_size, d_embed)
        self._pos_embed = torch.nn.Embedding(context_size, d_embed)
        self._attention = AttentionHead(d_embed, d_embed)
        self._unembed = torch.nn.Linear(d_embed, vocab_size)

    def forward(self, inputs, targets=None):  # (B, T), (B, T)
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        pos_embeddings = self._pos_embed(positions)  # (B, T, d_embed)
        embeddings = self._embed(inputs)  # (B, T, d_embed)
        x = embeddings + pos_embeddings

        att_out = self._attention(x, _causal_mask(inputs))  # (B, T, d_embed)
        pred = self._unembed(att_out)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_embed: int, n_head: int):
        super().__init__()
        d_head = d_embed // n_head
        self._heads = torch.nn.ModuleList(
            [AttentionHead(d_embed, d_head) for _ in range(n_head)]
        )
        self._blend = torch.nn.Linear(d_embed, d_embed)

    def forward(self, inputs, mask=None):
        return self._blend(
            torch.cat([att(inputs, mask) for att in self._heads], dim=-1)
        )


class MultiHeadCausalAttentionLM(LM):
    def __init__(self, vocab_size: int, d_embed: int, n_head: int, context_size: int):
        super().__init__(context_size)
        self._embed = torch.nn.Embedding(vocab_size, d_embed)
        self._pos_embed = torch.nn.Embedding(context_size, d_embed)
        self._attention = MultiHeadAttention(d_embed, n_head)
        self._unembed = torch.nn.Linear(d_embed, vocab_size)

    def forward(self, inputs, targets=None):
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        pos_embeddings = self._pos_embed(positions)  # (B, T, d_embed)
        embeddings = self._embed(inputs)  # (B, T, d_embed)
        x = embeddings + pos_embeddings

        mask = _causal_mask(inputs)
        att_out = self._attention(x, mask)
        pred = self._unembed(att_out)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(pred.view(-1, pred.shape[-1]), targets.view(-1))
        return pred, loss
