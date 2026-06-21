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
    def evaluate(model, data, get_batch, batch_size, block_size, eval_iters=200):
        was_training = model.training
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        if was_training:
            model.train()
        return losses.mean().item()


class NaiveLM(LM):
    def __init__(self, vocab_size: int):
        super().__init__(1)
        self._table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        y = self._table(x)
        loss = None
        if targets is not None:
            loss = cross_entropy(y.view(-1, y.shape[-1]), targets.view(-1))
        return y, loss


def _causal_mask(x):
    _, T = x.shape
    return torch.tril(torch.ones(T, T, device=x.device))


class AttentionHead(torch.nn.Module):
    def __init__(self, d_embed: int, d_head: int):
        super().__init__()
        self._Wq = torch.nn.Linear(d_embed, d_head, bias=False)
        self._Wk = torch.nn.Linear(d_embed, d_head, bias=False)
        self._Wv = torch.nn.Linear(d_embed, d_head, bias=False)
        self._dk = d_head

    def forward(self, x, mask=None):  # x: (B, T, d_embed)
        Q, K, V = self._Wq(x), self._Wk(x), self._Wv(x)  # (B, T, d_head)
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

    def forward(self, x, targets=None):  # (B, T), (B, T)
        mask = _causal_mask(x)
        positions = torch.arange(x.shape[1], device=x.device)
        pos_embeddings = self._pos_embed(positions)  # (B, T, d_embed)
        embeddings = self._embed(x)  # (B, T, d_embed)
        x = embeddings + pos_embeddings

        x = self._attention(x, mask)  # (B, T, d_embed)
        y = self._unembed(x)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(y.view(-1, y.shape[-1]), targets.view(-1))
        return y, loss


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_embed: int, n_head: int):
        super().__init__()
        d_head = d_embed // n_head
        self._heads = torch.nn.ModuleList(
            [AttentionHead(d_embed, d_head) for _ in range(n_head)]
        )
        self._blend = torch.nn.Linear(d_embed, d_embed)

    def forward(self, x, mask=None):
        return self._blend(torch.cat([att(x, mask) for att in self._heads], dim=-1))


class MultiHeadCausalAttentionLM(LM):
    def __init__(self, vocab_size: int, d_embed: int, n_head: int, context_size: int):
        super().__init__(context_size)
        self._embed = torch.nn.Embedding(vocab_size, d_embed)
        self._pos_embed = torch.nn.Embedding(context_size, d_embed)
        self._attention = MultiHeadAttention(d_embed, n_head)
        self._unembed = torch.nn.Linear(d_embed, vocab_size)

    def forward(self, x, targets=None):
        mask = _causal_mask(x)
        positions = torch.arange(x.shape[1], device=x.device)
        pos_embeddings = self._pos_embed(positions)  # (B, T, d_embed)
        embeddings = self._embed(x)  # (B, T, d_embed)
        x = embeddings + pos_embeddings

        x = self._attention(x, mask)
        y = self._unembed(x)  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(y.view(-1, y.shape[-1]), targets.view(-1))
        return y, loss


class FFN(torch.nn.Module):
    def __init__(self, d_embed: int, d_ffn):
        super().__init__()
        self._l1 = torch.nn.Linear(d_embed, d_ffn)
        self._relu = torch.nn.ReLU()
        self._l2 = torch.nn.Linear(d_ffn, d_embed)

    def forward(self, x):
        return self._l2(self._relu(self._l1(x)))


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_embed: int, n_head: int, dilation: int = 4):
        super().__init__()
        self._attention = MultiHeadAttention(d_embed, n_head)
        self._norm1 = torch.nn.LayerNorm(d_embed)
        self._ffn = FFN(d_embed, dilation * d_embed)
        self._norm2 = torch.nn.LayerNorm(d_embed)

    def forward(self, x, mask=None):
        x = self._attention(self._norm1(x), mask) + x
        x = self._ffn(self._norm2(x)) + x
        return x


class GPTLM(LM):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        n_head: int,
        n_block: int,
        context_size: int,
    ):
        super().__init__(context_size)
        self._embed = torch.nn.Embedding(vocab_size, d_embed)
        self._pos_embed = torch.nn.Embedding(context_size, d_embed)
        self._blocks = torch.nn.ModuleList(
            [TransformerBlock(d_embed, n_head) for _ in range(n_block)]
        )
        self._norm = torch.nn.LayerNorm(d_embed)
        self._unembed = torch.nn.Linear(d_embed, vocab_size)

    def forward(self, x, targets=None):
        mask = _causal_mask(x)
        positions = torch.arange(x.shape[1], device=x.device)
        pos_embeddings = self._pos_embed(positions)  # (B, T, d_embed)
        embeddings = self._embed(x)  # (B, T, d_embed)
        x = embeddings + pos_embeddings

        for block in self._blocks:
            x = block(x, mask)

        y = self._unembed(self._norm(x))  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = cross_entropy(y.view(-1, y.shape[-1]), targets.view(-1))
        return y, loss
