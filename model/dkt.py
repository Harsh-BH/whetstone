"""L3 — Deep Knowledge Tracing (docs/03; Piech 2015): an LSTM over the interaction stream
predicting next first-attempt P(solve). Input per step = tag multi-hot ⊕ scaled difficulty
⊕ previous outcome; the hidden state carries history.

torch is confined to L3 (kept out of the L1/L2 hot path — docs/05). Honest expectation
(docs/03): at n=1 DKT may not beat L1/L2 — reported either way. Public-CF-dataset
cold-start is a documented refinement, not built here. CPU, seeded → deterministic.
"""

import torch
import torch.nn as nn

from eval.dataset import Record


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [1, T, in_dim] -> [1, T]
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out)).squeeze(-1)


class DKT:
    def __init__(self, vocab: list[str], net: _Net) -> None:
        self.vocab = vocab
        self.net = net


def _encode(records: list[Record], vocab: list[str]):
    idx = {t: i for i, t in enumerate(vocab)}
    v_n = len(vocab)
    rows, ys = [], []
    prev_y = 0.5  # unknown before the first item
    for r in records:
        vec = [0.0] * v_n
        for t in r.tags:
            if t in idx:
                vec[idx[t]] = 1.0
        vec += [r.b / 3000.0, prev_y]
        rows.append(vec)
        ys.append(float(r.y))
        prev_y = float(r.y)
    return rows, ys


def train_dkt(
    train_records: list[Record], epochs: int = 80, hidden: int = 32, seed: int = 0
) -> DKT:
    torch.manual_seed(seed)
    vocab = sorted({t for r in train_records for t in r.tags})
    x, y = _encode(train_records, vocab)
    xt = torch.tensor([x], dtype=torch.float32)
    yt = torch.tensor([y], dtype=torch.float32)
    net = _Net(len(vocab) + 2, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return DKT(vocab, net)


def predict(model: DKT, records: list[Record]) -> list[float]:
    x, _ = _encode(records, model.vocab)
    with torch.no_grad():
        p = model.net(torch.tensor([x], dtype=torch.float32))
    return p.squeeze(0).tolist()
