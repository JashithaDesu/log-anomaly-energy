"""DeepLog-style LSTM next-event detector (Du et al., CCS 2017) in PyTorch,
with the two classic ways of making it lighter:
  - a smaller architecture (fewer units and layers)
  - post-training dynamic int8 quantization of the LSTM and output layers

Optional: run.py skips these detectors when PyTorch is not installed.
CPU-only, single-threaded, so costs are comparable with the other detectors.
"""
import io

import numpy as np

from logad.data import N_EVENTS
from logad.detectors import _NextEvent

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the environment
    torch = None

VOCAB = N_EVENTS + 2  # padding 0, events 1..28, end-of-session marker


if torch is not None:
    class _Net(nn.Module):
        def __init__(self, hidden, layers):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 16)
            self.lstm = nn.LSTM(16, hidden, layers, batch_first=True)
            self.out = nn.Linear(hidden, VOCAB)

        def forward(self, x):
            o, _ = self.lstm(self.emb(x))
            return self.out(o[:, -1, :])


class LSTMDetector(_NextEvent):
    family = "deep learning"

    def __init__(self, hidden=64, layers=2, name="lstm_deeplog", seed=0, epochs=10, quantize=False):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.hidden, self.layers, self.name = hidden, layers, name
        self.seed, self.epochs, self.quantize = seed, epochs, quantize

    def fit(self, sessions):
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        pairs = [w for s in sessions for w in self.windows(s)]
        X = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        y = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        net = _Net(self.hidden, self.layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss_fn(net(X[b]), y[b]).backward()
                opt.step()
        net.eval()
        if self.quantize:
            net = torch.ao.quantization.quantize_dynamic(net, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        self.m_ = net
        self._cache = {}
        return self

    def _top(self, hists, g):
        with torch.no_grad():
            logits = self.m_(torch.tensor(hists, dtype=torch.long))
        return torch.topk(logits, g, dim=1).indices.numpy()

    def topg(self, hist, g):
        key = (hist, g)
        if key not in self._cache:
            self._cache[key] = set(self._top([hist], g)[0].tolist())
        return self._cache[key]

    def predict(self, sessions):
        if not self.use_cache:
            out = np.zeros(len(sessions), dtype=int)
            for start in range(0, len(sessions), 2000):
                chunk = sessions[start:start + 2000]
                wins = [(i, h, n) for i, s in enumerate(chunk) for h, n in self.windows(s)]
                top = self._top([w[1] for w in wins], self.g_)
                for (i, _, nxt), row in zip(wins, top):
                    if nxt not in row:
                        out[start + i] = 1
            return out
        hists = list({h for s in sessions for h, _ in self.windows(s)} - {k[0] for k in self._cache if k[1] == self.g_})
        for start in range(0, len(hists), 4096):
            batch = hists[start:start + 4096]
            for hist, row in zip(batch, self._top(batch, self.g_)):
                self._cache[(hist, self.g_)] = set(row.tolist())
        return super().predict(sessions)

    def model_bytes(self):
        buf = io.BytesIO()
        torch.save(self.m_.state_dict(), buf)
        return buf.getbuffer().nbytes


def torch_detectors(seeds=(0, 1, 2)):
    if torch is None:
        return []
    dets = []
    for seed in seeds:
        dets += [LSTMDetector(64, 2, "lstm_deeplog", seed),
                 LSTMDetector(64, 2, "lstm_deeplog_int8", seed, quantize=True),
                 LSTMDetector(16, 1, "lstm_small", seed)]
    return dets
