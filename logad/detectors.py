"""Detectors, from cheapest to most expensive. All learn from normal sessions only.

Each detector:
    fit(sessions)              learn normal behaviour
    calibrate(sessions, fpr)   set its threshold on held-out normal sessions
    predict(sessions) -> 0/1   1 = anomalous session
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from logad.data import N_EVENTS, count_vectors


class UnseenEvent:
    """Rule: a session containing an event type never seen in normal training
    data is anomalous. No parameters, no model: the energy floor."""
    name = "unseen_event_rule"
    family = "rule"

    def fit(self, sessions):
        self.known_ = {e for s in sessions for e in s}
        return self

    def calibrate(self, sessions, fpr):
        return self

    def predict(self, sessions):
        return np.array([int(any(e not in self.known_ for e in s)) for s in sessions])


class _ScoreThreshold:
    """Score-based detectors: threshold = (1 - fpr) quantile of calibration scores."""

    def calibrate(self, sessions, fpr):
        s = self.score(sessions)
        self.thr_ = float(np.quantile(s, 1 - fpr)) if len(s) else np.inf
        return self

    def predict(self, sessions):
        return (self.score(sessions) > self.thr_).astype(int)


class PCADetector(_ScoreThreshold):
    """Count vectors projected onto normal-behaviour principal components;
    the squared residual (SPE) measures how unusual a session is."""
    name = "pca_spe"
    family = "classical ML"

    def fit(self, sessions):
        X = count_vectors(sessions)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=0.95, svd_solver="full").fit(self.scaler_.transform(X))
        return self

    def score(self, sessions):
        Z = self.scaler_.transform(count_vectors(sessions))
        R = Z - self.pca_.inverse_transform(self.pca_.transform(Z))
        return (R ** 2).sum(axis=1)


class IForestDetector(_ScoreThreshold):
    name = "isolation_forest"
    family = "classical ML"

    def fit(self, sessions):
        self.m_ = IsolationForest(n_estimators=100, random_state=0, n_jobs=1).fit(count_vectors(sessions))
        return self

    def score(self, sessions):
        return -self.m_.score_samples(count_vectors(sessions))


class _NextEvent:
    """DeepLog-style detection: predict the next event from the previous `h`;
    a session is anomalous if any actual next event is outside the model's
    top-g predictions. g is chosen on calibration normals: the smallest g whose
    false-positive rate there is at most the target.

    use_cache=True memoises predictions for repeated event histories; False
    recomputes every window (used to measure the uncached cost)."""
    h = 10
    use_cache = True
    END = N_EVENTS + 1   # end-of-session marker: a session that stops early is itself a deviation

    def windows(self, s):
        seq = tuple(s) + (self.END,)
        padded = (0,) * self.h + seq
        return [(padded[i:i + self.h], padded[i + self.h]) for i in range(len(seq))]

    def calibrate(self, sessions, fpr):
        for g in range(1, N_EVENTS + 2):
            self.g_ = g
            if self.predict(sessions).mean() <= fpr:
                break
        return self

    def predict(self, sessions):
        out = np.zeros(len(sessions), dtype=int)
        for i, s in enumerate(sessions):
            for hist, nxt in self.windows(s):
                if nxt not in self.topg(hist, self.g_):
                    out[i] = 1
                    break
        return out


class NGramDetector(_NextEvent):
    """Next-event model built from counts of event histories (with back-off to
    shorter histories). A few kilobytes of dictionaries: no training loop."""
    name = "ngram_next_event"
    family = "classical ML"
    h = 3

    def fit(self, sessions):
        from collections import Counter, defaultdict
        self.counts_ = defaultdict(Counter)
        for s in sessions:
            for hist, nxt in self.windows(s):
                for k in range(self.h + 1):           # all suffixes of the history
                    self.counts_[hist[len(hist) - k:]][nxt] += 1
        self._cache = {}
        return self

    def _lookup(self, hist, g):
        for k in range(self.h, -1, -1):
            c = self.counts_.get(hist[len(hist) - k:])
            if c:
                return {e for e, _ in c.most_common(g)}
        return set()

    def topg(self, hist, g):
        if not self.use_cache:
            return self._lookup(hist, g)
        key = (hist, g)
        if key not in self._cache:
            self._cache[key] = self._lookup(hist, g)
        return self._cache[key]


class MLPDetector(_NextEvent):
    """Neural next-event model: a multi-layer perceptron over one-hot encoded
    windows of the last 10 events. `hidden` sets its size, so the same model
    can be run in a full and a lightweight configuration."""
    family = "neural network"

    def __init__(self, hidden=(64, 64), name="mlp_next_event", seed=0):
        self.hidden, self.name, self.seed = hidden, name, seed

    def _encode(self, hists):
        V = N_EVENTS + 2  # padding 0, events 1..28, end marker
        X = np.zeros((len(hists), self.h * V), dtype=np.float32)
        for i, hist in enumerate(hists):
            for j, e in enumerate(hist):
                X[i, j * V + e] = 1
        return X

    def fit(self, sessions):
        pairs = [w for s in sessions for w in self.windows(s)]
        X = self._encode([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        self.m_ = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=200, early_stopping=True,
                                random_state=self.seed).fit(X, y)
        self._cache = {}
        return self

    def topg(self, hist, g):
        key = (hist, g)
        if key not in self._cache:
            p = self.m_.predict_proba(self._encode([hist]))[0]
            self._cache[key] = {self.m_.classes_[i] for i in np.argsort(-p)[:g]}
        return self._cache[key]

    def predict(self, sessions):
        if not self.use_cache:
            # Uncached: one forward pass per window of every session.
            out = np.zeros(len(sessions), dtype=int)
            for start in range(0, len(sessions), 2000):
                chunk = sessions[start:start + 2000]
                wins = [(i, h, n) for i, s in enumerate(chunk) for h, n in self.windows(s)]
                P = self.m_.predict_proba(self._encode([w[1] for w in wins]))
                top = np.argsort(-P, axis=1)[:, :self.g_]
                cls = self.m_.classes_
                for (i, _, nxt), row in zip(wins, top):
                    if nxt not in cls[row]:
                        out[start + i] = 1
            return out
        # Cached: batch every distinct history once, then look up.
        hists = {hist for s in sessions for hist, _ in self.windows(s)} - {k[0] for k in self._cache if k[1] == self.g_}
        if hists:
            hists = list(hists)
            P = self.m_.predict_proba(self._encode(hists))
            order = np.argsort(-P, axis=1)[:, :self.g_]
            for hist, row in zip(hists, order):
                self._cache[(hist, self.g_)] = {self.m_.classes_[i] for i in row}
        return super().predict(sessions)


def all_detectors(seeds=(0, 1, 2)):
    """Deterministic detectors once; neural ones once per training seed."""
    dets = [UnseenEvent(), PCADetector(), IForestDetector(), NGramDetector()]
    for seed in seeds:
        dets += [MLPDetector((64, 64), "mlp_next_event", seed), MLPDetector((16,), "mlp_next_event_small", seed)]
    return dets
