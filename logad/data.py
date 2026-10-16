"""HDFS sessions as event-ID sequences, plus helpers shared by the detectors."""
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
N_EVENTS = 28  # event IDs 1..28 in DeepLog's HDFS split


def load_sessions(name, data_dir=DATA):
    return [tuple(int(x) for x in line.split()) for line in open(Path(data_dir) / name) if line.strip()]


def load_split(data_dir=DATA, calib_frac=0.2, seed=0):
    """Training normals split into fit / calibration (thresholds are set on
    calibration normals only, never on test data). Test = normal + abnormal."""
    train = load_sessions("hdfs_train", data_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    cut = int(len(train) * (1 - calib_frac))
    fit = [train[i] for i in idx[:cut]]
    calib = [train[i] for i in idx[cut:]]
    normal = load_sessions("hdfs_test_normal", data_dir)
    abnormal = load_sessions("hdfs_test_abnormal", data_dir)
    return fit, calib, normal, abnormal


def unique_with_counts(sessions):
    """Distinct sequences and how often each occurs. Scoring each distinct
    sequence once is exact for deterministic detectors, and is itself an
    energy optimisation (a result cache)."""
    c = Counter(sessions)
    return list(c.keys()), np.array(list(c.values()))


def count_vectors(sessions):
    """Event-count matrix (sessions x events), the classic input for PCA and
    Isolation Forest log anomaly detection (Xu et al., SOSP 2009)."""
    X = np.zeros((len(sessions), N_EVENTS + 1), dtype=np.float32)
    for i, s in enumerate(sessions):
        for e in s:
            X[i, e] += 1
    return X[:, 1:]
