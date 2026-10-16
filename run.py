"""run.py - energy-aware log anomaly detection on HDFS:  python3 run.py  ->  results/

1. Parsing: our Drain implementation on LogHub's labelled HDFS sample
   (grouping accuracy, throughput, energy).
2. Detection: every detector learns from normal sessions only; its threshold
   is set on held-out normal sessions (1% false-positive target); it is then
   evaluated on all 570,204 test sessions.
3. Cost: training cost, and inference cost per 1,000 sessions measured on a
   fixed sample of test sessions, both without caching and with a cache of
   repeated sequences/histories. Model size on disk.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")  # single-threaded: CPU time comparable across detectors

import csv, json, pickle, platform, sys, warnings
from pathlib import Path

import numpy as np

from logad import energy
from logad.data import load_split, unique_with_counts
from logad.deeplog_torch import torch_detectors
from logad.detectors import all_detectors
from logad.parser import Drain, grouping_accuracy, hdfs_content

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"
TARGET_FPR = 0.01
COST_SAMPLE = 20000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def model_size(det):
    if hasattr(det, "model_bytes"):
        return det.model_bytes()
    cache = getattr(det, "_cache", None)
    if cache is not None:
        det._cache = {}
    n = len(pickle.dumps(det))
    if cache is not None:
        det._cache = cache
    return n


def parsing():
    data = Path(__file__).resolve().parent / "data"
    lines = open(data / "HDFS_2k.log").read().splitlines()
    truth = [r["EventId"] for r in csv.DictReader(open(data / "HDFS_2k.log_structured.csv"))]
    reps = 50  # the sample is small; repeat to get a measurable cost
    with energy.measure() as m:
        for _ in range(reps):
            d = Drain()
            pred = [d.add(hdfs_content(l)) for l in lines]
    return {"lines": len(lines), "templates_found": len(d.templates), "templates_true": len(set(truth)),
            "grouping_accuracy": grouping_accuracy(pred, truth),
            "lines_per_cpu_s": len(lines) * reps / m.cpu_s,
            "joules_per_million_lines": m.energy_j / (len(lines) * reps) * 1e6, "energy_source": m.energy_source}


def evaluate(det, fit, calib, test_u, test_c, test_y, sample):
    row = {"detector": det.name, "family": det.family, "seed": getattr(det, "seed", None)}
    with energy.measure() as m:
        det.fit(fit)
        det.calibrate(calib, TARGET_FPR)
    row.update(train_cpu_s=m.cpu_s, train_j=m.energy_j)
    if hasattr(det, "g_"):
        row["g"] = det.g_
    pred = det.predict(test_u)                       # exact: deterministic per distinct sequence
    tp = int((pred * test_c * test_y).sum())
    fp = int((pred * test_c * (1 - test_y)).sum())
    n_ab, n_no = int((test_c * test_y).sum()), int((test_c * (1 - test_y)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_ab
    row.update(tp=tp, fp=fp, fn=n_ab - tp, precision=prec, recall=rec,
               f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
               recall_ci=wilson(tp, n_ab), fpr=fp / n_no, fpr_ci=wilson(fp, n_no))
    # Inference cost on the same fixed sample, without and with caching.
    if hasattr(det, "_cache"):
        det._cache = {}
        det.use_cache = False
    with energy.measure() as m:
        det.predict(sample)
    row.update(infer_cpu_s_per_1k=m.cpu_s / len(sample) * 1000, infer_j_per_1k=m.energy_j / len(sample) * 1000)
    if hasattr(det, "_cache"):
        det._cache = {}
        det.use_cache = True
    with energy.measure() as m:
        uniq, _ = unique_with_counts(sample)
        det.predict(uniq)
    row.update(cached_cpu_s_per_1k=m.cpu_s / len(sample) * 1000, cached_j_per_1k=m.energy_j / len(sample) * 1000,
               model_bytes=model_size(det))
    return row


def summarise(rows):
    """Mean and range over seeds for neural detectors; single value otherwise."""
    keys = ["precision", "recall", "f1", "fpr", "train_cpu_s", "train_j", "infer_cpu_s_per_1k", "infer_j_per_1k",
            "cached_cpu_s_per_1k", "cached_j_per_1k", "model_bytes"]
    out = {}
    for name in dict.fromkeys(r["detector"] for r in rows if "error" not in r):
        rs = [r for r in rows if r["detector"] == name and "error" not in r]
        out[name] = {"family": rs[0]["family"], "runs": len(rs)}
        for k in keys:
            v = [r[k] for r in rs]
            out[name][k] = float(np.mean(v))
            out[name][k + "_range"] = (float(min(v)), float(max(v)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    fit, calib, normal, abnormal = load_split()
    test = normal + abnormal
    labels = [0] * len(normal) + [1] * len(abnormal)
    uniq = {}
    for s, y in zip(test, labels):
        c = uniq.setdefault((s, y), 0)
        uniq[(s, y)] = c + 1
    test_u = [k[0] for k in uniq]
    test_y = np.array([k[1] for k in uniq])
    test_c = np.array(list(uniq.values()))
    rng = np.random.default_rng(0)
    sample = [test[i] for i in rng.choice(len(test), COST_SAMPLE, replace=False)]
    dup_share = 1 - len(set(test)) / len(test)
    print(f"{len(fit)} fit / {len(calib)} calibration normals; test {len(normal)} normal + {len(abnormal)} abnormal "
          f"({len(set(test))} distinct sequences, {100 * dup_share:.1f}% repeats)")
    print("energy:", "RAPL measured" if energy.rapl_readable() else f"estimated at {energy.WATTS_PER_CORE:.2f} W per busy core")

    P = parsing()
    print(f"parser: grouping accuracy {P['grouping_accuracy']:.4f}, {P['lines_per_cpu_s']:.0f} lines/CPU-s")

    rows = []
    for det in all_detectors() + torch_detectors():
        try:
            r = evaluate(det, fit, calib, test_u, test_c, test_y, sample)
        except Exception as e:  # keep the experiment going; record what failed
            r = {"detector": det.name, "seed": getattr(det, "seed", None), "error": repr(e)}
        rows.append(r)
        print("  " + (f"{r['detector']:22} seed {r['seed']}  F1 {r['f1']:.3f}  train {r['train_cpu_s']:6.1f} CPU-s  "
                      f"infer {r['infer_cpu_s_per_1k']:7.3f} -> {r['cached_cpu_s_per_1k']:7.3f} CPU-s/1k (cached)"
                      if "error" not in r else f"{r['detector']}: FAILED {r['error']}"))
    S = summarise(rows)
    meta = {"python": platform.python_version(), "machine": platform.machine(), "processor": platform.processor(),
            "energy_source": "RAPL" if energy.rapl_readable() else f"estimate {energy.WATTS_PER_CORE} W/core",
            "target_fpr": TARGET_FPR, "cost_sample": COST_SAMPLE, "duplicate_share": dup_share,
            "n_fit": len(fit), "n_calib": len(calib), "n_test_normal": len(normal), "n_test_abnormal": len(abnormal)}
    json.dump({"meta": meta, "parsing": P, "runs": rows, "summary": S}, open(OUT / "metrics.json", "w"), indent=2, default=float)
    from logad import report
    report.write(OUT, meta, P, S, rows)
    print(f"-> {OUT}/report.md")


if __name__ == "__main__":
    sys.exit(main())
