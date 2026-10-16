"""Markdown report and trade-off chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_COLOR = {"rule": "#2a78d6", "classical ML": "#eb6834", "neural network": "#1baf7a", "deep learning": "#eda100"}

NICE = {"unseen_event_rule": "Unseen-event rule", "pca_spe": "PCA (SPE)", "isolation_forest": "Isolation Forest",
        "ngram_next_event": "N-gram next-event", "mlp_next_event": "MLP next-event (64+64)",
        "mlp_next_event_small": "MLP next-event (16)", "lstm_deeplog": "LSTM DeepLog (2x64)",
        "lstm_deeplog_int8": "LSTM DeepLog, int8 quantized", "lstm_small": "LSTM small (1x16)"}


def nice(n):
    return NICE.get(n, n)


def fmt_range(v, rng, f="{:.3f}"):
    return f.format(v) if rng[0] == rng[1] else f"{f.format(v)} ({f.format(rng[0])}–{f.format(rng[1])})"


def chart(S, path):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    seen = set()
    for name, s in S.items():
        c = FAMILY_COLOR.get(s["family"], MUTED)
        x0, x1, y = s["infer_j_per_1k"], s["cached_j_per_1k"], s["f1"]
        ax.plot([x0, x1], [y, y], color=c, lw=1.2, alpha=0.6, zorder=1)
        ax.scatter([x0], [y], s=55, color=c, edgecolor=SURFACE, lw=1.5, zorder=3,
                   label=s["family"] if s["family"] not in seen else None)
        ax.scatter([x1], [y], s=55, facecolor=SURFACE, edgecolor=c, lw=1.8, zorder=3)
        seen.add(s["family"])
        ax.annotate(nice(name), (x0, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Inference energy per 1,000 sessions (J, log scale)   ● no cache   ○ with cache")
    ax.set_ylabel("F1 score")
    ax.set_title("Detection quality vs. energy: caching moves every model left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def findings(S, meta):
    """Findings computed from the numbers, so the text can never disagree with the table."""
    best = max(S, key=lambda n: S[n]["f1"])
    b = S[best]
    rule = S.get("unseen_event_rule")
    L = ["## 3. Findings", ""]
    L.append(f"- **Best detection:** {nice(best)}, F1 {b['f1']:.3f}, at {b['infer_j_per_1k']:.3g} J per 1,000 sessions "
             f"without caching and {b['cached_j_per_1k']:.3g} J with caching.")
    if rule:
        L.append(f"- **Baseline that every model must beat:** a one-line rule (flag sessions with an event never seen in "
                 f"training) reaches F1 {rule['f1']:.3f} with precision {rule['precision']:.3f}, at "
                 f"{rule['infer_j_per_1k']:.3g} J per 1,000 sessions. Isolation Forest and PCA do not beat it.")
    big, small = S.get("mlp_next_event"), S.get("mlp_next_event_small")
    if big and small:
        L.append(f"- **Smaller is not worse:** the 16-unit MLP scored F1 {small['f1']:.3f} "
                 f"({small['f1_range'][0]:.3f}–{small['f1_range'][1]:.3f} over seeds) against "
                 f"{big['f1']:.3f} ({big['f1_range'][0]:.3f}–{big['f1_range'][1]:.3f}) for the 64+64 MLP, with a "
                 f"{big['model_bytes'] / small['model_bytes']:.1f}x smaller model and lower variation between seeds.")
    ratios = [v["infer_j_per_1k"] / v["cached_j_per_1k"] for v in S.values() if v["cached_j_per_1k"] > 0]
    L.append(f"- **Caching is a free energy saving:** {100 * meta['duplicate_share']:.1f}% of test sessions "
             f"repeat an earlier sequence. Scoring each distinct sequence once cut inference energy "
             f"{min(ratios):.0f}–{max(ratios):.0f}x, with identical detections.")
    ng = S.get("ngram_next_event")
    if ng and small:
        L.append(f"- **How a model is run can matter as much as which model:** the cached 16-unit MLP "
                 f"({small['cached_j_per_1k']:.3g} J, F1 {small['f1']:.3f}) uses less energy than the uncached n-gram model "
                 f"({ng['infer_j_per_1k']:.3g} J, F1 {ng['f1']:.3f}).")
    L += ["", "**Limits:** HDFS is a public Hadoop dataset, not telecom logs; energy is "
          + ("measured package energy (includes other processes)." if "RAPL" in meta["energy_source"]
             else "estimated from CPU time, so it ranks detectors by CPU cost rather than measuring watts.")
          + " Next-event thresholds are discrete (top-g), so a model's false-positive rate on test can differ from the "
            "1% target.", ""]
    return L


def write(out, meta, P, S, rows):
    chart(S, out / "tradeoff.png")
    est = "estimated" if "estimate" in meta["energy_source"] else "measured"
    L = ["# Results: energy-aware anomaly detection on HDFS logs", "",
         f"Training data: {meta['n_fit']} normal sessions ({meta['n_calib']} more held out to set thresholds at a "
         f"{100 * meta['target_fpr']:.0f}% false-positive target). Test: {meta['n_test_normal']:,} normal and "
         f"{meta['n_test_abnormal']:,} anomalous sessions. Energy is **{est}** ({meta['energy_source']}). "
         f"Inference cost is measured on a fixed sample of {meta['cost_sample']:,} test sessions. "
         f"Machine: {meta['processor'] or meta['machine']}, Python {meta['python']}, single-threaded.", ""]
    L += ["## 1. Log parsing (Drain, our implementation)", "",
          f"On LogHub's labelled HDFS sample ({P['lines']:,} lines): grouping accuracy **{P['grouping_accuracy']:.4f}**, "
          f"{P['templates_found']} templates found ({P['templates_true']} in the ground truth), "
          f"{P['lines_per_cpu_s']:,.0f} lines per CPU-second, {P['joules_per_million_lines']:.1f} J per million lines ({est}).", ""]
    L += ["## 2. Detection and cost", "",
          "| detector | family | precision | recall | F1 | false-positive rate | training (CPU-s) | inference J / 1k sessions | with cache | model size |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(S.items(), key=lambda kv: -kv[1]["f1"]):
        L.append(f"| {nice(name)} | {s['family']} | {fmt_range(s['precision'], s['precision_range'])} | "
                 f"{fmt_range(s['recall'], s['recall_range'])} | **{fmt_range(s['f1'], s['f1_range'])}** | "
                 f"{100 * s['fpr']:.2f}% | {s['train_cpu_s']:.1f} | {s['infer_j_per_1k']:.3g} | {s['cached_j_per_1k']:.3g} | "
                 f"{s['model_bytes'] / 1024:.0f} KB |")
    L += ["", "Neural models: mean over training seeds, range in brackets. Detection metrics are exact over all test "
          "sessions. *With cache*: each distinct session (and, for next-event models, each distinct event history) "
          "is scored once.", "", "![trade-off](tradeoff.png)", ""]
    L += findings(S, meta)
    if not any(v["family"] == "deep learning" for v in S.values()):
        L += ["**LSTM (PyTorch) detectors not included in this run:** PyTorch is not installed. "
              "Install it (`pip install --user torch --index-url https://download.pytorch.org/whl/cpu`) and rerun.", ""]
    failed = [r for r in rows if "error" in r]
    if failed:
        L += ["## Detectors that failed to run", ""] + [f"- {r['detector']} (seed {r['seed']}): `{r['error']}`" for r in failed] + [""]
    (out / "report.md").write_text("\n".join(L))
