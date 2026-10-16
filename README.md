# Energy-aware anomaly detection on system logs

**How much detection quality do you get for each joule spent?**

This project compares anomaly detectors for system logs, from a one-line rule to neural networks,
on both **detection performance** and **computational cost**: training time, inference energy and
model size. It also tests two energy-saving techniques, smaller architectures and caching, and an
optional int8-quantized LSTM.

## Pipeline

1. **Parse** raw log lines into event templates with our own implementation of Drain (He et al., 2017).
2. **Group** events into sessions (one HDFS block = one session) and learn normal behaviour from normal sessions only.
3. **Detect** anomalous sessions; thresholds are set on held-out normal sessions (1% false-positive target), never on test data.
4. **Measure** precision, recall and F1 on 570,204 test sessions, plus the CPU time and energy of training and inference.

## Data

- [DeepLog](https://dl.acm.org/doi/10.1145/3133956.3134015)'s HDFS split (Du et al., CCS 2017), as published in the
  [logdeep](https://github.com/donglee-afar/logdeep) repository: 4,855 normal training sessions; 553,366 normal and
  16,838 anomalous test sessions, as event-ID sequences.
- [LogHub](https://github.com/logpai/loghub)'s labelled raw HDFS sample (2,000 lines), to measure parsing accuracy.

`get_data.py` fetches both from pinned commits and checks them against `data/SHA256SUMS`.

## Results

Full tables and chart: [`results/report.md`](results/report.md).

| detector | F1 (3 seeds for neural models) | inference CPU-s / 1,000 sessions | with cache | training CPU-s |
|---|---|---|---|---|
| MLP next-event, 16 units | 0.923 (0.915–0.930) | 0.42 | 0.017 | 40 |
| MLP next-event, 64+64 units | 0.911 (0.903–0.923) | 0.52 | 0.022 | 58 |
| LSTM DeepLog, 2x64 | 0.898 (0.833–0.936) | 3.4 | 0.067 | 254 |
| LSTM DeepLog, int8 quantized | 0.898 (0.833–0.936) | 4.9 | 0.084 | 236 |
| LSTM small, 1x16 | 0.788 (0.692–0.933) | 0.55 | 0.019 | 40 |
| N-gram next-event | 0.797 | | | |
| Unseen-event rule | 0.633 | | | |
| PCA | 0.594 | | | |
| Isolation Forest | 0.014 | | | |

Run on an Intel i5-8350U laptop, single-threaded; energy is estimated from CPU time (see below), so all exact
costs are in [`results/report.md`](results/report.md). Detection metrics are exact and identical on every run;
timings vary by roughly ±30% between runs, so read costs as orders of magnitude.

- **A 16-unit MLP beat the DeepLog-style LSTM:** F1 0.923 vs 0.898, at about 8x lower inference cost and 6x lower
  training cost, and with far less variation between training seeds.
- **Quantization backfired on this CPU:** int8 dynamic quantization shrank the model 3.5x (225 KB to 65 KB) and kept
  F1 the same, but made inference about 40% slower. An "energy optimisation" has to be measured on the target hardware.
- **Small is not automatically good:** the 16-unit LSTM's F1 ranged from 0.69 to 0.93 depending only on the seed.
- **A one-line rule sets the bar:** flagging sessions that contain an event never seen in training reaches F1 0.63
  with 97% precision for almost no energy. PCA and Isolation Forest do not beat it.
- **Caching is a free energy saving:** 96.8% of sessions repeat an earlier sequence; scoring each distinct
  sequence (or event history) once cut inference cost 2–50x with identical detections (verified by a test).
- **Parsing:** our Drain implementation reaches 0.9975 grouping accuracy on LogHub's labelled HDFS sample.

![trade-off](results/tradeoff.png)

## Energy measurement

Energy is read from Intel RAPL when the user may read `/sys/class/powercap/intel-rapl:0/energy_uj`. On most Linux
systems that needs root, so otherwise it is **estimated** as CPU-seconds x average power per busy core (default
15 W / 4 cores, the lab machine's Intel i5-8350U; set `LOGAD_WATTS_PER_CORE` for another machine). All runs are
single-threaded. The report always states which method was used.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: LSTM detectors
python3 get_data.py
python3 run.py                       # minutes without PyTorch; the LSTMs add roughly half an hour on a laptop
python3 -m unittest discover -s tests -t .
```

## Limitations

- HDFS is a public Hadoop dataset, not telecom logs; it is the standard benchmark in this field.
- Estimated energy ranks detectors by CPU cost; it is not a wattmeter reading.
- Next-event thresholds are discrete (top-g), so test false-positive rates can differ from the 1% target.
- Inference cost is measured in Python; absolute numbers would be lower in an optimised serving stack.

## Layout
