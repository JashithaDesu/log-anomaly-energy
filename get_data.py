"""get_data.py - fetch the public HDFS log data and verify it.

Two public sources, pinned to exact commits:
  - DeepLog's HDFS session split (Du et al., CCS 2017), as published in the
    logdeep repository: event-ID sequences, one HDFS block (session) per line.
      hdfs_train          4,855 normal sessions (training)
      hdfs_test_normal  553,366 normal sessions
      hdfs_test_abnormal 16,838 anomalous sessions
  - LogHub's raw HDFS sample (HDFS_2k.log) with hand-labelled event templates,
    used to measure the accuracy of our own log parser.

Every file is checked against data/SHA256SUMS.

Usage: python3 get_data.py
"""
import hashlib, shutil, subprocess, sys, tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "https://github.com/donglee-afar/logdeep": ("ec18b29f95f089380568aee552843272e7e79164",
        {"data/hdfs/hdfs_train": "hdfs_train", "data/hdfs/hdfs_test_normal": "hdfs_test_normal",
         "data/hdfs/hdfs_test_abnormal": "hdfs_test_abnormal"}),
    "https://github.com/logpai/loghub": ("dd61d0952749ee7963bde24220d1be5ede023033",
        {"HDFS/HDFS_2k.log": "HDFS_2k.log", "HDFS/HDFS_2k.log_structured.csv": "HDFS_2k.log_structured.csv"}),
}


def expected():
    return {name: h for h, name in (l.split() for l in (DATA / "SHA256SUMS").read_text().splitlines() if l.strip())}


def verify():
    ok = True
    for name, want in expected().items():
        p = DATA / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        ok &= got == want
        print(f"{'OK  ' if got == want else 'FAIL'} {name}")
    return ok


def fetch():
    for url, (commit, files) in SOURCES.items():
        if all((DATA / dst).exists() for dst in files.values()):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True)
            subprocess.run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", url, commit], check=True)
            subprocess.run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"], check=True)
            for src, dst in files.items():
                shutil.copy(Path(tmp) / src, DATA / dst)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch()
    sys.exit(0 if verify() else "checksum mismatch")
