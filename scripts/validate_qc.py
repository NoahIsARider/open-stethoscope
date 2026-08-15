"""Validate the QC engine against real CirCor recordings.

Runs the full QC analysis on a sample of real expert-annotated PCG wavs and
reports the metric distributions (SNR, detected heart rate, spectral flatness,
duration) plus how many would pass/fail each QC gate. This proves the thresholds
are sane on genuine heart sounds, not synthetic audio.

Usage:
  python scripts/validate_qc.py --data-dir <training_data> [--max-n 60]
"""
import os, sys, argparse, glob
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from qc_engine import analyze, FLATNESS_BAD, FLATNESS_WARN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-n", type=int, default=60)
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True))[: args.max_n]
    if not wavs:
        sys.exit("[validate] no wavs found")

    rows = []
    fail_metrics = {}
    for p in wavs:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        res = analyze(x, sr)
        rows.append({
            "file": os.path.basename(p),
            "dur": res["duration_s"],
            "snr": res["snr_db"],
            "hr": res["heart_rate_bpm"],
            "flat": res["spectral_flatness"],
            "n_fail": sum(1 for c in res["checks"] if c["status"] == "fail"),
            "n_warn": sum(1 for c in res["checks"] if c["status"] == "warn"),
        })
        for c in res["checks"]:
            if c["status"] == "fail":
                fail_metrics[c["metric"]] = fail_metrics.get(c["metric"], 0) + 1

    snrs = [r["snr"] for r in rows if r["snr"] is not None]
    hrs = [r["hr"] for r in rows if r["hr"] is not None]
    flats = [r["flat"] for r in rows]
    print(f"[validate] {len(rows)} real CirCor recordings")
    print(f"  duration   : mean {np.mean([r['dur'] for r in rows]):.1f}s  "
          f"min {min(r['dur'] for r in rows):.1f}s  max {max(r['dur'] for r in rows):.1f}s")
    print(f"  SNR dB     : mean {np.mean(snrs):.1f}  median {np.median(snrs):.1f}  "
          f"{sum(1 for s in snrs if s >= 12)}/{len(snrs)} >=12dB  "
          f"{sum(1 for s in snrs if s < 6)}/{len(snrs)} <6dB")
    print(f"  HR bpm     : detected {len(hrs)}/{len(rows)}  "
          f"mean {np.mean(hrs):.0f}  range {min(hrs):.0f}-{max(hrs):.0f}  "
          f"in 60-100 {sum(1 for h in hrs if 60 <= h <= 100)}/{len(hrs)}")
    print(f"  flatness   : mean {np.mean(flats):.3f}  "
          f">={FLATNESS_BAD} (fail) {sum(1 for f in flats if f >= FLATNESS_BAD)}/{len(flats)}  "
          f">={FLATNESS_WARN} (warn) {sum(1 for f in flats if f >= FLATNESS_WARN)}/{len(flats)}")
    print(f"  fail breakdown: {fail_metrics}")
    print(f"  zero-fail    : {sum(1 for r in rows if r['n_fail'] == 0)}/{len(rows)}  "
          f"(>=1 warn) {sum(1 for r in rows if r['n_fail'] == 0 and r['n_warn'] >= 1)}/{len(rows)}")


if __name__ == "__main__":
    main()
