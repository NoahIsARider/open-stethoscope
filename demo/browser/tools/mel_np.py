#!/usr/bin/env python3
"""Pure-numpy replica of the librosa 1.0.0 preprocessing the model expects.

This is the *reference* for the browser JS implementation (js/mel.js): both
load the same mel_params.json and perform, in order:
  zero-pad (n_fft//2 each side, constant) → frame (hop) → hann window →
  rfft → power → mel filterbank → power_to_db(ref=max) → trim/pad to n_frames.

Usage:
    from mel_np import mel_spectrogram, to_frames, build_input
"""
import json
import os

import numpy as np

_PARAMS = None


def params():
    global _PARAMS
    if _PARAMS is None:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "assets", "mel_params.json")) as f:
            _PARAMS = json.load(f)
    return _PARAMS


def mel_spectrogram(x: np.ndarray, p: dict | None = None) -> np.ndarray:
    """x: float32 mono at p['sr'] Hz → (n_mels, n_frames) dB mel spectrogram."""
    p = p or params()
    n_fft, hop = p["n_fft"], p["hop"]
    win = np.asarray(p["window"], dtype=np.float32)
    fb = np.asarray(p["filterbank"], dtype=np.float32)

    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="constant")
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_frames]
    X = np.fft.rfft(frames * win, axis=1)
    power = (X.real ** 2 + X.imag ** 2).astype(np.float32)  # power=2.0
    mel = fb @ power.T  # (n_mels, n_frames)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10) / np.max(mel))
    db = np.maximum(db, db.max() - 80.0)  # librosa power_to_db top_db=80.0 default
    return db.astype(np.float32)


def to_frames(m: np.ndarray, p: dict | None = None) -> np.ndarray:
    """Trim to the first n_frames, or zero-pad on the right (inference rule)."""
    p = p or params()
    nf = p["n_frames"]
    if m.shape[1] >= nf:
        return m[:, :nf]
    return np.pad(m, ((0, 0), (0, nf - m.shape[1])))


def build_input(x: np.ndarray, loc_idx: int, p: dict | None = None):
    """Model input (1,4,1,n_mels,n_frames) + bool mask for one location."""
    p = p or params()
    m = to_frames(mel_spectrogram(x, p), p)
    xb = np.zeros((1, 4, 1, p["n_mels"], p["n_frames"]), np.float32)
    xb[0, loc_idx, 0] = m
    mask = np.zeros((1, 4), dtype=bool)
    mask[0, loc_idx] = True
    return xb, mask
