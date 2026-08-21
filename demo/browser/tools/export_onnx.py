#!/usr/bin/env python3
"""Export the trained FusionNet to ONNX for the in-browser demo, plus the
librosa-equivalent mel parameters (window + filterbank) as JSON.

Run from the repo root:
    python3 demo/browser/tools/export_onnx.py

Outputs:
    demo/browser/assets/model.onnx       (~1.6 MB)
    demo/browser/assets/mel_params.json  (hann window + slaney mel filterbank)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

# train_v4 imports librosa at module level but only *calls* it inside
# functions (mel/preproc at runtime) — none of which the export path needs.
# Stub it out so we can import the model definition without installing
# librosa + numba here; the mel math is replicated exactly in mel_params.
import types  # noqa: E402

_librosa = types.ModuleType("librosa")
_feature = types.ModuleType("librosa.feature")


def _stub(*a, **k):
    raise NotImplementedError("librosa stub — mel computed in mel_params")


_feature.melspectrogram = _stub
_librosa.feature = _feature
_librosa.resample = _stub
_librosa.power_to_db = _stub
sys.modules["librosa"] = _librosa

import train_v4 as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
CKPT = os.path.join(ROOT, "demo", "hmc-screening", "models", "best_model_v4_s43_30ep.pt")
ONNX_PATH = os.path.join(ASSETS, "model.onnx")
MEL_PATH = os.path.join(ASSETS, "mel_params.json")


# ─── librosa 1.0.0-equivalent mel filterbank (Slaney scale, norm='slaney') ──

def hz_to_mel_slaney(f):
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    f = np.asarray(f, dtype=np.float64)
    mels = f / f_sp
    if mels.ndim == 0:
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else mels
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def mel_to_hz_slaney(m):
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    m = np.asarray(m, dtype=np.float64)
    hz_lin = m * f_sp
    return np.where(hz_lin >= min_log_hz, min_log_hz * np.exp(logstep * (m - min_log_mel)), hz_lin)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, norm='slaney')."""
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = np.linspace(hz_to_mel_slaney(fmin), hz_to_mel_slaney(fmax), n_mels + 2)
    hz = mel_to_hz_slaney(mel_f)
    fb = np.zeros((n_mels, len(fftfreqs)), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (fftfreqs - lo) / (mid - lo)
        right = (hi - fftfreqs) / (hi - mid)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]  # norm='slaney'
    return fb.astype(np.float32)


def main():
    os.makedirs(ASSETS, exist_ok=True)

    # ── 1. Export ONNX ─────────────────────────────────────────────────────
    print(f"[1/3] loading {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = T.FusionNet(n_class=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"      arch={ckpt.get('arch')} seed={ckpt.get('seed')} "
          f"epoch={ckpt.get('epoch')}")

    x = torch.zeros(1, 4, 1, T.N_MELS, T.N_FRAMES, dtype=torch.float32)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True  # AV filled for shape inference

    print(f"[2/3] exporting ONNX → {ONNX_PATH}")
    torch.onnx.export(
        model, (x, mask), ONNX_PATH,
        input_names=["x", "mask"],
        output_names=["logits", "aux", "alpha"],
        opset_version=17,
        dynamo=False,
    )

    # ── 2. Verify ONNX == torch ────────────────────────────────────────────
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    xt = torch.randn(1, 4, 1, T.N_MELS, T.N_FRAMES)
    mk = torch.zeros(1, 4, dtype=torch.bool)
    mk[0, 1] = True  # PV only
    with torch.no_grad():
        logits_t, aux_t, alpha_t = model(xt, mk)
    ort_out = sess.run(None, {"x": xt.numpy(), "mask": mk.numpy()})
    for name, a, b in zip(["logits", "aux", "alpha"], [logits_t, aux_t, alpha_t], ort_out):
        err = float((a.detach().numpy() - b).__abs__().max())
        print(f"      {name}: max abs diff = {err:.3e}")
        assert err < 1e-4, f"{name} mismatch"

    # ── 3. Dump mel params ─────────────────────────────────────────────────
    print(f"[3/3] dumping mel params → {MEL_PATH}")
    win = np.hanning(T.N_FFT).astype(np.float32)  # scipy.signal.windows.hann(sym=True)
    fb = mel_filterbank(T.SR, T.N_FFT, T.N_MELS, T.FMIN, T.FMAX)
    params = {
        "sr": T.SR, "n_fft": T.N_FFT, "hop": T.HOP, "n_mels": T.N_MELS,
        "fmin": T.FMIN, "fmax": T.FMAX, "n_frames": T.N_FRAMES,
        "window": win.tolist(), "filterbank": fb.tolist(),
        "locs": T.LOCS, "classes": T.CLASS_NAMES,
        "note": "librosa 1.0.0-equivalent: hann(sym) window, Slaney mel, norm='slaney', power_to_db(ref=max)",
    }
    with open(MEL_PATH, "w") as f:
        json.dump(params, f)
    print(f"      window={len(win)} filterbank={fb.shape} → "
          f"{os.path.getsize(MEL_PATH)//1024} KB")
    print("✅ export done")


if __name__ == "__main__":
    main()
