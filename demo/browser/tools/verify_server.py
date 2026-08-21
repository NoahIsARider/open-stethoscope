#!/usr/bin/env python3
"""Runs on the P4 training server: librosa 1.0.0 mel + torch model → truth."""
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/root/open-stethoscope")
import train_v4 as T  # noqa: E402

CKPT = "/root/open-stethoscope/demo/hmc-screening/models/best_model_v4_s43_30ep.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = T.FusionNet(n_class=3)
model.load_state_dict(ckpt["state_dict"])
model.eval()

results = {}
for wav in sys.argv[1:]:
    x, sr = sf.read(wav, dtype="float32")
    if sr != T.SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=T.SR)
    m = librosa.feature.melspectrogram(y=x, sr=T.SR, n_fft=T.N_FFT, hop_length=T.HOP,
                                       n_mels=T.N_MELS, fmin=T.FMIN, fmax=T.FMAX)
    m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    m = m[:, :T.N_FRAMES] if m.shape[1] >= T.N_FRAMES else np.pad(
        m, ((0, 0), (0, T.N_FRAMES - m.shape[1])))
    xb = np.zeros((1, 4, 1, T.N_MELS, T.N_FRAMES), np.float32)
    xb[0, T.LOC2IDX["MV"], 0] = m
    mask = np.zeros((1, 4), dtype=bool)
    mask[0, T.LOC2IDX["MV"]] = True
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(xb), torch.from_numpy(mask))
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    results[os.path.basename(wav)] = {
        "probs": probs.tolist(),
        "mel": m.tolist(),
        "mel_mean": float(m.mean()),
        "mel_max": float(m.max()),
        "librosa": __import__("librosa").__version__,
    }
print(json.dumps(results, indent=1))
