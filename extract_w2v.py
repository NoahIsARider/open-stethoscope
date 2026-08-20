#!/usr/bin/env python3
"""Extract wav2vec2-base (frozen) features for all CirCor wavs -> npz.
Per-wav: mean-pooled last hidden state (768-d). Per-patient: mean over locations.
Output: /root/heart-train/w2v_features.npz {wav_path: feat}
"""
import os, sys, re, glob, time
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

W2V_DIR = '/root/wav2vec2'
DATA_DIR = '/root/heart-data/training_data'
OUT = '/root/heart-train/w2v_features.npz'
SR_TARGET = 16000

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device}', flush=True)
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(W2V_DIR)
    model = Wav2Vec2Model.from_pretrained(W2V_DIR)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] wav2vec2-base params={n_params/1e6:.1f}M', flush=True)

    wavs = sorted(glob.glob(os.path.join(DATA_DIR, '*.wav')))
    print(f'[data] {len(wavs)} wavs', flush=True)
    feats = {}
    batch = []
    batch_paths = []
    t0 = time.time()
    n_done = 0
    for w in wavs:
        x, sr = sf.read(w, dtype='float32')
        if sr != SR_TARGET:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR_TARGET)
        batch.append(x)
        batch_paths.append(w)
        if len(batch) >= 2:
            inp = feat_ext(batch, sampling_rate=SR_TARGET, return_tensors='pt', padding=True,
                           return_attention_mask=True)
            with torch.no_grad():
                out = model(input_values=inp['input_values'].to(device),
                            attention_mask=inp['attention_mask'].to(device))
                last = out.last_hidden_state  # B x T x 768
                T = last.shape[1]
                mf = torch.nn.functional.interpolate(
                    inp['attention_mask'].unsqueeze(1).float().to(device), size=T, mode='nearest')
                mf = mf.squeeze(1).bool()
                pooled = (last * mf.unsqueeze(-1)).sum(1) / mf.sum(1).clamp(min=1).unsqueeze(-1)  # B x 768
            for p, fv in zip(batch_paths, pooled.cpu().numpy()):
                feats[p] = fv
            batch, batch_paths = [], []
            n_done += 2
            if n_done % 200 == 0:
                print(f'[feat] {n_done}/{len(wavs)} ({time.time()-t0:.0f}s)', flush=True)
    if batch:
        inp = feat_ext(batch, sampling_rate=SR_TARGET, return_tensors='pt', padding=True,
                       return_attention_mask=True)
        with torch.no_grad():
            out = model(input_values=inp['input_values'].to(device),
                        attention_mask=inp['attention_mask'].to(device))
            last = out.last_hidden_state
            T = last.shape[1]
            mf = torch.nn.functional.interpolate(
                inp['attention_mask'].unsqueeze(1).float().to(device), size=T, mode='nearest')
            mf = mf.squeeze(1).bool()
            pooled = (last * mf.unsqueeze(-1)).sum(1) / mf.sum(1).clamp(min=1).unsqueeze(-1)
        for p, fv in zip(batch_paths, pooled.cpu().numpy()):
            feats[p] = fv
    # save keyed by basename for easy join
    save = {os.path.basename(p): v for p, v in feats.items()}
    np.savez(OUT, **save)
    print(f'[save] {OUT} ({len(save)} wavs, {time.time()-t0:.0f}s total)', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
