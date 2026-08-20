#!/usr/bin/env python3
"""Tune decision offsets (dP, dU) on OOF predictions to see s_murmur headroom."""
import os
import json
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(base, 'experiments', 'results') if os.path.isdir(os.path.join(base, 'experiments', 'results')) else base
r = json.load(open(os.path.join(res_dir, 'kfold_results.json')))
# rebuild OOF probs from per-fold npz? We didn't save per-fold probs; use kfold_results OOF summary only.
# Instead: recompute OOF probs by loading fold models? Heavy. For now: simulate from saved npz of seeds.
# Quick approach: load the 4 seed test npz (s42,s43,s44,s45) + their val probs to tune.
import glob, os

# Load val+test probs for the seed family
probs_dir = os.path.join(base, 'experiments', 'probs') if os.path.isdir(os.path.join(base, 'experiments', 'probs')) else base
files = [os.path.join(probs_dir, f'v4_probs_{t}_30ep.npz') for t in ('s42', 's43', 's44', 's45')]
CH_W = np.array([1.0, 3.0, 5.0])

def smurmur(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    num = float((CH_W * np.diag(cm)).sum())
    den = float((CH_W * cm.sum(1)).sum())
    return num / den if den > 0 else 0.0

def tune(probs, y, grid=0.2, rng_d=(-1.2, 1.2)):
    best = (-1, None)
    dP = np.arange(rng_d[0], rng_d[1] + 1e-9, grid)
    dU = np.arange(rng_d[0], rng_d[1] + 1e-9, grid)
    for a in dP:
        for b in dU:
            p = probs.copy()
            p[:, 1] += b   # Unknown
            p[:, 2] += a   # Present
            pred = p.argmax(1)
            s = smurmur(y, pred)
            if s > best[0]:
                best = (s, (round(float(a), 2), round(float(b), 2)))
    return best

# per-model: tune on VAL, apply to TEST (like original), report
for f in files:
    if not os.path.exists(f):
        print('missing', f); continue
    d = np.load(f)
    vp, vy = d['val_fused'], d['val_y']
    tp, ty = d['test_fused'], d['test_y']
    s0, _ = tune(vp, vy)
    s_test0 = smurmur(ty, tp.argmax(1))
    best_s, best_d = tune(vp, vy)
    # apply to test
    p = tp.copy()
    p[:, 1] += best_d[1]; p[:, 2] += best_d[0]
    s_test = smurmur(ty, p.argmax(1))
    tag = os.path.basename(f).replace('v4_probs_', '').replace('.npz', '')
    print(f'{tag}: argmax_test={s_test0:.4f} tuned({best_d}) val={best_s:.4f} -> test={s_test:.4f}')

# ensemble: mean of all 4 seeds, tune on val, apply test
vp_all = np.mean([np.load(f)['val_fused'] for f in files], axis=0)
vy_all = np.load(files[0])['val_y']
tp_all = np.mean([np.load(f)['test_fused'] for f in files], axis=0)
ty_all = np.load(files[0])['test_y']
s0 = smurmur(ty_all, tp_all.argmax(1))
bs, bd = tune(vp_all, vy_all)
p = tp_all.copy(); p[:, 1] += bd[1]; p[:, 2] += bd[0]
print(f'ENS4: argmax_test={s0:.4f} tuned({bd}) val={bs:.4f} -> test={smurmur(ty_all, p.argmax(1)):.4f}')
