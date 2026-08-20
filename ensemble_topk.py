#!/usr/bin/env python3
"""Ensemble analysis: rank seeds by val s_murmur, build top-k probavg ensembles, report test.
Usage: python3 ensemble_topk.py  (requires v4_probs_<tag>.npz for all trained seeds)"""
import os
import glob, os
import numpy as np

CH_W = np.array([1.0, 3.0, 5.0])
probs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'probs')
NPZ = os.path.join(probs_dir if os.path.isdir(probs_dir) else os.path.dirname(os.path.abspath(__file__)), 'v4_probs_*.npz')


def smurmur(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    num = float((CH_W * np.diag(cm)).sum())
    den = float((CH_W * cm.sum(1)).sum())
    return num / den if den > 0 else 0.0, cm


def metrics(y, p):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    acc = (y == p).mean()
    rec = np.array([cm[i, i] / max(1, cm[i].sum()) for i in range(3)])
    return acc, rec, cm


files = sorted(glob.glob(NPZ))
data = {}
for f in files:
    tag = os.path.basename(f).replace('v4_probs_', '').replace('.npz', '')
    d = np.load(f)
    data[tag] = d
print(f'loaded {len(data)}: {sorted(data.keys())}', flush=True)

rows = []
for tag, d in data.items():
    s_v, _ = smurmur(d['val_y'], d['val_fused'].argmax(1))
    s_t, _ = smurmur(d['test_y'], d['test_fused'].argmax(1))
    rows.append((tag, s_v, s_t))
rows.sort(key=lambda r: -r[1])
print('\nseed ranking by VAL s_murmur:')
for tag, sv, st in rows:
    print(f'  {tag}: val={sv:.4f} test={st:.4f}', flush=True)

# top-k ensembles
ref = data[rows[0][0]]
vy, ty = ref['val_y'], ref['test_y']
print('\ntop-k probavg ensembles (selected by val, evaluated on test):')
for k in range(1, min(len(rows), 6) + 1):
    sel = [r[0] for r in rows[:k]]
    vp = np.mean([data[t]['val_fused'] for t in sel], axis=0)
    tp = np.mean([data[t]['test_fused'] for t in sel], axis=0)
    sv, _ = smurmur(vy, vp.argmax(1))
    st, cm = smurmur(ty, tp.argmax(1))
    acc, rec, _ = metrics(ty, tp.argmax(1))
    print(f'  top{k} {sel}: val={sv:.4f} test={st:.4f} acc={acc:.4f} rec={np.round(rec,3).tolist()}', flush=True)
    print(f'     cm={cm.tolist()}', flush=True)

# tuned offsets on top ensembles (coarse grid, val)
print('\ntop-k ensembles + val-tuned offsets:')
for k in [1, 2, 3, 4]:
    sel = [r[0] for r in rows[:k]]
    vp = np.mean([data[t]['val_fused'] for t in sel], axis=0)
    tp = np.mean([data[t]['test_fused'] for t in sel], axis=0)
    best = (-1, None)
    for a in np.arange(-1.0, 1.01, 0.2):
        for b in np.arange(-1.0, 1.01, 0.2):
            p = vp.copy(); p[:, 1] += b; p[:, 2] += a
            s, _ = smurmur(vy, p.argmax(1))
            if s > best[0]:
                best = (s, (round(float(a), 2), round(float(b), 2)))
    p = tp.copy(); p[:, 1] += best[1][1]; p[:, 2] += best[1][0]
    st, cm = smurmur(ty, p.argmax(1))
    print(f'  top{k} tuned dP={best[1][0]} dU={best[1][1]} (val {best[0]:.4f}) -> test={st:.4f}', flush=True)
print('DONE', flush=True)
