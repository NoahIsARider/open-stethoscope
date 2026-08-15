#!/usr/bin/env python3
"""
tune_ensemble_w.py -- val-weighted ensemble + threshold tuning.

Usage: python3 tune_ensemble_w.py tag1 tag2 ... [--power 2]
Weights probs by val s_murmur^power before averaging, then tunes 2D
threshold offsets (dP, dU) on val and reports test metrics.
"""
import os, sys
import numpy as np

WORKDIR = '/root/heart-train'
CH_W = np.array([1.0, 3.0, 5.0])
WA_W = np.array([0.1, 0.2, 1.0])


def smurmur(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return float((CH_W * np.diag(cm)).sum()) / float((CH_W * cm.sum(1)).sum())


def metrics(y_true, y_pred, n=3):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = (y_true == y_pred).mean()
    recall = np.array([cm[i, i] / max(1, cm[i].sum()) for i in range(n)])
    prec = np.array([cm[i, i] / max(1, cm[:, i].sum()) for i in range(n)])
    f1 = 2 * prec * recall / (prec + recall + 1e-9)
    ch_wa = float((CH_W * np.diag(cm)).sum()) / float((CH_W * cm.sum(1)).sum())
    return acc, recall, f1, ch_wa, cm


def tune(probs, y_val):
    best = (-1.0, None, None)
    for dP in np.arange(-1.6, 1.61, 0.2):
        for dU in np.arange(-1.2, 1.21, 0.2):
            adj = probs * np.exp(np.array([0.0, dU, dP]))[None, :]
            s = smurmur(y_val, adj.argmax(1))
            if s > best[0]:
                best = (s, dP, dU)
    return best


def main():
    args = sys.argv[1:]
    power = 2.0
    if '--power' in args:
        i = args.index('--power'); power = float(args[i + 1]); del args[i:i + 2]
    tags = args
    vf, tf, vy, ty = [], [], None, None
    vals = []
    for t in tags:
        d = np.load(os.path.join(WORKDIR, f'v4_probs_{t}.npz'))
        vf.append(d['val_fused']); tf.append(d['test_fused'])
        if vy is None:
            vy, ty = d['val_y'], d['test_y']
        assert (vy == d['val_y']).all() and (ty == d['test_y']).all()
        vals.append(smurmur(vy, d['val_fused'].argmax(1)))
    vals = np.array(vals)
    print(f'[load] {tags}')
    print(f'[info] val s_murmur per model: {np.round(vals,4).tolist()}')
    print(f'[info] weights (val^power, power={power}): {np.round(vals**power,4).tolist()}')
    w = vals ** power
    w = w / w.sum()
    vp = sum(wi * pi for wi, pi in zip(w, vf))
    tp = sum(wi * pi for wi, pi in zip(w, tf))
    val_s, dP, dU = tune(vp, vy)
    adj = tp * np.exp(np.array([0.0, dU, dP]))[None, :]
    acc, recall, f1, ch, cm = metrics(ty, adj.argmax(1))
    print(f'--- WEIGHTED ENSEMBLE ({len(tags)} models) tuned dP={dP:.2f} dU={dU:.2f} (val {val_s:.4f}) ---')
    print(f'  accuracy      : {acc:.4f}')
    print(f'  macro-F1      : {f1.mean():.4f}  (per-class {np.round(f1,3).tolist()})')
    print(f'  CHALLENGE WA  : {ch:.4f} (official s_murmur)')
    print(f'  per-class recall/sens: {np.round(recall,3).tolist()}')
    print(f'  confusion (rows=true [Absent,Unknown,Present], cols=pred):')
    print('  ' + str(cm.tolist()))


if __name__ == '__main__':
    main()
