#!/usr/bin/env python3
"""
ensemble_v4.py -- average softmax probs across saved v4 test npz files.

Usage: python3 ensemble_v4.py tag1 tag2 ...   (npz = v4_test_{tag}.npz)
Prints per-model test metrics + averaged ensemble metrics (fused and vote)
+ confusion matrix.
"""
import os, sys, glob
import numpy as np

WORKDIR = '/root/heart-train'
CH_W = np.array([1.0, 3.0, 5.0])
WA_W = np.array([0.1, 0.2, 1.0])
CLASS_NAMES = ['Absent', 'Unknown', 'Present']


def metrics(y_true, y_pred, n=3):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = (y_true == y_pred).mean()
    recall = np.array([cm[i, i] / max(1, cm[i].sum()) for i in range(n)])
    prec = np.array([cm[i, i] / max(1, cm[:, i].sum()) for i in range(n)])
    f1 = 2 * prec * recall / (prec + recall + 1e-9)
    wa_legacy = float((WA_W * recall).sum() / WA_W.sum())
    ch_num = float((CH_W * np.diag(cm)).sum())
    ch_den = float((CH_W * cm.sum(1)).sum())
    ch_wa = ch_num / ch_den if ch_den > 0 else float('nan')
    return acc, recall, f1, wa_legacy, ch_wa, cm


def show(name, y, p):
    acc, recall, f1, wa_leg, ch_wa, cm = metrics(y, p)
    print(f'--- {name} ---')
    print(f'  accuracy      : {acc:.4f}')
    print(f'  macro-F1      : {f1.mean():.4f}  (per-class {np.round(f1,3).tolist()})')
    print(f'  legacy wacc   : {wa_leg:.4f}')
    print(f'  CHALLENGE WA  : {ch_wa:.4f} (official s_murmur)')
    print(f'  per-class recall/sens: {np.round(recall,3).tolist()}')
    print(f'  confusion (rows=true [Absent,Unknown,Present], cols=pred):')
    print('  ' + str(cm.tolist()))
    return {'accuracy': float(acc), 'macro_f1': float(f1.mean()), 'challenge_wa': float(ch_wa),
            'recall': np.round(recall, 4).tolist(), 'cm': cm.tolist()}


def main():
    tags = sys.argv[1:]
    if not tags:
        tags = sorted(os.path.basename(p)[8:-4] for p in glob.glob(os.path.join(WORKDIR, 'v4_test_*.npz')))
    probs_f, probs_v, y = [], [], None
    per_model = []
    for t in tags:
        p = os.path.join(WORKDIR, f'v4_test_{t}.npz')
        d = np.load(p)
        if y is None:
            y = d['y']
        assert (y == d['y']).all(), f'y mismatch for {t}'
        probs_f.append(d['fused_probs']); probs_v.append(d['vote_probs'])
        r = show(f'MODEL {t} (fused)', y, d['fused_pred'])
        show(f'MODEL {t} (vote)', y, d['vote_pred'])
        per_model.append((t, r))
    print('\n================ ENSEMBLE ================')
    pf = np.mean(probs_f, axis=0)
    pv = np.mean(probs_v, axis=0)
    r1 = show(f'ENSEMBLE fused (avg {len(tags)} seeds)', y, pf.argmax(1))
    r2 = show(f'ENSEMBLE vote  (avg {len(tags)} seeds)', y, pv.argmax(1))
    print('\n--- summary ---')
    print(f'{"model":28s} {"acc":>8s} {"macroF1":>8s} {"s_murmur":>9s}  recall[A,U,P]')
    for t, r in per_model:
        print(f'{t:28s} {r["accuracy"]:8.4f} {r["macro_f1"]:8.4f} {r["challenge_wa"]:9.4f}  {r["recall"]}')
    print(f'{"ENSEMBLE-fused":28s} {r1["accuracy"]:8.4f} {r1["macro_f1"]:8.4f} {r1["challenge_wa"]:9.4f}  {r1["recall"]}')
    print(f'{"ENSEMBLE-vote":28s} {r2["accuracy"]:8.4f} {r2["macro_f1"]:8.4f} {r2["challenge_wa"]:9.4f}  {r2["recall"]}')


if __name__ == '__main__':
    main()
