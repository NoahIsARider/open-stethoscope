#!/usr/bin/env python3
"""
tune_ensemble.py -- threshold-tuned ensembling on val, evaluated on test.

Loads v4_probs_{tag}.npz for each tag. Combines models by:
  - probavg : mean of softmax probs
  - logitavg: mean of log-probs (geometric mean of probs)
Tunes 2D decision offsets (dP for Present, dU for Unknown) on VAL by
maximizing val s_murmur (official challenge metric). Applies best offsets
to TEST and reports full metrics + confusion matrix per ensemble config.

Usage: python3 tune_ensemble.py --mode probavg tag1 tag2 ...
"""
import os, sys, glob
import numpy as np


def _resolve_workdir():
    base = os.path.dirname(os.path.abspath(__file__))
    exp = os.path.join(base, 'experiments')
    if os.path.isdir(exp):
        return exp
    return '/root/heart-train'  # server layout fallback
WORKDIR = os.environ.get('OS_WORKDIR', _resolve_workdir())

CH_W = np.array([1.0, 3.0, 5.0])
WA_W = np.array([0.1, 0.2, 1.0])


def smurmur(y_true, y_pred):
    n = 3
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    ch_num = float((CH_W * np.diag(cm)).sum())
    ch_den = float((CH_W * cm.sum(1)).sum())
    return ch_num / ch_den if ch_den > 0 else float('nan'), cm


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


def tune(probs, y_val, mode):
    """Grid search dP in [-1.6, 1.6], dU in [-1.2, 1.2] on val s_murmur."""
    if mode == 'probavg':
        base = probs
        adjust = lambda dP, dU: base * np.exp(np.array([0.0, dU, dP]))[None, :]
    else:  # logitavg
        base = np.log(np.clip(probs, 1e-12, None))
        adjust = lambda dP, dU: base + np.array([0.0, dU, dP])[None, :]
    best = (-1.0, None, None)
    for dP in np.arange(-1.6, 1.61, 0.2):
        for dU in np.arange(-1.2, 1.21, 0.2):
            adj = adjust(dP, dU)
            s, _ = smurmur(y_val, adj.argmax(1))
            if s > best[0]:
                best = (s, dP, dU)
    return best


def main():
    args = sys.argv[1:]
    mode = 'probavg'
    if args and args[0].startswith('--mode'):
        mode = args[1]
        args = args[2:]
    tags = args
    probs_dir = os.path.join(WORKDIR, 'probs') if os.path.isdir(os.path.join(WORKDIR, 'probs')) else WORKDIR
    if not tags:
        tags = sorted(os.path.basename(p)[11:-4] for p in glob.glob(os.path.join(probs_dir, 'v4_probs_*.npz')))
    vf, vv, vy, tf, tv, ty = [], [], None, [], [], None
    for t in tags:
        p = os.path.join(probs_dir, f'v4_probs_{t}.npz')
        d = np.load(p)
        if vy is None:
            vy, ty = d['val_y'], d['test_y']
        assert (vy == d['val_y']).all() and (ty == d['test_y']).all()
        vf.append(d['val_fused']); vv.append(d['val_vote'])
        tf.append(d['test_fused']); tv.append(d['test_vote'])
        print(f'[load] {p}  (val fused s_murmur: '
              f'{smurmur(vy, d["val_fused"].argmax(1))[0]:.4f}, test: '
              f'{smurmur(ty, d["test_fused"].argmax(1))[0]:.4f})')
    print(f'\n===== ENSEMBLE mode={mode} tags={tags} =====')
    for pname, vp, tp in [('fused', np.mean(vf, 0), np.mean(tf, 0)),
                          ('vote', np.mean(vv, 0), np.mean(tv, 0))]:
        val_s, dP, dU = tune(vp, vy, mode)
        if mode == 'probavg':
            adj_t = tp * np.exp(np.array([0.0, dU, dP]))[None, :]
        else:
            adj_t = np.log(np.clip(tp, 1e-12, None)) + np.array([0.0, dU, dP])[None, :]
        show(f'ENSEMBLE {pname} ({len(tags)} models) tuned dP={dP:.2f} dU={dU:.2f} '
             f'(val s_murmur={val_s:.4f})', ty, adj_t.argmax(1))
    # per-model tuned (single-model threshold shift)
    print('\n----- per-model tuned thresholds -----')
    for i, t in enumerate(tags):
        d = np.load(os.path.join(probs_dir, f'v4_probs_{t}.npz'))
        val_s, dP, dU = tune(d['val_fused'], d['val_y'], mode)
        if mode == 'probavg':
            adj = d['test_fused'] * np.exp(np.array([0.0, dU, dP]))[None, :]
        else:
            adj = np.log(np.clip(d['test_fused'], 1e-12, None)) + np.array([0.0, dU, dP])[None, :]
        acc, recall, f1, wa, ch, cm = metrics(ty, adj.argmax(1))
        print(f'  {t:22s} dP={dP:+.2f} dU={dU:+.2f} (val {val_s:.4f}) -> test s_murmur={ch:.4f} '
              f'acc={acc:.4f} recall={np.round(recall,3).tolist()}')


if __name__ == '__main__':
    main()
