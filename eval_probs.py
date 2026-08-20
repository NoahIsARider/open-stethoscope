#!/usr/bin/env python3
"""
eval_probs.py -- for saved v4 model checkpoints, re-evaluate VAL + TEST and
dump patient-level fused/vote softmax probs to npz for threshold tuning and
ensembling.

Usage: python3 eval_probs.py tag1.pt tag2.pt ...
Each tag maps to best_model_v4_{tag}.pt ; output v4_probs_{tag}.npz
with keys: val_fused, val_vote, val_y, test_fused, test_vote, test_y
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_v4 as T


def _resolve_workdir():
    base = os.path.dirname(os.path.abspath(__file__))
    exp = os.path.join(base, 'experiments')
    if os.path.isdir(exp):
        return exp
    return '/root/heart-train'  # server layout fallback
WORKDIR = os.environ.get('OS_WORKDIR', _resolve_workdir())

MEL_CACHE_DIR = os.path.join(WORKDIR, 'mel_cache_v3')


def main():
    tags = sys.argv[1:]
    if not tags:
        print('usage: eval_probs.py tag [tag...]')
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device}', flush=True)
    patient_locs, labels = T.collect_data()
    train_pids, val_pids, test_pids = T.load_or_make_split(patient_locs, labels)
    all_paths = []
    for pid in list(train_pids) + list(val_pids) + list(test_pids):
        for paths in patient_locs[pid].values():
            all_paths.extend(paths)
    mel_cache = {}
    T.precompute_mels(all_paths, mel_cache)
    for tag in tags:
        pt_path = os.path.join(WORKDIR, 'models', f'best_model_v4_{tag}.pt')
        if not os.path.exists(pt_path):
            pt_path = os.path.join(WORKDIR, f'best_model_v4_{tag}.pt')
        ckpt = torch.load(pt_path, map_location='cpu')
        model = T.FusionNet(n_class=3).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        vf, vfp, vv, vvp, vy = T.evaluate_probs(model, val_pids, labels, patient_locs, mel_cache, device)
        tf, tfp, tv, tvp, ty = T.evaluate_probs(model, test_pids, labels, patient_locs, mel_cache, device)
        probs_dir = os.path.join(WORKDIR, 'probs')
        out = os.path.join(probs_dir if os.path.isdir(probs_dir) else WORKDIR, f'v4_probs_{tag}.npz')
        np.savez(out, val_fused=vf, val_vote=vv, val_y=vy,
                 test_fused=tf, test_vote=tv, test_y=ty)
        acc, recall, f1, wa, ch, cm, spec = T.metrics(ty, tfp)
        print(f'[{tag}] test s_murmur={ch:.4f} acc={acc:.4f} recall={np.round(recall,3).tolist()} '
              f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
