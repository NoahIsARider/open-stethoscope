#!/usr/bin/env python3
"""Focal-loss variant of train_v4 (fixed split, seed 43, 30ep).
Focal loss on fused logits: FL = -alpha * (1-p)^gamma * log(p); alpha = sqrt-inv-freq ce_w.
Aux murmur-present head stays CE. Everything else identical to v4 config."""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_v4 as T


def _resolve_workdir():
    base = os.path.dirname(os.path.abspath(__file__))
    exp = os.path.join(base, 'experiments')
    if os.path.isdir(exp):
        return exp
    return '/root/heart-train'  # server layout fallback
WORKDIR = os.environ.get('OS_WORKDIR', _resolve_workdir())



class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        alpha = self.alpha.to(logits.device)
        at = alpha[targets]
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        return -(at * (1 - pt).pow(self.gamma) * logp.gather(1, targets.unsqueeze(1)).squeeze(1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--gamma', type=float, default=2.0)
    args = ap.parse_args()

    T.set_seed(args.seed)
    device = torch.device('cuda')
    patient_locs, labels = T.collect_data()
    train_pids, val_pids, test_pids = T.load_or_make_split(patient_locs, labels)
    dist = lambda pids: [sum(1 for p in pids if labels[p] == c) for c in range(3)]
    print(f'[split] train={len(train_pids)} {dist(train_pids)} val={len(val_pids)} {dist(val_pids)} '
          f'test={len(test_pids)} {dist(test_pids)}', flush=True)
    all_paths = []
    for pid in list(train_pids) + list(val_pids) + list(test_pids):
        for paths in patient_locs[pid].values():
            all_paths.extend(paths)
    mel_cache = {}
    T.precompute_mels(all_paths, mel_cache)

    counts = np.array(dist(train_pids), dtype=float)
    N = len(train_pids)
    class_w = np.sqrt(N / (counts * 3))
    ce_w = class_w / class_w.mean()
    print(f'[imb] alpha={np.round(ce_w,3).tolist()} gamma={args.gamma}', flush=True)
    focal = FocalLoss(ce_w, args.gamma)
    aux_fn = nn.CrossEntropyLoss()
    patient_w = np.array([ce_w[labels[p]] for p in train_pids], dtype=float)
    patient_probs = patient_w / patient_w.sum()

    tag = f'focal_g{args.gamma}_s{args.seed}'
    model = T.FusionNet(n_class=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_m, best_ep, bad, best_state = -1.0, -1, 0, None
    t0 = time.time()
    rng = np.random.RandomState(args.seed)
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = rng.choice(N, size=N, p=patient_probs, replace=True)
        n_b = (N + 31) // 32
        run_loss, run_corr = 0.0, 0
        for b in range(n_b):
            x, mask, y = T.make_patient_batch(
                [train_pids[i] for i in idx[b*32:(b+1)*32]],
                labels, patient_locs, mel_cache, rng, train=True, aug=True)
            xb = torch.from_numpy(x).to(device); mb = torch.from_numpy(mask).to(device)
            yb = torch.from_numpy(y).to(device)
            opt.zero_grad()
            logits, aux_logits, _ = model(xb, mb)
            loss = focal(logits, yb) + T.AUX_LAMBDA * aux_fn(aux_logits, (yb > 0).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(x); run_corr += (logits.argmax(1).cpu().numpy() == y).sum()
        sched.step()
        f_p, f_pr, v_p, v_pr, y_v = T.evaluate_probs(model, val_pids, labels, patient_locs, mel_cache, device)
        acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(y_v, f_pr)
        es_val = float(f1.mean())
        print(f'[ep{ep:02d}] loss={run_loss/N:.4f} train_acc={run_corr/N:.4f} '
              f'val_acc={acc:.4f} macroF1={f1.mean():.4f} chWA={ch_wa:.4f} recall={np.round(recall,3).tolist()}', flush=True)
        if es_val > best_m:
            best_m, best_ep, bad = es_val, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    print(f'[best] ep={best_ep} val_macroF1={best_m:.4f} total={time.time()-t0:.0f}s', flush=True)
    f_p, f_pr, v_p, v_pr, y_t = T.evaluate_probs(model, test_pids, labels, patient_locs, mel_cache, device)
    acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(y_t, f_pr)
    print(f'[TEST] acc={acc:.4f} macroF1={f1.mean():.4f} s_murmur={ch_wa:.4f} recall={np.round(recall,3).tolist()}', flush=True)
    print(cm, flush=True)
    torch.save({'state_dict': best_state, 'epoch': best_ep, 'val_es': best_m,
                'arch': 'FusionNet', 'kind': f'fusion-{tag}', 'seed': args.seed,
                'classes': T.CLASS_NAMES}, os.path.join(WORKDIR, f'best_model_v4_{tag}.pt'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
