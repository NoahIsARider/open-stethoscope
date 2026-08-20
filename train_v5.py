#!/usr/bin/env python3
"""FusionNetV5: scaled-up encoder (4 conv blocks, 384 ch, ~2.3M params).
Same v4 training config; fixed split; report test s_murmur."""
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



class FusionNetV5(nn.Module):
    def __init__(self, n_class=3, embed=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_vec = nn.Parameter(torch.randn(embed) * 0.02)
        self.attn_bias = nn.Parameter(torch.zeros(1))
        self.fc1 = nn.Linear(embed, 256)
        self.fc2 = nn.Linear(256, n_class)
        self.aux = nn.Linear(256, 2)

    def encode(self, x):
        return self.avgpool(self.encoder(x)).flatten(1)

    def forward(self, x, mask):
        B, K = x.shape[0], x.shape[1]
        e = self.encode(x.view(B * K, 1, T.N_MELS, T.N_FRAMES)).view(B, K, -1)
        scores = torch.einsum('bkf,f->bk', e, self.attn_vec) + self.attn_bias
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)
        fused = (alpha.unsqueeze(-1) * e).sum(1)
        h = F.dropout(F.relu(self.fc1(fused)), 0.3, training=self.training)
        return self.fc2(h), self.aux(h), alpha

    def per_location_logits(self, e):
        h = F.relu(self.fc1(e))
        return self.fc2(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    tag = args.tag or f'v5_s{args.seed}'

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
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_w, dtype=torch.float32, device=device))
    aux_fn = nn.CrossEntropyLoss()
    patient_w = np.array([ce_w[labels[p]] for p in train_pids], dtype=float)
    patient_probs = patient_w / patient_w.sum()

    model = FusionNetV5().to(device)
    print(f'[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
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
            loss = loss_fn(logits, yb) + T.AUX_LAMBDA * aux_fn(aux_logits, (yb > 0).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(x); run_corr += (logits.argmax(1).cpu().numpy() == y).sum()
        sched.step()
        f_p, f_pr, v_p, v_pr, y_v = T.evaluate_probs(model, val_pids, labels, patient_locs, mel_cache, device)
        acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(y_v, f_pr)
        es_val = float(f1.mean())
        print(f'[ep{ep:02d}] loss={run_loss/N:.4f} train_acc={run_corr/N:.4f} '
              f'val_acc={acc:.4f} macroF1={f1.mean():.4f} chWA={ch_wa:.4f} recall={np.round(recall,3).tolist()} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if es_val > best_m:
            best_m, best_ep, bad = es_val, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[early stop] ep {ep}', flush=True)
                break
    model.load_state_dict(best_state)
    print(f'[best] ep={best_ep} val_macroF1={best_m:.4f} total={time.time()-t0:.0f}s', flush=True)
    f_p, f_pr, v_p, v_pr, y_t = T.evaluate_probs(model, test_pids, labels, patient_locs, mel_cache, device)
    acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(y_t, f_pr)
    print(f'[TEST] acc={acc:.4f} macroF1={f1.mean():.4f} s_murmur={ch_wa:.4f} recall={np.round(recall,3).tolist()}', flush=True)
    print(cm, flush=True)
    torch.save({'state_dict': best_state, 'epoch': best_ep, 'val_es': best_m,
                'arch': 'FusionNetV5', 'kind': f'fusion-{tag}', 'seed': args.seed,
                'classes': T.CLASS_NAMES}, os.path.join(WORKDIR, f'best_model_v4_{tag}.pt'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
