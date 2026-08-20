#!/usr/bin/env python3
"""Train lightweight classifier on frozen wav2vec2 features (patient-level).
Same fixed split as v4 (v3_split_seed42.json). Two heads:
  1) MLP on per-patient mean-pooled w2v features
  2) [optional] per-location attention fusion over w2v features (reuse FusionNet-style)
Report test s_murmur on the 142 held-out patients.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/root/open-stethoscope')
import train_v4 as T

WORKDIR = '/root/heart-train'
FEAT = np.load(os.path.join(WORKDIR, 'w2v_features.npz'), allow_pickle=True)
CH_W = np.array([1.0, 3.0, 5.0])


class MLP(nn.Module):
    def __init__(self, din=768, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 3))
    def forward(self, x):
        return self.net(x)


class AttnFusionW2V(nn.Module):
    """Per-location w2v features -> learned attention fusion -> MLP head."""
    def __init__(self, din=768, hidden=128):
        super().__init__()
        self.proj = nn.Linear(din, hidden)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x, mask):
        # x: B x 4 x din, mask: B x 4 bool
        h = torch.relu(self.proj(x))
        a = self.attn(h).squeeze(-1)
        a = a.masked_fill(~mask, float('-inf'))
        a = torch.softmax(a, dim=-1)
        z = (h * a.unsqueeze(-1)).sum(1)
        return self.head(z)


def load_patient_feats(patient_locs, labels):
    """per-patient: {loc: feat}; also patient mean."""
    pf = {}
    for pid, locs in patient_locs.items():
        d = {}
        for loc, paths in locs.items():
            feats = [FEAT[os.path.basename(p)] for p in paths if os.path.basename(p) in FEAT]
            if feats:
                d[loc] = np.mean(feats, axis=0)
        pf[pid] = d
    return pf


def main():
    device = torch.device('cuda')
    patient_locs, labels = T.collect_data()
    train_pids, val_pids, test_pids = T.load_or_make_split(patient_locs, labels)
    pf = load_patient_feats(patient_locs, labels)

    def to_batch(pids):
        X = np.zeros((len(pids), 4, 768), np.float32)
        M = np.zeros((len(pids), 4), bool)
        Y = np.array([labels[p] for p in pids])
        for i, p in enumerate(pids):
            for loc in pf[p]:
                k = T.LOC2IDX[loc]
                X[i, k] = pf[p][loc]
                M[i, k] = True
        return torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(Y).to(device)

    Xtr, Mtr, Ytr = to_batch(train_pids)
    Xva, Mva, Yva = to_batch(val_pids)
    Xte, Mte, Yte = to_batch(test_pids)
    print(f'[data] train {Xtr.shape[0]} val {Xva.shape[0]} test {Xte.shape[0]}', flush=True)

    def patient_mean(X, M):
        return (X * M.unsqueeze(-1)).sum(1) / M.sum(1).clamp(min=1).unsqueeze(-1)

    Pm_tr, Pm_va, Pm_te = patient_mean(Xtr, Mtr), patient_mean(Xva, Mva), patient_mean(Xte, Mte)

    counts = np.array([sum(1 for p in train_pids if labels[p] == c) for c in range(3)], dtype=float)
    N = len(train_pids)
    ce_w = (np.sqrt(N / (counts * 3)) / np.sqrt(N / (counts * 3)).mean()).astype(np.float32)
    print(f'[imb] ce_weights={np.round(ce_w,3).tolist()}', flush=True)

    results = {}
    for name, build, Xt, Mt, Yt, Xe, Me, Ye in [
        ('mlp', lambda: MLP().to(device), Pm_tr, None, Ytr, Pm_te, None, Yte),
        ('attn_fusion', lambda: AttnFusionW2V().to(device), Xtr, Mtr, Ytr, Xte, Mte, Yte),
    ]:
        torch.manual_seed(43)
        model = build()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_w, device=device))
        best_m, best_state = -1, None
        t0 = time.time()
        for ep in range(60):
            model.train()
            perm = torch.randperm(len(Yt))
            run_loss, run_corr = 0.0, 0
            for i in range(0, len(perm), 64):
                idx = perm[i:i+64]
                xb, mb, yb = Xt[idx], (Mt[idx] if Mt is not None else None), Yt[idx]
                opt.zero_grad()
                logits = model(xb) if mb is None else model(xb, mb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(xb)
                run_corr += (logits.argmax(1) == yb).sum().item()
            model.eval()
            with torch.no_grad():
                Xv = Pm_va if Mt is None else Xva
                va = model(Xv) if Mt is None else model(Xv, Mva)
                acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(Yva.cpu().numpy(), va.argmax(1).cpu().numpy())
            es_val = float(f1.mean())
            if es_val > best_m:
                best_m = es_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te = model(Xe) if Me is None else model(Xe, Me)
            acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(Ye.cpu().numpy(), te.argmax(1).cpu().numpy())
        print(f'\n[{name}] TEST acc={acc:.4f} macroF1={f1.mean():.4f} s_murmur={ch_wa:.4f} '
              f'recall={np.round(recall,3).tolist()}', flush=True)
        print(cm, flush=True)
        results[name] = {'acc': acc, 'macro_f1': float(f1.mean()), 's_murmur': ch_wa,
                         'recall': recall.tolist(), 'cm': cm.tolist(), 'val_es': best_m}
    print(json.dumps(results, indent=1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
