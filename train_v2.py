#!/usr/bin/env python3
"""
train_v2.py -- Heart-sound murmur classification v2
(multi-location fusion + class-imbalance handling + augmentation).

Improvements vs MVP (train_mvp.py, val acc 0.8278 but Unknown recall=0,
Present recall=0.408):

  1. MULTI-LOCATION FUSION (core novelty)
     - One patient = one training sample. All available auscultation positions
       (AV/PV/TV/MV) of that patient are encoded by a SHARED per-location 2D
       CNN, then fused with a MASKED LEARNED ATTENTION (softmax over the
       positions that actually exist for this patient).
     - Rationale: a clinician listens to all four positions before judging; most
       CirCor-2022 solutions classify single recordings, and cross-position
       agreement is exactly the signal a human uses. Attention lets the model
       learn which positions matter (e.g. the murmur is most audible at TV/MV
       for many patients) and the mask cleanly handles missing positions
       (587/941 patients have all 4, 128 have 3, 160 have 2, 66 have 1).
     - Repeat recordings of the same position (e.g. 49748_AV_1/_2) are treated
       as interchangeable candidates: one is sampled at random per epoch during
       training (free augmentation), the longest one is used at eval.
  2. CLASS IMBALANCE (Absent 695 / Present 179 / Unknown 68 patients)
     a. per-epoch weighted patient sampling with sqrt-inverse-frequency
        probabilities -> minority classes are seen every epoch;
     b. sqrt-inverse-frequency class weights in the CE loss (softened vs plain
        inverse frequency so the tiny Unknown class is not overfit);
     c. auxiliary binary head "murmur present?" (Absent vs Present|Unknown) on
        the fused representation: the binary task is well populated (695 vs
        247) and forces the shared representation to keep murmur-presence
        structure even where the 3-class labels are sparse (lambda=0.3).
  3. LIGHTWEIGHT AUGMENTATION (mel domain; equivalent to wav-domain ops on a
     log-mel front end): random 8 s window crop, +/-10% time stretch (time-axis
     linear interpolation), SpecAugment-lite frequency/time masking, small
     Gaussian noise (sigma=0.05 dB).
  4. FULL training data, patient-stratified 80/20, seed 42, AdamW lr=6e-4 with
     cosine decay, gradient clipping (norm 1.0), 20 epochs, early stopping
     patience 5 (selection metric: val macro-F1 -- robust against degenerate
     all-Present solutions that the challenge weighted-acc alone would reward).
  5. METRICS: 3-class accuracy, per-class recall, challenge-style weighted
     accuracy (w_Unknown=0.2, w_Absent=0.1, w_Present=1.0), plus a
     single-location vs fusion ablation:
       - Fusion model: trained on patient-level (multi-position) samples.
       - Single-location baseline: same architecture, trained on wav-level
         samples (one position per sample); patient-level prediction = majority
         vote of its positions. Both models see the SAME number of optimizer
         steps per epoch (753 samples) and the same LR schedule, so the
         ablation isolates 'multi-position fusion vs unit-position' cleanly.
"""
import os, re, sys, time, argparse, random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
SR = 4000
N_FRAMES = 126                 # mel frames of an 8 s window (hop 256, center=True)
N_MELS = 40
N_FFT = 512
HOP = 256
FMIN, FMAX = 25, 2000

DATA_CSV = '/root/heart-data/training_data.csv'
DATA_DIR = '/root/heart-data/training_data'
WORKDIR = '/root/heart-train'
BEST_MODEL = os.path.join(WORKDIR, 'best_model_v2.pt')
BEST_SINGLE = os.path.join(WORKDIR, 'best_model_v2_single.pt')

LOCS = ['AV', 'PV', 'TV', 'MV']
LOC2IDX = {l: i for i, l in enumerate(LOCS)}
MURMUR_MAP = {'Absent': 0, 'Unknown': 1, 'Present': 2}
CLASS_NAMES = ['Absent', 'Unknown', 'Present']
WA_W = np.array([0.1, 0.2, 1.0])   # weighted-acc: Unknown 0.2, Absent 0.1, Present 1.0
AUX_LAMBDA = 0.3


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def collect_data():
    df = pd.read_csv(DATA_CSV)
    labels = {}
    for pid, mur in zip(df['Patient ID'].astype(str).str.strip(), df['Murmur']):
        if mur in MURMUR_MAP:
            labels[int(pid)] = MURMUR_MAP[mur]
    patient_locs = {}   # pid -> {loc: [path, ...]}
    n_wav = 0
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.endswith('.wav'):
                continue
            m = re.match(r'(\d+)_(AV|PV|TV|MV)(?:_\d+)?\.wav', f)
            if not m:
                continue
            pid, loc = int(m.group(1)), m.group(2)
            if pid not in labels:
                continue
            patient_locs.setdefault(pid, {}).setdefault(loc, []).append(os.path.join(root, f))
            n_wav += 1
    patient_locs = {p: v for p, v in patient_locs.items() if v}
    dist = {c: sum(1 for p in patient_locs if labels[p] == i) for i, c in enumerate(CLASS_NAMES)}
    print(f'[data] patients={len(patient_locs)} wavs={n_wav} labels={dist}', flush=True)
    return patient_locs, labels


def full_mel(path):
    x, sr = sf.read(path, dtype='float32')
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    m = librosa.power_to_db(m, ref=np.max)   # normalized per full recording
    return m.astype(np.float32)


def precompute_mels(paths, mel_cache):
    t0 = time.time()
    for i, p in enumerate(paths):
        if p not in mel_cache:
            mel_cache[p] = full_mel(p)
        if (i + 1) % 400 == 0:
            print(f'[feat] {i+1}/{len(paths)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[feat] cached {len(mel_cache)} mels in {time.time()-t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------
def augment_mel(m, rng):
    """m: (40, T) full-recording log-mel. Returns (40, 126) augmented patch."""
    s = rng.uniform(0.9, 1.1)                     # 1) time stretch +/-10%
    if abs(s - 1.0) > 0.02:
        T = m.shape[1]
        new_T = max(40, int(round(T * s)))
        idx = np.linspace(0, T - 1, new_T)
        m = np.stack([np.interp(idx, np.arange(T), m[i]) for i in range(m.shape[0])], axis=0)
    T = m.shape[1]                                # 2) random 8 s window
    if T >= N_FRAMES:
        st = rng.randint(0, T - N_FRAMES + 1)
        m = m[:, st:st + N_FRAMES]
    else:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - T)))
    if rng.random() < 0.5:                        # 3) freq mask
        f0 = rng.randint(0, N_MELS - 2)
        f_len = rng.randint(2, min(5, N_MELS - f0) + 1)
        m[f0:f0 + f_len, :] -= rng.uniform(1.5, 3.0)
    if rng.random() < 0.5:                        # 4) time mask
        t0 = rng.randint(0, N_FRAMES - 5)
        t_len = rng.randint(5, min(16, N_FRAMES - t0) + 1)
        m[:, t0:t0 + t_len] -= rng.uniform(1.5, 3.0)
    m += rng.normal(0.0, 0.05, size=m.shape).astype(np.float32)   # 5) noise
    return m.astype(np.float32)


def eval_mel(m):
    """Deterministic first-8s window for evaluation."""
    if m.shape[1] >= N_FRAMES:
        return m[:, :N_FRAMES].astype(np.float32)
    return np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1]))).astype(np.float32)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class FusionNet(nn.Module):
    """Shared per-location CNN encoder + masked learned-attention fusion + heads.

    Same architecture is used for the single-location baseline (K=1 samples),
    so the ablation is exactly 'multi-position fusion vs unit-position'.
    """
    def __init__(self, n_class=3, embed=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_vec = nn.Parameter(torch.randn(embed) * 0.02)   # learned location-attention vector
        self.attn_bias = nn.Parameter(torch.zeros(1))
        self.fc1 = nn.Linear(embed, 128)
        self.fc2 = nn.Linear(128, n_class)
        self.aux = nn.Linear(128, 2)          # binary murmur-present head

    def encode(self, x):
        e = self.encoder(x)
        return self.avgpool(e).flatten(1)

    def forward(self, x, mask):
        B, K = x.shape[0], x.shape[1]
        e = self.encode(x.view(B * K, 1, N_MELS, N_FRAMES)).view(B, K, -1)
        scores = torch.einsum('bkf,f->bk', e, self.attn_vec) + self.attn_bias
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)
        fused = (alpha.unsqueeze(-1) * e).sum(1)
        h = F.dropout(F.relu(self.fc1(fused)), 0.3, training=self.training)
        return self.fc2(h), self.aux(h), alpha

    def per_location_logits(self, e):
        h = F.relu(self.fc1(e))
        return self.fc2(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# batch building / metrics
# ----------------------------------------------------------------------------
def make_patient_batch(pid_list, labels, patient_locs, mel_cache, rng, train=True):
    B = len(pid_list)
    x = np.zeros((B, 4, 1, N_MELS, N_FRAMES), np.float32)
    mask = np.zeros((B, 4), dtype=bool)
    y = np.zeros(B, np.int64)
    for bi, pid in enumerate(pid_list):
        y[bi] = labels[pid]
        for loc, paths in patient_locs[pid].items():
            k = LOC2IDX[loc]
            if train:
                path = paths[rng.randint(len(paths))]
                x[bi, k, 0] = augment_mel(mel_cache[path], rng)
            else:
                path = max(paths, key=lambda p: mel_cache[p].shape[1])
                x[bi, k, 0] = eval_mel(mel_cache[path])
            mask[bi, k] = True
    return x, mask, y


def make_wav_batch(wav_list, labels, mel_cache, rng, train=True):
    B = len(wav_list)
    x = np.zeros((B, 4, 1, N_MELS, N_FRAMES), np.float32)
    mask = np.zeros((B, 4), dtype=bool)
    y = np.zeros(B, np.int64)
    for bi, (pid, loc, path) in enumerate(wav_list):
        y[bi] = labels[pid]
        k = LOC2IDX[loc]
        if train:
            x[bi, k, 0] = augment_mel(mel_cache[path], rng)
        else:
            x[bi, k, 0] = eval_mel(mel_cache[path])
        mask[bi, k] = True
    return x, mask, y


def metrics(y_true, y_pred, n=3):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = (y_true == y_pred).mean()
    recall = np.array([cm[i, i] / max(1, cm[i].sum()) for i in range(n)])
    prec = np.array([cm[i, i] / max(1, cm[:, i].sum()) for i in range(n)])
    f1 = 2 * prec * recall / (prec + recall + 1e-9)
    wa = float((WA_W * recall).sum() / WA_W.sum())
    return acc, recall, f1, wa, cm


def evaluate(model, pid_list, labels, patient_locs, mel_cache, device, batch=64):
    """Patient-level eval: returns (fusion preds, vote preds, labels, loc_y, loc_p)."""
    model.eval()
    all_fused, all_vote, all_y = [], [], []
    loc_y, loc_p = [], []
    with torch.no_grad():
        for i in range(0, len(pid_list), batch):
            pl = pid_list[i:i + batch]
            x, mask, y = make_patient_batch(pl, labels, patient_locs, mel_cache, None, train=False)
            xb = torch.from_numpy(x).to(device)
            maskb = torch.from_numpy(mask).to(device)
            logits, _, _ = model(xb, maskb)
            all_fused.extend(logits.argmax(1).cpu().numpy())
            B, K = x.shape[0], x.shape[1]
            e = model.encode(xb.view(B * K, 1, N_MELS, N_FRAMES))
            plog = torch.softmax(model.per_location_logits(e).view(B, K, 3), dim=-1).cpu().numpy()
            all_vote.extend(plog.mean(axis=1).argmax(axis=1))
            all_y.extend(y)
            for bi in range(B):
                for k in range(K):
                    if mask[bi, k]:
                        loc_y.append(int(y[bi]))
                        loc_p.append(int(plog[bi, k].argmax()))
    return (np.array(all_fused), np.array(all_vote), np.array(all_y),
            np.array(loc_y), np.array(loc_p))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20, help='epochs (both models, equal budget)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--grad-clip', type=float, default=1.0, help='max grad norm (0 = off)')
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device} torch={torch.__version__}', flush=True)
    if torch.cuda.is_available():
        print(f'[env] gpu={torch.cuda.get_device_name(0)}', flush=True)

    patient_locs, labels = collect_data()

    # ---- patient-stratified 80/20 split ----
    rng = np.random.RandomState(SEED)
    all_pids = np.array(sorted(patient_locs.keys()))
    groups = {c: [] for c in range(3)}
    for p in all_pids:
        groups[labels[p]].append(p)
    val_set = set()
    for c in range(3):
        lst = groups[c][:]
        rng.shuffle(lst)
        n = max(1, int(round(0.2 * len(lst))))
        val_set.update(lst[:n])
    train_pids = [p for p in all_pids if p not in val_set]
    val_pids = [p for p in all_pids if p in val_set]
    print(f'[split] patients train={len(train_pids)} val={len(val_pids)} '
          f'val-dist={{Absent:{sum(1 for p in val_pids if labels[p]==0)}, '
          f'Unknown:{sum(1 for p in val_pids if labels[p]==1)}, '
          f'Present:{sum(1 for p in val_pids if labels[p]==2)}}}', flush=True)

    # ---- precompute mels ----
    all_paths = []
    for pid in list(train_pids) + val_pids:
        for paths in patient_locs[pid].values():
            all_paths.extend(paths)
    mel_cache = {}
    precompute_mels(all_paths, mel_cache)

    # ---- class weights & sampling probs ----
    counts = np.array([sum(1 for p in train_pids if labels[p] == c) for c in range(3)], dtype=float)
    class_w = np.sqrt(len(train_pids) / (counts * 3))
    class_w = class_w / class_w.mean()
    print(f'[imb] train counts={counts.astype(int).tolist()} class_weights='
          f'{np.round(class_w, 3).tolist()}', flush=True)
    patient_w = np.array([class_w[labels[p]] for p in train_pids], dtype=float)
    patient_probs = patient_w / patient_w.sum()
    wav_list_tr = [(pid, loc, path) for pid in train_pids
                   for loc, paths in patient_locs[pid].items() for path in paths]
    wav_w = np.array([class_w[labels[pid]] for pid, _, _ in wav_list_tr], dtype=float)
    wav_probs = wav_w / wav_w.sum()

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    aux_fn = nn.CrossEntropyLoss()
    patience = 5
    n_per_epoch = len(train_pids)   # equal optimizer budget for both models

    # ================= FUSION MODEL =================
    print('\n================ TRAIN FUSION MODEL (patient-level) ================', flush=True)
    model = FusionNet(n_class=3).to(device)
    n_params = count_params(model)
    print(f'[model] params={n_params:,}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    torch.cuda.reset_peak_memory_stats()
    best_f1, best_ep, bad_epochs, best_state = -1.0, -1, 0, None
    t0 = time.time(); ep_times = []
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = rng.choice(n_per_epoch, size=n_per_epoch, p=patient_probs, replace=True)
        n_batches = (n_per_epoch + args.batch - 1) // args.batch
        run_loss, run_corr = 0.0, 0
        t_ep = time.time()
        for b in range(n_batches):
            x, mask, y = make_patient_batch([train_pids[i] for i in idx[b * args.batch:(b + 1) * args.batch]],
                                            labels, patient_locs, mel_cache, rng, train=True)
            xb = torch.from_numpy(x).to(device); mb = torch.from_numpy(mask).to(device)
            yb = torch.from_numpy(y).to(device)
            opt.zero_grad()
            logits, aux_logits, _ = model(xb, mb)
            loss = loss_fn(logits, yb) + AUX_LAMBDA * aux_fn(aux_logits, (yb > 0).long())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run_loss += loss.item() * len(x); run_corr += (logits.argmax(1).cpu().numpy() == y).sum()
        sched.step()
        f_pred, v_pred, y_v, loc_y, loc_p = evaluate(model, val_pids, labels, patient_locs, mel_cache, device)
        acc, recall, f1, wa, cm = metrics(y_v, f_pred)
        ep_times.append(time.time() - t_ep)
        print(f'[fusion ep {ep:02d}] loss={run_loss/n_per_epoch:.4f} train_acc={run_corr/n_per_epoch:.4f} '
              f'val_acc={acc:.4f} macroF1={f1.mean():.4f} wacc={wa:.4f} '
              f'recall={np.round(recall, 3).tolist()} ({ep_times[-1]:.1f}s)', flush=True)
        if f1.mean() > best_f1:
            best_f1, best_ep, bad_epochs = f1.mean(), ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'[fusion] early stop at epoch {ep}', flush=True)
                break
    torch.save({'state_dict': best_state, 'epoch': best_ep, 'val_macro_f1': float(best_f1),
                'n_params': n_params, 'arch': 'FusionNet', 'kind': 'fusion',
                'classes': CLASS_NAMES}, BEST_MODEL)
    mem_fusion = torch.cuda.max_memory_allocated() / 1e6
    t_fusion = time.time() - t0
    print(f'[fusion] best epoch={best_ep} macroF1={best_f1:.4f} | peak mem={mem_fusion:.0f}MB '
          f'| {np.mean(ep_times):.1f}s/epoch | total {t_fusion:.0f}s', flush=True)

    # ================= SINGLE-LOCATION BASELINE =================
    print('\n======== TRAIN SINGLE-LOCATION BASELINE (wav-level, equal budget) ========', flush=True)
    model_s = FusionNet(n_class=3).to(device)
    print(f'[model] params={count_params(model_s):,} (same arch, one position per sample)', flush=True)
    opt_s = torch.optim.AdamW(model_s.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=args.epochs)
    torch.cuda.reset_peak_memory_stats()
    best_f1s, best_eps, bad_epochs_s, best_state_s = -1.0, -1, 0, None
    t0s = time.time(); ep_times_s = []
    for ep in range(1, args.epochs + 1):
        model_s.train()
        idx = rng.choice(len(wav_list_tr), size=n_per_epoch, p=wav_probs, replace=True)
        n_batches = (n_per_epoch + args.batch - 1) // args.batch
        run_loss, run_corr = 0.0, 0
        t_ep = time.time()
        for b in range(n_batches):
            x, mask, y = make_wav_batch([wav_list_tr[i] for i in idx[b * args.batch:(b + 1) * args.batch]],
                                        labels, mel_cache, rng, train=True)
            xb = torch.from_numpy(x).to(device); mb = torch.from_numpy(mask).to(device)
            yb = torch.from_numpy(y).to(device)
            opt_s.zero_grad()
            logits, aux_logits, _ = model_s(xb, mb)
            loss = loss_fn(logits, yb) + AUX_LAMBDA * aux_fn(aux_logits, (yb > 0).long())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), args.grad_clip)
            opt_s.step()
            run_loss += loss.item() * len(x); run_corr += (logits.argmax(1).cpu().numpy() == y).sum()
        sched_s.step()
        f_pred, v_pred, y_v, loc_y, loc_p = evaluate(model_s, val_pids, labels, patient_locs, mel_cache, device)
        acc_v, recall_v, f1_v, wa_v, cm_v = metrics(y_v, v_pred)   # patient-level vote
        acc_l, recall_l, f1_l, wa_l, cm_l = metrics(loc_y, loc_p)  # wav-level
        ep_times_s.append(time.time() - t_ep)
        print(f'[single ep {ep:02d}] loss={run_loss/n_per_epoch:.4f} train_acc={run_corr/n_per_epoch:.4f} '
              f'vote_acc={acc_v:.4f} macroF1={f1_v.mean():.4f} wacc={wa_v:.4f} '
              f'wav_acc={acc_l:.4f} ({ep_times_s[-1]:.1f}s)', flush=True)
        if f1_v.mean() > best_f1s:
            best_f1s, best_eps, bad_epochs_s = f1_v.mean(), ep, 0
            best_state_s = {k: v.cpu().clone() for k, v in model_s.state_dict().items()}
        else:
            bad_epochs_s += 1
            if bad_epochs_s >= patience:
                print(f'[single] early stop at epoch {ep}', flush=True)
                break
    torch.save({'state_dict': best_state_s, 'epoch': best_eps, 'val_macro_f1': float(best_f1s),
                'n_params': n_params, 'arch': 'FusionNet', 'kind': 'single',
                'classes': CLASS_NAMES}, BEST_SINGLE)
    mem_single = torch.cuda.max_memory_allocated() / 1e6
    t_single = time.time() - t0s
    print(f'[single] best epoch={best_eps} macroF1={best_f1s:.4f} | peak mem={mem_single:.0f}MB '
          f'| {np.mean(ep_times_s):.1f}s/epoch | total {t_single:.0f}s', flush=True)

    # ================= FINAL REPORT (both models, their best EMA checkpoints) =================
    model.load_state_dict(best_state)
    model_s.load_state_dict(best_state_s)
    f_f, v_f, y_f, lf_y, lf_p = evaluate(model, val_pids, labels, patient_locs, mel_cache, device)
    f_s, v_s, y_s, ls_y, ls_p = evaluate(model_s, val_pids, labels, patient_locs, mel_cache, device)
    acc_f, recall_f, f1_f, wa_f, cm_f = metrics(y_f, f_f)       # fusion: attention-fused
    acc_vf, recall_vf, f1_vf, wa_vf, cm_vf = metrics(y_f, v_f)  # fusion model: per-loc vote
    acc_vs, recall_vs, f1_vs, wa_vs, cm_vs = metrics(y_s, v_s)  # single model: patient vote
    acc_ls, recall_ls, f1_ls, wa_ls, cm_ls = metrics(ls_y, ls_p)  # single model: wav-level

    print('\n================ FINAL REPORT (val, patient-level) ================', flush=True)
    print('--- FUSION model (attention over available positions) ---', flush=True)
    print(f'accuracy       : {acc_f:.4f}', flush=True)
    print(f'macro-F1       : {f1_f.mean():.4f}', flush=True)
    print(f'weighted acc   : {wa_f:.4f}  (w: Absent 0.1, Unknown 0.2, Present 1.0)', flush=True)
    print('confusion (rows=true [Absent,Unknown,Present]):', flush=True)
    print(np.array2string(cm_f), flush=True)
    for i, c in enumerate(CLASS_NAMES):
        print(f'  recall {c:7s}: {cm_f[i,i]}/{cm_f[i].sum()} = {recall_f[i]:.3f}', flush=True)
    print(f'--- FUSION model, majority vote of per-position preds (ablation) ---', flush=True)
    print(f'accuracy={acc_vf:.4f} macroF1={f1_vf.mean():.4f} weighted_acc={wa_vf:.4f}', flush=True)
    print('--- SINGLE-LOCATION baseline (trained on one position per sample) ---', flush=True)
    print(f'wav-level acc : {acc_ls:.4f}  (all positions pooled, MVP-style)', flush=True)
    print(f'patient vote  : acc={acc_vs:.4f} macroF1={f1_vs.mean():.4f} weighted_acc={wa_vs:.4f}', flush=True)
    print(f'  recalls     : {np.round(recall_vs, 3).tolist()}', flush=True)
    print(f'--- single vs fusion (patient-level) ---', flush=True)
    print(f'  acc    : {acc_f:.4f} (fusion) vs {acc_vs:.4f} (single-vote)', flush=True)
    print(f'  macroF1: {f1_f.mean():.4f} (fusion) vs {f1_vs.mean():.4f} (single-vote)', flush=True)
    print(f'  wacc   : {wa_f:.4f} (fusion) vs {wa_vs:.4f} (single-vote)', flush=True)

    # attention analysis (presence-mask based)
    model.eval()
    alphas, masks = [], []
    with torch.no_grad():
        for i in range(0, len(val_pids), 64):
            x, mask, y = make_patient_batch(val_pids[i:i + 64], labels, patient_locs, mel_cache, None, train=False)
            _, _, a = model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
            alphas.append(a.cpu().numpy()); masks.append(mask)
    alphas = np.concatenate(alphas); masks = np.concatenate(masks)
    print('mean attention weight per position (val, over patients where position exists):', flush=True)
    for k, loc in enumerate(LOCS):
        vals = alphas[masks[:, k], k]
        print(f'  {loc}: {vals.mean():.3f} (n={len(vals)})', flush=True)
    print('  patients by #positions: ' + str(np.bincount(masks.sum(1), minlength=5)[1:5].tolist()), flush=True)

    print(f'params          : {n_params:,}', flush=True)
    print(f'GPU peak memory : fusion {mem_fusion:.0f} MB | single {mem_single:.0f} MB', flush=True)
    print(f'epoch time      : fusion {np.mean(ep_times):.1f}s | single {np.mean(ep_times_s):.1f}s', flush=True)
    print(f'total time      : fusion {t_fusion:.0f}s | single {t_single:.0f}s', flush=True)
    print(f'saved           : {BEST_MODEL} | {BEST_SINGLE}', flush=True)
    print('==============================================================', flush=True)

    # ================= VERIFICATION: reload from disk and re-evaluate =================
    print('\n================ VERIFICATION (reload saved .pt from disk) ================', flush=True)
    m2 = FusionNet(n_class=3).to(device)
    ck = torch.load(BEST_MODEL, map_location=device, weights_only=False)
    m2.load_state_dict(ck['state_dict'])
    f_f, v_f, y_f, lf_y, lf_p = evaluate(m2, val_pids, labels, patient_locs, mel_cache, device)
    acc, recall, f1, wa, cm = metrics(y_f, f_f)
    print(f'{BEST_MODEL}: saved from epoch {ck["epoch"]}, val_macro_f1={ck["val_macro_f1"]:.4f}', flush=True)
    print(f'  reloaded eval: acc={acc:.4f} macroF1={f1.mean():.4f} wacc={wa:.4f} '
          f'recall={np.round(recall, 3).tolist()}', flush=True)
    m2s = FusionNet(n_class=3).to(device)
    ck_s = torch.load(BEST_SINGLE, map_location=device, weights_only=False)
    m2s.load_state_dict(ck_s['state_dict'])
    f_s, v_s, y_s, ls_y, ls_p = evaluate(m2s, val_pids, labels, patient_locs, mel_cache, device)
    acc_v, recall_v, f1_v, wa_v, cm_v = metrics(y_s, v_s)
    acc_l, recall_l, f1_l, wa_l, cm_l = metrics(ls_y, ls_p)
    print(f'{BEST_SINGLE}: saved from epoch {ck_s["epoch"]}, val_macro_f1={ck_s["val_macro_f1"]:.4f}', flush=True)
    print(f'  reloaded eval: wav_acc={acc_l:.4f} vote_acc={acc_v:.4f} macroF1={f1_v.mean():.4f} '
          f'wacc={wa_v:.4f} recall={np.round(recall_v, 3).tolist()}', flush=True)
    print('======================================================================', flush=True)


if __name__ == '__main__':
    main()
