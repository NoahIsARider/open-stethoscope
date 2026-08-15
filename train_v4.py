#!/usr/bin/env python3
"""
train_v4.py -- v4 iteration on top of train_v3.py (FusionNet, patient-level).

Changes vs v3:
1. EARLY STOPPING on official challenge metric val s_murmur (CH_W 1/3/5),
   instead of val macro-F1  (--es-metric smurmur|macro_f1).
2. PRESENT BOOST: multiply the Present class weight by --present-boost
   (applied to BOTH class-balance sampling probs and CE loss weight,
   renormalized to mean 1.0). sqrt-inverse-freq baseline = boost 1.0.
   Optional --sampling linear (w ~ N/count) also implemented.
3. MULTI-SEED: --seed controls all training RNG (split is ALWAYS the fixed
   v3_split_seed42.json, so different seeds = different training randomness,
   identical data split).
4. Saves per-run test softmax probs to npz for ensemble evaluation
   (patient-level fused probs + per-location vote probs + y_true).
5. Fusion model only (no single-location baseline, no ablations).

Model: FusionNet with masked learned-attention fusion over 4 locations,
aux murmur-present head, sqrt-inverse-freq class weights, class-balance
patient sampling, spec-augment + time-stretch + gaussian noise.
"""
import os, re, sys, time, argparse, json, hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 4000
N_FRAMES = 126
N_MELS = 40
N_FFT = 512
HOP = 256
FMIN, FMAX = 25, 2000

DATA_CSV = '/root/heart-data/training_data.csv'
DATA_DIR = '/root/heart-data/training_data'
WORKDIR = '/root/heart-train'
MEL_CACHE_DIR = os.path.join(WORKDIR, 'mel_cache_v3')
SPLIT_JSON = os.path.join(WORKDIR, 'v3_split_seed42.json')
RESULT_JSON = os.path.join(WORKDIR, 'exp_results_v4.json')

LOCS = ['AV', 'PV', 'TV', 'MV']
LOC2IDX = {l: i for i, l in enumerate(LOCS)}
MURMUR_MAP = {'Absent': 0, 'Unknown': 1, 'Present': 2}
CLASS_NAMES = ['Absent', 'Unknown', 'Present']
WA_W = np.array([0.1, 0.2, 1.0])            # legacy weighted-acc proxy
CH_W = np.array([1.0, 3.0, 5.0])            # OFFICIAL challenge weights
AUX_LAMBDA = 0.3


def set_seed(s):
    random_seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def random_seed(s):
    import random
    random.seed(s)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def collect_data():
    df = pd.read_csv(DATA_CSV)
    labels = {}
    for pid, mur in zip(df['Patient ID'].astype(str).str.strip(), df['Murmur']):
        if mur in MURMUR_MAP:
            labels[int(pid)] = MURMUR_MAP[mur]
    patient_locs = {}
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
    m = librosa.power_to_db(m, ref=np.max)
    return m.astype(np.float32)


def _cache_path(path):
    return os.path.join(MEL_CACHE_DIR, hashlib.sha1(path.encode()).hexdigest()[:20] + '.npy')


def precompute_mels(paths, mel_cache):
    os.makedirs(MEL_CACHE_DIR, exist_ok=True)
    t0 = time.time(); n_computed = 0
    for i, p in enumerate(paths):
        if p in mel_cache:
            continue
        cp = _cache_path(p)
        if os.path.exists(cp):
            mel_cache[p] = np.load(cp)
        else:
            mel_cache[p] = full_mel(p)
            np.save(cp, mel_cache[p])
            n_computed += 1
        if (i + 1) % 400 == 0:
            print(f'[feat] {i+1}/{len(paths)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'[feat] cache ready: {len(mel_cache)} mels ({n_computed} computed, '
          f'{time.time()-t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------
def augment_mel(m, rng):
    s = rng.uniform(0.9, 1.1)
    if abs(s - 1.0) > 0.02:
        T = m.shape[1]
        new_T = max(40, int(round(T * s)))
        idx = np.linspace(0, T - 1, new_T)
        m = np.stack([np.interp(idx, np.arange(T), m[i]) for i in range(m.shape[0])], axis=0)
    T = m.shape[1]
    if T >= N_FRAMES:
        st = rng.randint(0, T - N_FRAMES + 1)
        m = m[:, st:st + N_FRAMES]
    else:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - T)))
    if rng.random() < 0.5:
        f0 = rng.randint(0, N_MELS - 2)
        f_len = rng.randint(2, min(5, N_MELS - f0) + 1)
        m[f0:f0 + f_len, :] -= rng.uniform(1.5, 3.0)
    if rng.random() < 0.5:
        t0 = rng.randint(0, N_FRAMES - 5)
        t_len = rng.randint(5, min(16, N_FRAMES - t0) + 1)
        m[:, t0:t0 + t_len] -= rng.uniform(1.5, 3.0)
    m += rng.normal(0.0, 0.05, size=m.shape).astype(np.float32)
    return m.astype(np.float32)


def eval_mel(m):
    if m.shape[1] >= N_FRAMES:
        return m[:, :N_FRAMES].astype(np.float32)
    return np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1]))).astype(np.float32)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class FusionNet(nn.Module):
    def __init__(self, n_class=3, embed=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_vec = nn.Parameter(torch.randn(embed) * 0.02)
        self.attn_bias = nn.Parameter(torch.zeros(1))
        self.fc1 = nn.Linear(embed, 128)
        self.fc2 = nn.Linear(128, n_class)
        self.aux = nn.Linear(128, 2)

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
def make_patient_batch(pid_list, labels, patient_locs, mel_cache, rng, train=True, aug=True):
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
                x[bi, k, 0] = augment_mel(mel_cache[path], rng) if aug else eval_mel(mel_cache[path])
            else:
                path = max(paths, key=lambda p: mel_cache[p].shape[1])
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
    wa_legacy = float((WA_W * recall).sum() / WA_W.sum())
    ch_num = float((CH_W * np.diag(cm)).sum())
    ch_den = float((CH_W * cm.sum(1)).sum())
    ch_wa = ch_num / ch_den if ch_den > 0 else float('nan')
    total = cm.sum()
    spec = np.zeros(n)
    for i in range(n):
        tn = total - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec[i] = tn / max(1, tn + fp)
    return acc, recall, f1, wa_legacy, ch_wa, cm, spec


def evaluate_probs(model, pid_list, labels, patient_locs, mel_cache, device, batch=64):
    """Returns (fused_probs Bx3, fused_pred, vote_probs Bx3, vote_pred, y)."""
    model.eval()
    fp_list, vp_list, y_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(pid_list), batch):
            pl = pid_list[i:i + batch]
            x, mask, y = make_patient_batch(pl, labels, patient_locs, mel_cache, None, train=False)
            xb = torch.from_numpy(x).to(device)
            maskb = torch.from_numpy(mask).to(device)
            logits, _, _ = model(xb, maskb)
            fused_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            B, K = x.shape[0], x.shape[1]
            e = model.encode(xb.view(B * K, 1, N_MELS, N_FRAMES))
            plog = torch.softmax(model.per_location_logits(e).view(B, K, 3), dim=-1).cpu().numpy()
            vote_probs = plog.mean(axis=1)
            fp_list.append(fused_probs); vp_list.append(vote_probs); y_list.append(y)
    fused_probs = np.concatenate(fp_list)
    vote_probs = np.concatenate(vp_list)
    y = np.concatenate(y_list)
    return fused_probs, fused_probs.argmax(1), vote_probs, vote_probs.argmax(1), y


def report(name, y_true, y_pred):
    acc, recall, f1, wa_leg, ch_wa, cm, spec = metrics(y_true, y_pred)
    print(f'--- {name} ---', flush=True)
    print(f'  accuracy      : {acc:.4f}', flush=True)
    print(f'  macro-F1      : {f1.mean():.4f}  (per-class {np.round(f1,3).tolist()})', flush=True)
    print(f'  legacy wacc   : {wa_leg:.4f} (w 0.1/0.2/1.0)', flush=True)
    print(f'  CHALLENGE WA  : {ch_wa:.4f} (official s_murmur, w Absent1/Unknown3/Present5)', flush=True)
    print(f'  per-class recall/sens: {np.round(recall,3).tolist()}', flush=True)
    print(f'  per-class spec       : {np.round(spec,3).tolist()}', flush=True)
    print('  confusion (rows=true [Absent,Unknown,Present], cols=pred):', flush=True)
    print('  ' + str(cm.tolist()), flush=True)
    return {'accuracy': float(acc), 'macro_f1': float(f1.mean()),
            'legacy_wacc': float(wa_leg), 'challenge_wa': float(ch_wa),
            'recall': np.round(recall, 4).tolist(), 'specificity': np.round(spec, 4).tolist(),
            'cm': cm.tolist()}


def load_or_make_split(patient_locs, labels):
    if os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON) as f:
            d = json.load(f)
        print(f'[split] loaded existing {SPLIT_JSON}', flush=True)
        return d['train'], d['val'], d['test']
    rng = np.random.RandomState(42)
    all_pids = sorted(patient_locs.keys())
    groups = {c: [] for c in range(3)}
    for p in all_pids:
        groups[labels[p]].append(p)
    train, val, test = [], [], []
    for c in range(3):
        lst = groups[c][:]
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(round(0.70 * n)); n_va = int(round(0.15 * n))
        train.extend(lst[:n_tr]); val.extend(lst[n_tr:n_tr + n_va]); test.extend(lst[n_tr + n_va:])
    train = sorted(train); val = sorted(val); test = sorted(test)
    with open(SPLIT_JSON, 'w') as f:
        json.dump({'train': train, 'val': val, 'test': test}, f)
    return train, val, test


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--present-boost', type=float, default=1.0,
                    help='multiplier on Present class weight, default 1.0')
    ap.add_argument('--boost-mode', default='both', choices=['both', 'ce'],
                    help='both = boost sampling probs AND CE weight; ce = CE weight only')
    ap.add_argument('--sampling', default='sqrt', choices=['sqrt', 'linear'])
    ap.add_argument('--es-metric', default='smurmur', choices=['smurmur', 'macro_f1'])
    ap.add_argument('--tag', default=None, help='filename suffix override')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device} torch={torch.__version__} seed={args.seed} '
          f'boost={args.present_boost} mode={args.boost_mode} sampling={args.sampling} es={args.es_metric}', flush=True)
    if torch.cuda.is_available():
        print(f'[env] gpu={torch.cuda.get_device_name(0)}', flush=True)

    rng = np.random.RandomState(args.seed)
    patient_locs, labels = collect_data()
    train_pids, val_pids, test_pids = load_or_make_split(patient_locs, labels)
    dist = lambda pids: [sum(1 for p in pids if labels[p] == c) for c in range(3)]
    print(f'[split] train={len(train_pids)} dist={dist(train_pids)}', flush=True)
    print(f'[split] val  ={len(val_pids)} dist={dist(val_pids)}', flush=True)
    print(f'[split] test ={len(test_pids)} dist={dist(test_pids)}', flush=True)

    all_paths = []
    for pid in list(train_pids) + list(val_pids) + list(test_pids):
        for paths in patient_locs[pid].values():
            all_paths.extend(paths)
    mel_cache = {}
    precompute_mels(all_paths, mel_cache)

    counts = np.array(dist(train_pids), dtype=float)
    N = len(train_pids)
    if args.sampling == 'sqrt':
        class_w = np.sqrt(N / (counts * 3))
    else:  # linear inverse frequency
        class_w = N / counts
    ce_w = class_w.copy()
    ce_w[2] *= args.present_boost            # boost Present in CE weight
    ce_w = ce_w / ce_w.mean()                # renormalize to mean 1.0
    if args.boost_mode == 'both':
        samp_w = ce_w                        # boost also affects sampling
    else:
        samp_w = class_w / class_w.mean()    # sampling stays sqrt-inverse-freq
    print(f'[imb] train counts={counts.astype(int).tolist()} sampling={args.sampling} '
          f'boost={args.present_boost} mode={args.boost_mode} '
          f'ce_weights={np.round(ce_w, 3).tolist()} samp_weights={np.round(samp_w, 3).tolist()}', flush=True)
    patient_w = np.array([samp_w[labels[p]] for p in train_pids], dtype=float)
    patient_probs = patient_w / patient_w.sum()

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_w, dtype=torch.float32, device=device))
    aux_fn = nn.CrossEntropyLoss()
    patience = args.patience
    n_per_epoch = len(train_pids)

    tag = args.tag or f's{args.seed}_pb{args.present_boost}_{args.boost_mode}_{args.sampling}_{args.es_metric}'
    npz_path = os.path.join(WORKDIR, f'v4_test_{tag}.npz')
    pt_path = os.path.join(WORKDIR, f'best_model_v4_{tag}.pt')

    # ================= FUSION MODEL =================
    print(f'\n======== TRAIN FUSION MODEL tag={tag} ========', flush=True)
    model = FusionNet(n_class=3).to(device)
    n_params = count_params(model)
    print(f'[model] params={n_params:,}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_m, best_ep, bad_epochs, best_state = -1.0, -1, 0, None
    t0 = time.time(); ep_times = []
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = rng.choice(n_per_epoch, size=n_per_epoch, p=patient_probs, replace=True)
        n_batches = (n_per_epoch + args.batch - 1) // args.batch
        run_loss, run_corr = 0.0, 0
        t_ep = time.time()
        for b in range(n_batches):
            x, mask, y = make_patient_batch(
                [train_pids[i] for i in idx[b * args.batch:(b + 1) * args.batch]],
                labels, patient_locs, mel_cache, rng, train=True, aug=True)
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
        f_p, f_pr, v_p, v_pr, y_v = evaluate_probs(model, val_pids, labels, patient_locs, mel_cache, device)
        acc, recall, f1, wa_leg, ch_wa, cm, spec = metrics(y_v, f_pr)
        es_val = ch_wa if args.es_metric == 'smurmur' else float(f1.mean())
        ep_times.append(time.time() - t_ep)
        print(f'[fusion ep {ep:02d}] loss={run_loss/n_per_epoch:.4f} train_acc={run_corr/n_per_epoch:.4f} '
              f'val_acc={acc:.4f} macroF1={f1.mean():.4f} chWA={ch_wa:.4f} '
              f'recall={np.round(recall, 3).tolist()} ({ep_times[-1]:.1f}s)', flush=True)
        if es_val > best_m:
            best_m, best_ep, bad_epochs = es_val, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'[fusion] early stop at epoch {ep}', flush=True)
                break
    model.load_state_dict(best_state)
    print(f'[fusion] best epoch={best_ep} val_{args.es_metric}={best_m:.4f} | '
          f'{np.mean(ep_times):.1f}s/epoch | total {time.time()-t0:.0f}s', flush=True)

    # ---- VAL ----
    f_p, f_pr, v_p, v_pr, y_v = evaluate_probs(model, val_pids, labels, patient_locs, mel_cache, device)
    print('\n===== FUSION MODEL: VAL =====', flush=True)
    val_fused = report('FUSION val (attention-fused)', y_v, f_pr)
    val_vote = report('FUSION-vote val (per-loc majority)', y_v, v_pr)

    # ---- TEST ----
    f_p, f_pr, v_p, v_pr, y_t = evaluate_probs(model, test_pids, labels, patient_locs, mel_cache, device)
    print('\n===== FUSION MODEL: HELD-OUT TEST =====', flush=True)
    test_fused = report('FUSION test (attention-fused)', y_t, f_pr)
    test_vote = report('FUSION-vote test (per-loc majority)', y_t, v_pr)

    np.savez(npz_path, fused_probs=f_p, fused_pred=f_pr,
             vote_probs=v_p, vote_pred=v_pr, y=y_t)
    torch.save({'state_dict': best_state, 'epoch': best_ep, 'val_es': best_m,
                'val_es_metric': args.es_metric, 'n_params': n_params, 'arch': 'FusionNet',
                'kind': 'fusion-v4', 'seed': args.seed, 'present_boost': args.present_boost,
                'boost_mode': args.boost_mode, 'sampling': args.sampling,
                'es_metric': args.es_metric, 'classes': CLASS_NAMES,
                'val_fused': val_fused, 'test_fused': test_fused}, pt_path)
    print(f'[save] {pt_path}', flush=True)
    print(f'[save] {npz_path}', flush=True)

    # persist results
    results = {}
    if os.path.exists(RESULT_JSON):
        with open(RESULT_JSON) as f:
            results = json.load(f)
    results[tag] = {'seed': args.seed, 'present_boost': args.present_boost,
                    'boost_mode': args.boost_mode, 'sampling': args.sampling, 'es_metric': args.es_metric,
                    'best_epoch': best_ep, 'val_es': float(best_m),
                    'val_fused': val_fused, 'val_vote': val_vote,
                    'test_fused': test_fused, 'test_vote': test_vote}
    with open(RESULT_JSON, 'w') as f:
        json.dump(results, f, indent=1, default=lambda o: int(o) if isinstance(o, np.integer)
                  else (float(o) if isinstance(o, np.floating) else str(o)))
    print(f'[result] saved to {RESULT_JSON}', flush=True)
    print('================ DONE ================', flush=True)


if __name__ == '__main__':
    main()
