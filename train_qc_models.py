"""Train the QC auxiliary models on the real CirCor DigiScope dataset.

Two heads over one shared 2D-CNN encoder on log-mel spectrograms (SR=4 kHz,
40 mel bands x 126 frames, band 25-2000 Hz, identical to the murmur model):
  1. position head  -> auscultation location AV / PV / TV / MV
  2. murmur head    -> expert murmur label Absent / Unknown / Present

The patient-level 70/15/15 split is REUSED from v3_split_seed42.json so there
is no leakage with the main murmur model's evaluation. All labels and audio are
the real expert-annotated CirCor data; no synthetic samples are used.

Usage:
  python train_qc_models.py --data-csv <training_data.csv> --data-dir <wavdir> \
      --workdir ./qc_work --epochs 25
"""
import os, re, sys, time, json, argparse, hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
SR = 4000
N_FRAMES = 126
N_MELS = 40
N_FFT = 512
HOP = 256
FMIN, FMAX = 25, 2000

LOCS = ["AV", "PV", "TV", "MV"]
MURMUR_MAP = {"Absent": 0, "Unknown": 1, "Present": 2}
MURMUR_NAMES = ["Absent", "Unknown", "Present"]
CH_W = np.array([1.0, 3.0, 5.0])
AUX_LAMBDA = 0.3


def set_seed(s):
    random = __import__("random")
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def full_mel(path):
    x, sr = sf.read(path, dtype="float32")
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    m = librosa.power_to_db(m, ref=np.max)
    return m.astype(np.float32)


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


class PCGNet(nn.Module):
    def __init__(self, n_pos=4, n_mur=3, embed=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head_pos = nn.Sequential(nn.Linear(embed, 128), nn.ReLU(inplace=True), nn.Linear(128, n_pos))
        self.head_mur = nn.Sequential(nn.Linear(embed, 128), nn.ReLU(inplace=True), nn.Linear(128, n_mur))

    def forward(self, x):
        e = self.encoder(x)
        e = self.avgpool(e).flatten(1)
        return self.head_pos(e), self.head_mur(e)

    def embed(self, x):
        e = self.encoder(x)
        return self.avgpool(e).flatten(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workdir", default="./qc_work")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cpu")
    os.makedirs(args.workdir, exist_ok=True)
    mel_cache_dir = os.path.join(args.workdir, "mel_cache")

    df = pd.read_csv(args.data_csv)
    labels = {}
    for pid, mur in zip(df["Patient ID"].astype(str).str.strip(), df["Murmur"]):
        if mur in MURMUR_MAP:
            labels[int(pid)] = MURMUR_MAP[mur]

    rows = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            m = re.match(r"(\d+)_(AV|PV|TV|MV)(?:_\d+)?\.wav", f)
            if not m:
                continue
            pid, loc = int(m.group(1)), m.group(2)
            if pid not in labels:
                continue
            rows.append({"pid": pid, "loc": LOCS.index(loc), "mur": labels[pid], "path": os.path.join(root, f)})
    if not rows:
        sys.exit("[data] no wav files matched; check --data-dir")
    print(f"[data] wavs={len(rows)} patients={len(set(r['pid'] for r in rows))}")

    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_split_seed42.json")):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_split_seed42.json")) as f:
            split = json.load(f)
        tr_set, va_set, te_set = set(split["train"]), set(split["val"]), set(split["test"])
    else:
        rng = np.random.RandomState(SEED)
        pids = sorted(set(r["pid"] for r in rows))
        rng.shuffle(pids)
        n = len(pids)
        tr_set = set(pids[:int(0.70 * n)]); va_set = set(pids[int(0.70 * n):int(0.85 * n)])
        te_set = set(pids[int(0.85 * n):])

    def part(pid): return ("train" if pid in tr_set else "val" if pid in va_set else "test")
    for r in rows:
        r["part"] = part(r["pid"])
    tr, va, te = [r for r in rows if r["part"] == "train"], \
                 [r for r in rows if r["part"] == "val"], \
                 [r for r in rows if r["part"] == "test"]
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} (patient-disjoint)")

    mel_cache = {}
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(mel_cache_dir, exist_ok=True)
    todo = [p for r in tr + va + te
            for p in [r["path"]]
            if p not in mel_cache and not os.path.exists(
                os.path.join(mel_cache_dir, hashlib.sha1(p.encode()).hexdigest()[:20] + ".npy"))]
    if todo:
        print(f"[feat] computing {len(todo)} mels (parallel)...", flush=True)
        with ProcessPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(full_mel, todo, chunksize=32))
        for p, m in zip(todo, results):
            cp = os.path.join(mel_cache_dir, hashlib.sha1(p.encode()).hexdigest()[:20] + ".npy")
            np.save(cp, m)
    for r in tr + va + te:
        p = r["path"]
        if p in mel_cache:
            continue
        cp = os.path.join(mel_cache_dir, hashlib.sha1(p.encode()).hexdigest()[:20] + ".npy")
        mel_cache[p] = np.load(cp)
    print(f"[feat] cached {len(mel_cache)} mels")

    pos_counts = np.bincount([r["loc"] for r in tr], minlength=4)
    mur_counts = np.bincount([r["mur"] for r in tr], minlength=3)
    pos_w = np.sqrt(len(tr) / (pos_counts * 4 + 1e-9)); pos_w = pos_w / pos_w.mean()
    mur_w = np.sqrt(len(tr) / (mur_counts * 3 + 1e-9)); mur_w = mur_w / mur_w.mean()
    print(f"[imb] position counts={pos_counts.tolist()} weights={np.round(pos_w,3).tolist()}")
    print(f"[imb] murmur   counts={mur_counts.tolist()} weights={np.round(mur_w,3).tolist()}")

    model = PCGNet().to(device)
    print(f"[model] params={count_params(model):,}")

    pos_fn = nn.CrossEntropyLoss(weight=torch.tensor(pos_w, dtype=torch.float32))
    mur_fn = nn.CrossEntropyLoss(weight=torch.tensor(mur_w, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def make_batch(rows_b):
        B = len(rows_b)
        x = np.zeros((B, 1, N_MELS, N_FRAMES), np.float32)
        for bi, r in enumerate(rows_b):
            x[bi, 0] = eval_mel(mel_cache[r["path"]])
        return torch.from_numpy(x)

    def evaluate(rows_ev):
        model.eval()
        pos_y, mur_y, pos_p, mur_p = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(rows_ev), args.batch):
                chunk = rows_ev[i:i + args.batch]
                xb = make_batch(chunk)
                lp, lm = model(xb)
                pos_p.extend(lp.argmax(1).tolist())
                mur_p.extend(lm.argmax(1).tolist())
                pos_y.extend([r["loc"] for r in chunk])
                mur_y.extend([r["mur"] for r in chunk])
        return np.array(pos_y), np.array(mur_y), np.array(pos_p), np.array(mur_p)

    def acc(y, p):
        return float((y == p).mean())

    def macro_f1(y, p, n):
        cm = np.zeros((n, n), dtype=int)
        for a, b in zip(y, p):
            cm[a, b] += 1
        recall = np.array([cm[i, i] / max(1, cm[i].sum()) for i in range(n)])
        prec = np.array([cm[i, i] / max(1, cm[:, i].sum()) for i in range(n)])
        return float((2 * prec * recall / (prec + recall + 1e-9)).mean())

    def s_murmur(y, p):
        cm = np.zeros((3, 3), dtype=int)
        for a, b in zip(y, p):
            cm[a, b] += 1
        num = float((CH_W * np.diag(cm)).sum())
        den = float((CH_W * cm.sum(1)).sum())
        return num / den if den > 0 else float("nan")

    best_score, best_ep, bad = -1.0, -1, 0
    best_state = None
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = rng.permutation(len(tr))
        run_pos, run_mur = 0, 0
        for i in range(0, len(tr), args.batch):
            chunk_idx = idx[i:i + args.batch]
            chunk = [tr[j] for j in chunk_idx]
            xb = make_batch(chunk)
            xa = torch.stack([torch.from_numpy(augment_mel(mel_cache[r["path"]], rng)) for r in chunk]).unsqueeze(1)
            opt.zero_grad()
            lp, lm = model(xa)
            loss = pos_fn(lp, torch.tensor([r["loc"] for r in chunk])) + \
                mur_fn(lm, torch.tensor([r["mur"] for r in chunk]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_pos += (lp.argmax(1).numpy() == np.array([r["loc"] for r in chunk])).sum()
            run_mur += (lm.argmax(1).numpy() == np.array([r["mur"] for r in chunk])).sum()
        sched.step()
        pos_y, mur_y, pos_p, mur_p = evaluate(va)
        pos_acc = acc(pos_y, pos_p)
        mur_acc = acc(mur_y, mur_p)
        mur_f1 = macro_f1(mur_y, mur_p, 3)
        smur = s_murmur(mur_y, mur_p)
        score = 0.5 * pos_acc + 0.5 * mur_f1
        print(f"[ep {ep:02d}] loss={loss.item():.4f} tr_pos={run_pos/len(tr):.3f} tr_mur={run_mur/len(tr):.3f} "
              f"| val pos_acc={pos_acc:.4f} mur_acc={mur_acc:.4f} mur_macroF1={mur_f1:.4f} s_murmur={smur:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    print(f"\n[batch] best epoch={best_ep} val score={best_score:.4f}")

    pos_y, mur_y, pos_p, mur_p = evaluate(te)
    print("\n===== HELD-OUT TEST (patient-disjoint) =====")
    pos_cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(pos_y, pos_p):
        pos_cm[a, b] += 1
    print("position confusion (rows=truth AV/PV/TV/MV):")
    print(pos_cm.tolist())
    print(f"position test accuracy = {acc(pos_y, pos_p):.4f}")
    print(f"position macro-F1      = {macro_f1(pos_y, pos_p, 4):.4f}")
    print(f"murmur test accuracy   = {acc(mur_y, mur_p):.4f}")
    print(f"murmur macro-F1        = {macro_f1(mur_y, mur_p, 3):.4f}")
    print(f"murmur s_murmur        = {s_murmur(mur_y, mur_p):.4f}")
    print(f"[time] total {time.time()-t0:.0f}s")

    out = os.path.join(args.workdir, "qc_models.pt")
    torch.save({
        "state_dict": best_state, "epoch": best_ep, "arch": "PCGNet",
        "split": "v3_split_seed42.json",
        "test": {
            "position_accuracy": acc(pos_y, pos_p), "position_macro_f1": macro_f1(pos_y, pos_p, 4),
            "position_cm": pos_cm.tolist(),
            "murmur_accuracy": acc(mur_y, mur_p), "murmur_macro_f1": macro_f1(mur_y, mur_p, 3),
            "murmur_s_murmur": s_murmur(mur_y, mur_p),
        },
        "preproc": {"sr": SR, "n_mels": N_MELS, "n_frames": N_FRAMES, "n_fft": N_FFT,
                    "hop": HOP, "fmin": FMIN, "fmax": FMAX},
        "loc_names": LOCS, "murmur_names": MURMUR_NAMES,
    }, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
