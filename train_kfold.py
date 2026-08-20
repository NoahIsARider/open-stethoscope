#!/usr/bin/env python3
"""5-fold patient-stratified CV for FusionNet (v4 config).
Reuses train_v4's model/training machinery. Outputs:
  - per-fold models + metrics
  - out-of-fold (OOF) ensemble evaluation
  - CV mean +/- std of official s_murmur
"""
import os, sys, json, time, argparse
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

CH_W = np.array([1.0, 3.0, 5.0])


def smurmur(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    num = float((CH_W * np.diag(cm)).sum())
    den = float((CH_W * cm.sum(1)).sum())
    return num / den if den > 0 else float('nan'), cm


def stratified_folds(pids, labels, n_folds=5, seed=42):
    by_label = {c: [] for c in range(3)}
    for p in pids:
        by_label[labels[p]].append(p)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    for c in range(3):
        arr = list(by_label[c])
        rng.shuffle(arr)
        for i, p in enumerate(arr):
            folds[i % n_folds].append(p)
    return folds


def stratified_split(pids, labels, val_frac=0.15, seed=1000 + int(time.time()) % 1000):
    by_label = {c: [] for c in range(3)}
    for p in pids:
        by_label[labels[p]].append(p)
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in range(3):
        arr = list(by_label[c])
        rng.shuffle(arr)
        nv = max(1, int(round(len(arr) * val_frac)))
        va.extend(arr[:nv]); tr.extend(arr[nv:])
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--es-metric', default='macro_f1')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] device={device}', flush=True)

    patient_locs, labels = T.collect_data()
    all_pids = sorted(patient_locs.keys())
    folds = stratified_folds(all_pids, labels, args.folds, args.seed)
    print(f'[folds] sizes={[len(f) for f in folds]} '
          f'dist={[[sum(1 for p in f if labels[p]==c) for c in range(3)] for f in folds]}', flush=True)

    all_paths = []
    for pid in all_pids:
        for paths in patient_locs[pid].values():
            all_paths.extend(paths)
    mel_cache = {}
    T.precompute_mels(all_paths, mel_cache)

    results = {}
    oof = {}  # pid -> (probs, true_label) from the model that never saw this patient
    for fold in range(args.folds):
        T.set_seed(args.seed + fold * 7 + 1)
        test_pids = folds[fold]
        pool = [p for i in range(args.folds) if i != fold for p in folds[i]]
        train_pids, val_pids = stratified_split(pool, labels, val_frac=0.15, seed=args.seed + fold * 31)
        dist = lambda pids: [sum(1 for p in pids if labels[p] == c) for c in range(3)]
        print(f'\n======== FOLD {fold+1}/{args.folds} ========', flush=True)
        print(f'[split] train={len(train_pids)} dist={dist(train_pids)} '
              f'val={len(val_pids)} dist={dist(val_pids)} test={len(test_pids)} dist={dist(test_pids)}', flush=True)

        counts = np.array(dist(train_pids), dtype=float)
        N = len(train_pids)
        class_w = np.sqrt(N / (counts * 3))
        ce_w = class_w / class_w.mean()
        print(f'[imb] ce_weights={np.round(ce_w,3).tolist()}', flush=True)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(ce_w, dtype=torch.float32, device=device))
        aux_fn = torch.nn.CrossEntropyLoss()
        patient_w = np.array([ce_w[labels[p]] for p in train_pids], dtype=float)
        patient_probs = patient_w / patient_w.sum()

        model = T.FusionNet(n_class=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_m, best_ep, bad, best_state = -1.0, -1, 0, None
        t0 = time.time()
        rng = np.random.RandomState(args.seed + fold * 7 + 1)
        for ep in range(1, args.epochs + 1):
            model.train()
            idx = rng.choice(len(train_pids), size=len(train_pids), p=patient_probs, replace=True)
            n_b = (len(train_pids) + 31) // 32
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
            es_val = ch_wa if args.es_metric == 'smurmur' else float(f1.mean())
            print(f'[fold{fold+1} ep{ep:02d}] loss={run_loss/len(train_pids):.4f} '
                  f'train_acc={run_corr/len(train_pids):.4f} val_acc={acc:.4f} macroF1={f1.mean():.4f} '
                  f'chWA={ch_wa:.4f} recall={np.round(recall,3).tolist()}', flush=True)
            if es_val > best_m:
                best_m, best_ep, bad = es_val, ep, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'[fold{fold+1}] early stop ep {ep}', flush=True)
                    break
        model.load_state_dict(best_state)
        print(f'[fold{fold+1}] best_ep={best_ep} val_{args.es_metric}={best_m:.4f} '
              f'total={time.time()-t0:.0f}s', flush=True)

        # fold test evaluation
        f_p, f_pr, v_p, v_pr, y_t = T.evaluate_probs(model, test_pids, labels, patient_locs, mel_cache, device)
        acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(y_t, f_pr)
        print(f'[fold{fold+1} TEST] acc={acc:.4f} macroF1={f1.mean():.4f} s_murmur={ch_wa:.4f} '
              f'recall={np.round(recall,3).tolist()}', flush=True)
        s_m, cm_t = smurmur(y_t, f_pr)
        results[f'fold{fold+1}'] = {'acc': acc, 'macro_f1': float(f1.mean()), 's_murmur': s_m,
                                    'recall': recall.tolist(), 'best_ep': best_ep, 'val_es': best_m}
        torch.save({'state_dict': best_state, 'epoch': best_ep, 'val_es': best_m,
                    'arch': 'FusionNet', 'kind': 'fusion-kfold', 'fold': fold, 'seed': args.seed,
                    'classes': T.CLASS_NAMES}, os.path.join(WORKDIR, f'best_model_kfold_f{fold+1}.pt'))
        for pid, pr, y in zip(test_pids, f_p, y_t):
            oof[pid] = (pr, y)

    # CV aggregate
    sms = [results[f'fold{i+1}']['s_murmur'] for i in range(args.folds)]
    print(f'\n===== CV SUMMARY (official s_murmur) =====', flush=True)
    print(f'per-fold: {[round(s,4) for s in sms]}', flush=True)
    print(f'mean={np.mean(sms):.4f} std={np.std(sms):.4f}', flush=True)

    # OOF ensemble
    oof_pids = sorted(oof.keys())
    oof_probs = np.stack([oof[p][0] for p in oof_pids])
    oof_y = np.array([oof[p][1] for p in oof_pids])
    oof_pred = oof_probs.argmax(1)
    acc, recall, f1, wa_leg, ch_wa, cm, spec = T.metrics(oof_y, oof_pred)
    print(f'\n===== OOF ENSEMBLE (each patient voted by a model that never saw them) =====', flush=True)
    print(f'acc={acc:.4f} macroF1={f1.mean():.4f} s_murmur={ch_wa:.4f} recall={np.round(recall,3).tolist()}', flush=True)
    print(f'confusion:\n{cm}', flush=True)
    results['cv_mean'] = float(np.mean(sms)); results['cv_std'] = float(np.std(sms))
    results['oof'] = {'acc': acc, 'macro_f1': float(f1.mean()), 's_murmur': ch_wa,
                      'recall': recall.tolist(), 'cm': cm.tolist()}
    with open(os.path.join(WORKDIR, 'kfold_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[save] kfold_results.json', flush=True)
    print('================ KFOLD DONE ================', flush=True)


if __name__ == '__main__':
    main()
