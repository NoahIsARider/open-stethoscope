"""Check position-head confidence calibration on the held-out test set.

Answers: when the model says confidence >= X, how often is it right?
Used to set honest verdict thresholds in the QC UI.
"""
import os, sys, re, json
import numpy as np, pandas as pd
import soundfile as sf, librosa, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATA_CSV = sys.argv[1]
DATA_DIR = sys.argv[2]
CKPT = sys.argv[3]

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
from app.model_defs import PCGNet
model = PCGNet(n_pos=4, n_mur=3)
model.load_state_dict(ckpt["state_dict"])
model.eval()
pp = ckpt["preproc"]
locs = ckpt["loc_names"]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v3_split_seed42.json")) as f:
    split = json.load(f)
test_set = {str(p) for p in split["test"]}

df = pd.read_csv(DATA_CSV)
labels = {str(p): m for p, m in zip(df["Patient ID"], df["Murmur"])}

rows = []
for root, _, files in os.walk(DATA_DIR):
    for fn in files:
        m = re.match(r"(\d+)_(AV|PV|TV|MV)(?:_\d+)?\.wav", fn)
        if not m:
            continue
        pid, loc = m.group(1), m.group(2)
        if pid in test_set and labels.get(pid) in ("Absent", "Present", "Unknown"):
            rows.append((os.path.join(root, fn), loc))

confs = []
for path, loc in rows:
    x, sr = sf.read(path, dtype="float32")
    if sr != pp["sr"]:
        x = librosa.resample(x, orig_sr=sr, target_sr=pp["sr"])
    mel = librosa.feature.melspectrogram(y=x, sr=pp["sr"], n_fft=pp["n_fft"], hop_length=pp["hop"],
                                         n_mels=pp["n_mels"], fmin=pp["fmin"], fmax=pp["fmax"])
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if mel.shape[1] >= pp["n_frames"]:
        mel = mel[:, :pp["n_frames"]]
    else:
        mel = np.pad(mel, ((0, 0), (0, pp["n_frames"] - mel.shape[1])))
    with torch.no_grad():
        lp, _ = model(torch.from_numpy(mel[None, None]))
    prob = F.softmax(lp[0], dim=0).numpy()
    confs.append((float(prob.max()), int(loc == locs[int(prob.argmax())])))

confs = np.array(confs)
print(f"n_test_wavs={len(confs)} overall_acc={confs[:,1].mean():.3f}")
for lo, hi in [(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.0)]:
    mask = (confs[:,0]>=lo)&(confs[:,0]<hi)
    if mask.sum()>0:
        print(f"  conf[{lo},{hi}) n={mask.sum():4d} acc={confs[mask,1].mean():.3f}")
print("  conf>=0.60:", f"n={(confs[:,0]>=0.6).sum()} acc={confs[confs[:,0]>=0.6,1].mean():.3f}")
print("  conf>=0.75:", f"n={(confs[:,0]>=0.75).sum()} acc={confs[confs[:,0]>=0.75,1].mean():.3f}")
