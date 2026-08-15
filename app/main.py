"""FastAPI backend for the Open Stethoscope QC companion.

Endpoints
---------
POST /api/qc/analyze   raw float32 PCM -> real DSP QC metrics + position/murmur inference
GET  /api/simulator/manifest   teaching-simulator clip manifest (real CirCor data)
GET  /api/simulator/audio/<id> stream one real clip as WAV
GET  /api/models/info   honest held-out test metrics of the deployed models
GET  /api/health
"""
import os, json, io, argparse
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from qc_engine import analyze as qc_analyze

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
# 模型可能由 train_qc_models.py 存到 --workdir（默认 ./qc_work），也可能手动放到 models/；
# 按候选顺序查找，任一存在即加载。
MODEL_CANDIDATES = [
    os.path.join(ROOT, "models", "qc_models.pt"),
    os.path.join(ROOT, "qc_work", "qc_models.pt"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])
ASSETS_DIR = os.path.join(BASE, "assets")

app = FastAPI(title="Open Stethoscope QC Companion", version="1.0.0")

MODEL = None
MANIFEST = None
CLIP_INDEX = {}
INFER = None


def _load_model():
    global MODEL, INFER
    import torch
    import torch.nn.functional as F
    import librosa
    sysmod = __import__("sys")
    sysmod.path.insert(0, os.path.join(BASE, ".."))

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    from model_defs import PCGNet
    model = PCGNet(n_pos=4, n_mur=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pp = ckpt["preproc"]
    locs = ckpt["loc_names"]
    murs = ckpt["murmur_names"]
    test = ckpt["test"]
    MODEL = (model, pp, locs, murs, test)

    def infer(x, sr):
        if sr != pp["sr"]:
            x = librosa.resample(x.astype(np.float32), orig_sr=int(sr), target_sr=pp["sr"])
        m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=pp["sr"], n_fft=pp["n_fft"],
                                           hop_length=pp["hop"], n_mels=pp["n_mels"],
                                           fmin=pp["fmin"], fmax=pp["fmax"])
        m = librosa.power_to_db(m, ref=np.max).astype(np.float32)
        if m.shape[1] >= pp["n_frames"]:
            m = m[:, :pp["n_frames"]]
        else:
            m = np.pad(m, ((0, 0), (0, pp["n_frames"] - m.shape[1])))
        t = torch.from_numpy(m[None, None])
        with torch.no_grad():
            lp, lm = model(t)
        pos_prob = F.softmax(lp[0], dim=0).numpy()
        mur_prob = F.softmax(lm[0], dim=0).numpy()
        return pos_prob, mur_prob

    INFER = infer


def _load_manifest():
    global MANIFEST, CLIP_INDEX
    manifest_path = os.path.join(ASSETS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        MANIFEST = {"clips": []}
        return
    with open(manifest_path) as f:
        MANIFEST = json.load(f)
    for c in MANIFEST["clips"]:
        CLIP_INDEX[c["id"]] = c


def position_verdict(pos_prob, declared=None):
    top = int(np.argmax(pos_prob))
    conf = float(pos_prob[top])
    names = MODEL[2]
    gap = float(np.sort(pos_prob)[-1] - np.sort(pos_prob)[-2]) if len(pos_prob) > 1 else conf
    if conf >= 0.60:
        status = "ok"
    elif conf >= 0.40 and gap >= 0.15:
        status = "warn"
    else:
        status = "fail"
    verdict = {
        "top_location": names[top], "confidence": round(conf, 3), "gap": round(gap, 3),
        "probabilities": {n: round(float(p), 3) for n, p in zip(names, pos_prob)},
        "status": status,
    }
    if declared:
        verdict["declared"] = declared
        verdict["match"] = (names[top] == declared)
    return verdict


@app.on_event("startup")
def startup():
    _load_manifest()
    try:
        _load_model()
        print(f"[startup] model loaded: {MODEL_PATH}")
    except Exception as e:  # FileNotFoundError / ModuleNotFoundError / torch errors
        print(f"[startup] WARNING: QC model unavailable ({type(e).__name__}: {e}) — position/murmur inference disabled")
    print(f"[startup] simulator clips: {len(CLIP_INDEX)}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None,
            "simulator_clips": len(CLIP_INDEX)}


@app.post("/api/qc/analyze")
async def qc_analyze_ep(request: Request):
    sr = int(request.headers.get("X-Sample-Rate", "48000"))
    declared = request.headers.get("X-Declared-Position", None)
    body = await request.body()
    if len(body) < 128:
        return JSONResponse({"error": "audio too short"}, status_code=422)
    x = np.frombuffer(body, dtype=np.float32).copy()
    if np.max(np.abs(x)) > 2.0:
        x = x / (np.max(np.abs(x)) + 1e-9)
    res = qc_analyze(x, sr)
    if MODEL is None:
        res["position"] = {"error": "model not loaded (run train_qc_models.py first)"}
        res["murmur"] = {"error": "model not loaded (run train_qc_models.py first)"}
        return JSONResponse(res)
    pos_prob, mur_prob = INFER(x, sr)
    res["position"] = position_verdict(pos_prob, declared)
    res["murmur"] = {
        "probabilities": {n: round(float(p), 3) for n, p in zip(MODEL[3], mur_prob)},
        "top": MODEL[3][int(np.argmax(mur_prob))],
        "note": "筛查信号，非诊断结论。murmur 概率由真实数据训练的模型给出。",
    }
    return JSONResponse(res)


@app.get("/api/simulator/manifest")
def simulator_manifest():
    return MANIFEST


@app.get("/api/simulator/audio/{clip_id}")
def simulator_audio(clip_id: str):
    if clip_id not in CLIP_INDEX:
        return JSONResponse({"error": "clip not found"}, status_code=404)
    path = os.path.join(ASSETS_DIR, "clips", CLIP_INDEX[clip_id]["file"])
    x, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"X-Sample-Rate": str(sr)})


@app.get("/api/models/info")
def model_info():
    if MODEL is None:
        return {"error": "model not loaded (run train_qc_models.py first)"}
    return {
        "position_model": {
            "name": "PCGNet position head",
            "trained_on": "CirCor DigiScope 1.0.3 (real expert-annotated recordings)",
            "split": "patient-disjoint 70/15/15 (shared with main murmur model)",
            "test": MODEL[4]["position_accuracy"],
            "test_macro_f1": MODEL[4]["position_macro_f1"],
            "confusion_matrix": MODEL[4]["position_cm"],
            "classes": MODEL[2],
        },
        "murmur_screening": {
            "name": "PCGNet murmur head",
            "classes": MODEL[3],
            "test_accuracy": MODEL[4]["murmur_accuracy"],
            "test_macro_f1": MODEL[4]["murmur_macro_f1"],
            "test_s_murmur": MODEL[4]["murmur_s_murmur"],
            "disclaimer": "Screening signal only, not a diagnosis.",
        },
    }


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
