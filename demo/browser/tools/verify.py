#!/usr/bin/env python3
"""Verify the browser inference chain against the training environment.

Local chain (this machine, no librosa):
    clip.wav → mel_np (numpy replica) → ONNX → probabilities

Server chain (P4 training env, ground truth):
    clip.wav → librosa 1.0.0 mel → torch FusionNet → probabilities

The mel matrices and probabilities must agree — that proves the browser
demo (JS mel + ONNX) reproduces the trained model's behaviour.

    python3 demo/browser/tools/verify.py <clip.wav> ... [--server-json out.json]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mel_np import build_input, mel_spectrogram, params, to_frames  # noqa: E402

ONNX = os.path.join(HERE, "..", "assets", "model.onnx")
SERVER_SCRIPT = os.path.join(HERE, "verify_server.py")
SERVER_HOST = "root@awzmdesbrtd1rhbtsnow.deepln.com"
SERVER_PORT = "41942"
SERVER_DIR = "/root/verify-browser-demo"


def local_probs(wav: str):
    x, sr = sf.read(wav, dtype="float32")
    p = params()
    if sr != p["sr"]:
        x = np.interp(np.linspace(0, len(x), int(len(x) * p["sr"] / sr), endpoint=False), np.arange(len(x)), x).astype(np.float32)
    xb, mask = build_input(x, loc_idx=p["locs"].index("MV"), p=p)
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"x": xb, "mask": mask})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return probs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--server-json", default=None, help="run server verification and save results here")
    args = ap.parse_args()

    p = params()
    local_mel = {}
    local_prob = {}
    for wav in args.wavs:
        x, sr = sf.read(wav, dtype="float32")
        if sr != p["sr"]:
            x = np.interp(np.linspace(0, len(x), int(len(x) * p["sr"] / sr), endpoint=False), np.arange(len(x)), x).astype(np.float32)
        m = to_frames(mel_spectrogram(x, p), p)
        local_mel[os.path.basename(wav)] = m
        probs, _ = local_probs(wav)
        local_prob[os.path.basename(wav)] = probs
        print(f"[local] {os.path.basename(wav)} → probs {np.round(probs, 4).tolist()}")

    if args.server_json:
        subprocess.run(["sshpass", "-p", "9OgGEcXtMp7Ivh1O", "ssh", "-o", "StrictHostKeyChecking=no",
                        "-p", SERVER_PORT, SERVER_HOST,
                        f"mkdir -p {SERVER_DIR}/clips"], check=True)
        for wav in args.wavs:
            subprocess.run(["sshpass", "-p", "9OgGEcXtMp7Ivh1O", "scp", "-o", "StrictHostKeyChecking=no",
                            "-P", SERVER_PORT, wav,
                            f"{SERVER_HOST}:{SERVER_DIR}/clips/"], check=True)
        subprocess.run(["sshpass", "-p", "9OgGEcXtMp7Ivh1O", "scp", "-o", "StrictHostKeyChecking=no",
                        "-P", SERVER_PORT, SERVER_SCRIPT,
                        f"{SERVER_HOST}:{SERVER_DIR}/"], check=True)
        clip_names = " ".join(f"{SERVER_DIR}/clips/{os.path.basename(w)}" for w in args.wavs)
        cmd = (f"cd {SERVER_DIR} && /root/venv_heart/bin/python3 verify_server.py {clip_names} "
               f"> {SERVER_DIR}/results.json")
        subprocess.run(["sshpass", "-p", "9OgGEcXtMp7Ivh1O", "ssh", "-o", "StrictHostKeyChecking=no",
                        "-p", SERVER_PORT, SERVER_HOST, cmd], check=True)
        subprocess.run(["sshpass", "-p", "9OgGEcXtMp7Ivh1O", "scp", "-o", "StrictHostKeyChecking=no",
                        "-P", SERVER_PORT, f"{SERVER_HOST}:{SERVER_DIR}/results.json",
                        args.server_json], check=True)

        with open(args.server_json) as f:
            server = json.load(f)
        for name, s in server.items():
            sp = np.array(s["probs"])
            lp = local_prob[name]
            mel_err = float(np.abs(np.array(s["mel"]) - local_mel[name]).max())
            print(f"[match] {name}: prob max-abs-diff = {np.abs(sp - lp).max():.6f} | "
                  f"mel max-abs-diff = {mel_err:.5f} (server probs {np.round(sp, 4).tolist()})")
            if np.abs(sp - lp).max() > 1e-3 or mel_err > 0.5:
                print(f"  ⚠️  MISMATCH above tolerance")
                sys.exit(1)
        print("🎉 LOCAL CHAIN == SERVER GROUND TRUTH")


if __name__ == "__main__":
    main()
