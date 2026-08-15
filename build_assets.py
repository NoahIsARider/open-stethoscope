"""Build the teaching-simulator asset library from the real CirCor dataset.

Selects real expert-annotated recordings (all 4 auscultation positions x the 3
murmur classes), copies the audio into app/assets/clips/ and writes a manifest
with the true patient metadata from training_data.csv (age group, sex, murmur
class, murmur characterization, outcome). No synthetic audio, no fabricated
labels.

Usage:
  python build_assets.py --data-csv <training_data.csv> --data-dir <wavdir> --out ./app/assets
"""
import os, re, json, sys, argparse, shutil
import soundfile as sf

MAX_PER_BUCKET = 12
MIN_SECONDS = 6.0

LOCS = ["AV", "PV", "TV", "MV"]
MURMUR_NAMES = ["Absent", "Unknown", "Present"]
SEX_CN = {"Female": "女", "Male": "男"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="./app/assets")
    args = ap.parse_args()

    with open(args.data_csv) as f:
        lines = f.readlines()
    header = [h.strip() for h in lines[0].rstrip("\n").split(",")]
    meta = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        vals = ln.rstrip("\n").split(",")
        row = dict(zip(header, vals))
        pid = str(row["Patient ID"]).strip()
        if row.get("Murmur") not in MURMUR_NAMES:
            continue
        def clean(v):
            if v in ("", "nan", "None"):
                return None
            return v
        meta[pid] = {
            "age": clean(row.get("Age")),
            "sex": SEX_CN.get(clean(row.get("Sex")), clean(row.get("Sex"))),
            "pregnant": clean(row.get("Pregnancy status")),
            "murmur": row["Murmur"],
            "outcome": clean(row.get("Outcome")),
            "most_audible": clean(row.get("Most audible location")),
            "systolic_timing": clean(row.get("Systolic murmur timing")),
            "systolic_shape": clean(row.get("Systolic murmur shape")),
            "systolic_grading": clean(row.get("Systolic murmur grading")),
            "systolic_pitch": clean(row.get("Systolic murmur pitch")),
            "systolic_quality": clean(row.get("Systolic murmur quality")),
            "diastolic_timing": clean(row.get("Diastolic murmur timing")),
            "diastolic_grading": clean(row.get("Diastolic murmur grading")),
            "diastolic_pitch": clean(row.get("Diastolic murmur pitch")),
            "diastolic_quality": clean(row.get("Diastolic murmur quality")),
        }

    buckets = {}
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            m = re.match(r"(\d+)_(AV|PV|TV|MV)(?:_\d+)?\.wav", f)
            if not m:
                continue
            pid, loc = m.group(1), m.group(2)
            if pid not in meta:
                continue
            path = os.path.join(root, f)
            with sf.SoundFile(path) as s:
                dur = len(s) / s.samplerate
            buckets.setdefault((loc, meta[pid]["murmur"]), []).append((pid, f, path, dur))

    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest = []
    seen = set()
    for (loc, mur), items in sorted(buckets.items()):
        items = [it for it in items if it[3] >= MIN_SECONDS]
        items.sort(key=lambda it: it[3], reverse=True)
        for pid, f, path, dur in items[:MAX_PER_BUCKET]:
            if pid in seen:
                continue
            seen.add(pid)
            dest = f"{pid}_{loc}.wav"
            shutil.copy2(path, os.path.join(clips_dir, dest))
            info = meta[pid]
            entry = {
                "id": f"{pid}_{loc}", "file": dest, "patient_id": pid,
                "age": info["age"], "sex": info["sex"],
                "murmur": mur, "outcome": info["outcome"],
                "location": loc, "duration_s": round(dur, 1),
            }
            if mur == "Present":
                entry["murmur_detail"] = {
                    "most_audible": info["most_audible"],
                    "systolic_timing": info["systolic_timing"],
                    "systolic_shape": info["systolic_shape"],
                    "systolic_grading": info["systolic_grading"],
                    "systolic_pitch": info["systolic_pitch"],
                    "systolic_quality": info["systolic_quality"],
                    "diastolic_timing": info["diastolic_timing"],
                    "diastolic_grading": info["diastolic_grading"],
                }
            manifest.append(entry)

    manifest.sort(key=lambda e: (e["location"], e["murmur"]))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"source": "CirCor DigiScope 1.0.3 (PhysioNet 2022)",
                   "note": "Expert-annotated real PCG recordings; patient metadata from training_data.csv.",
                   "clips": manifest}, fh, ensure_ascii=False, indent=1)
    counts = {}
    for e in manifest:
        counts.setdefault((e["location"], e["murmur"]), 0)
        counts[(e["location"], e["murmur"])] += 1
    print(f"[assets] {len(manifest)} clips -> {clips_dir}")
    for k in sorted(counts):
        print(f"  {k[0]:>2} {k[1]:<8} {counts[k]}")
    total_mb = sum(os.path.getsize(os.path.join(clips_dir, e["file"])) for e in manifest) / 1e6
    print(f"[assets] total {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
