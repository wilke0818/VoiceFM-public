#!/usr/bin/env python3
"""Frozen-Whisper baseline on the 138 held-out (paper Fig-3 protocol), as an ADDED reference.

The paper's committed prospective_test_probes.json has NO frozen baseline; this computes one:
does contrastive VoiceFM training beat raw frozen Whisper on genuinely-unseen participants?

Frozen Whisper = openai/whisper-large-v2 encoder (mean-pool -> 1280d, no task embedding, no
projection) -> run-independent (single result, not per-run/per-seed). Reuses the saved
846-train frozen recording embeddings (results_v3/whisper_recording_embeddings.npz) and
extracts the 138 held-out recs fresh with the SAME encoder/pooling. Probe = authors' run_probe
(StandardScaler fit on 846 train, LR C=1.0). Output: results_v3/prospective_test_probes_FROZEN.json
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", required=True,
                 help="path to the VoiceFM-public checkout holding data/ + results_v3/")
RUN = _ap.parse_args().run_dir
sys.path.insert(0, RUN)
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES
from transformers import WhisperModel, WhisperFeatureExtractor

CATS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
DIAGS = ["gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_depression", "gsd_airway_stenosis",
         "gsd_laryngeal_dystonia", "gsd_mtd", "gsd_anxiety", "gsd_benign_lesion", "gsd_copd_asthma"]
BATCH = 16
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rec = pd.read_parquet(f"{RUN}/data/processed_v300/recordings.parquet")
part = pd.read_parquet(f"{RUN}/data/processed_v300/participants.parquet")
train = part[part["cohort_split"] == "train"]
test = part[part["cohort_split"] == "test"]
HELD_AUDIO = f"{RUN}/data/audio_v300/test"

# --- frozen encoder (identical to whisper_extract_embeddings_v3.py) ---
model = WhisperModel.from_pretrained("openai/whisper-large-v2", torch_dtype=torch.float32)
encoder = model.encoder.float().to(dev).eval()
for p in encoder.parameters():
    p.requires_grad = False
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

held_rec = rec[rec["record_id"].isin(set(test.index))]
items = [(r["record_id"], os.path.join(HELD_AUDIO, r["recording_id"] + ".wav"))
         for _, r in held_rec.iterrows() if os.path.exists(os.path.join(HELD_AUDIO, r["recording_id"] + ".wav"))]
print(f"[FROZEN] held-out recs with audio: {len(items)}", flush=True)

ho_pid = {}
with torch.no_grad():
    for i in range(0, len(items), BATCH):
        wavs, pids, lengths = [], [], []
        for pid, path in items[i:i + BATCH]:
            try:
                w = load_and_preprocess(str(path), max_samples=MAX_SAMPLES)
            except Exception:
                continue
            w = w.numpy() if isinstance(w, torch.Tensor) else w
            if len(w) < 400:
                continue
            wavs.append(w); pids.append(pid); lengths.append(len(w))
        if not wavs:
            continue
        mel = torch.stack([fe(w, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0) for w in wavs]).to(dev)
        hidden = encoder(mel.float(), return_dict=True).last_hidden_state
        for j in range(len(wavs)):
            tl = min(max(1, int(lengths[j] / MAX_SAMPLES * hidden.shape[1])), hidden.shape[1])
            ho_pid.setdefault(pids[j], []).append(hidden[j, :tl, :].mean(dim=0).cpu().numpy())
        if i % 1600 == 0:
            print(f"  {i}/{len(items)}", flush=True)
ho_mean = {p: np.mean(v, axis=0) for p, v in ho_pid.items()}

# --- saved 846-train frozen recording embeddings -> participant-mean (load_frozen_whisper_embeddings) ---
z = np.load(f"{RUN}/results_v3/whisper_recording_embeddings.npz", allow_pickle=True)
tr_pid = {}
for idx, pid in enumerate(z["participant_ids"]):
    tr_pid.setdefault(str(pid), []).append(idx)
tr_mean = {pid: z["embeddings"][idxs].mean(axis=0) for pid, idxs in tr_pid.items()}
pid = {**{str(k): v for k, v in tr_mean.items()}, **{str(k): v for k, v in ho_mean.items()}}
print(f"[FROZEN] train pids={len(tr_mean)}  held-out pids={len(ho_mean)}  dim={len(next(iter(pid.values())))}", flush=True)

def run_probe(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or ytr.sum() < 2 or yte.sum() < 2:
        return float("nan")
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

tr_ids = [p for p in train.index if str(p) in pid]
te_ids = [p for p in test.index if str(p) in pid]
Xtr = np.array([pid[str(p)] for p in tr_ids]); Xte = np.array([pid[str(p)] for p in te_ids])
print(f"[FROZEN] probe: train {len(tr_ids)} / held-out {len(te_ids)}", flush=True)
# deterministic: frozen embeddings + fixed 846 train + LR rs=42 -> single value per label (n_seed=1)
results = {}
for lab in CATS + DIAGS:
    if lab not in train.columns:
        continue
    a = run_probe(Xtr, train.loc[tr_ids, lab].values.astype(int), Xte, test.loc[te_ids, lab].values.astype(int))
    if not np.isnan(a):
        results[f"frozen_whisper/{lab}"] = [a]

out = f"{RUN}/results_v3/prospective_test_probes_FROZEN.json"
json.dump(results, open(out, "w"), indent=2)
cm = np.mean([results[f"frozen_whisper/{c}"][0] for c in CATS if f"frozen_whisper/{c}" in results])
print(f"[FROZEN] held-out category mean AUROC = {cm:.4f} -> {out}", flush=True)
