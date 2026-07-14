#!/usr/bin/env python3
"""Held-out (138) prospective eval (paper Fig 3) for one run (Run B or Run A).

Reuses the run's saved per-seed TRAIN embeddings (voicefm_whisper_seed{S}_embeddings.npz,
846 participants) and extracts the 138 held-out participants' embeddings from that run's
checkpoints, USING THE TRAINING TASK MAP (811 tasks; the 16 held-out-only task types ->
id 0 'unknown', which is correct for unseen tasks and avoids the task_embedding overflow).
Then trains LR probes on the 846 train, evaluates on the 138 held-out (paper protocol).

Usage: python run_heldout138.py --run-dir <VoiceFM-public|VoiceFM-paper-faithful> --tag RUNB|RUNA
Output: <run-dir>/results_v3/prospective_test_probes_<tag>.json
"""
import argparse, os, sys, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44, 45, 46]
CATS = ["gsd_control", "cat_voice", "cat_neuro", "cat_mood", "cat_respiratory"]
DIAGS = ["gsd_parkinsons", "gsd_alz_dementia_mci", "gsd_depression", "gsd_airway_stenosis",
         "gsd_laryngeal_dystonia", "gsd_mtd", "gsd_anxiety", "gsd_benign_lesion", "gsd_copd_asthma"]
BATCH = 16

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
RUN = args.run_dir
sys.path.insert(0, RUN)  # use THIS run's code (patched for Run A)
from scripts.unified_gsd_probes_v3 import load_whisper_encoder
from src.data.audio_dataset import build_task_type_map
from src.utils.preprocessing import load_and_preprocess, MAX_SAMPLES

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processed_v300 = full 984-cohort export; it (unlike processed_v3/v3eval) contains the
# 138 held-out participants' 5824 recordings. Training task-map is still built ONLY from
# the train-audio-available recs below (-> 811), matching the trained task_embedding.
rec = pd.read_parquet(f"{RUN}/data/processed_v300/recordings.parquet")
part = pd.read_parquet(f"{RUN}/data/processed_v300/participants.parquet")
train = part[part["cohort_split"] == "train"]
test = part[part["cohort_split"] == "test"]
TRAIN_AUDIO = f"{RUN}/data/audio"                 # -> audio_v300/train (846)
HELD_AUDIO = f"{RUN}/data/audio_v300/test"        # 138 held-out wavs

# TRAINING task map (811) — the encoder's task_embedding was sized from this.
train_avail = rec[rec["recording_id"].astype(str).isin(
    {f[:-4] for f in os.listdir(TRAIN_AUDIO) if f.endswith(".wav")})]
TASK_MAP = build_task_type_map(train_avail)
print(f"[{args.tag}] training task_map={len(TASK_MAP)} (maxid {max(TASK_MAP.values())})", flush=True)

held_ids = set(test.index)
held_rec = rec[rec["record_id"].isin(held_ids)]

@torch.no_grad()
def extract_heldout(encoder):
    items = [(r["record_id"], r.get("recording_name", ""), os.path.join(HELD_AUDIO, r["recording_id"] + ".wav"))
             for _, r in held_rec.iterrows() if os.path.exists(os.path.join(HELD_AUDIO, r["recording_id"] + ".wav"))]
    pid_embs = {}
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        wavs, pids, tids = [], [], []
        for pid, tname, path in batch:
            w = load_and_preprocess(path, max_samples=MAX_SAMPLES)
            w = w.numpy() if isinstance(w, torch.Tensor) else w
            if len(w) < 400:
                continue
            wavs.append(w); pids.append(pid); tids.append(TASK_MAP.get(tname, 0))  # unseen task -> 0
        if not wavs:
            continue
        ml = max(len(w) for w in wavs)
        pad = np.zeros((len(wavs), ml), np.float32); msk = np.zeros((len(wavs), ml), np.int64)
        for j, w in enumerate(wavs):
            pad[j, :len(w)] = w; msk[j, :len(w)] = 1
        embs = encoder(torch.tensor(pad, device=dev), torch.tensor(msk, device=dev),
                       torch.tensor(tids, dtype=torch.long, device=dev)).cpu().numpy()
        for pid, e in zip(pids, embs):
            pid_embs.setdefault(pid, []).append(e)
    return {p: np.mean(v, axis=0) for p, v in pid_embs.items()}

def run_probe(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2 or ytr.sum() < 2 or yte.sum() < 2:
        return float("nan")
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

results = {}
for seed in SEEDS:
    enc = load_whisper_encoder(seed, dev)
    ho = extract_heldout(enc)
    del enc; torch.cuda.empty_cache()
    tr_npz = np.load(f"{RUN}/results_v3/voicefm_whisper_seed{seed}_embeddings.npz", allow_pickle=True)
    trn = {str(p): e for p, e in zip(tr_npz["pids"], tr_npz["embeddings"])}
    pid = {**trn, **{str(k): v for k, v in ho.items()}}
    tr_ids = [p for p in train.index if str(p) in pid]
    te_ids = [p for p in test.index if str(p) in pid]
    Xtr = np.array([pid[str(p)] for p in tr_ids]); Xte = np.array([pid[str(p)] for p in te_ids])
    print(f"[{args.tag}] seed {seed}: train {len(tr_ids)} / held-out {len(te_ids)}", flush=True)
    for lab in CATS + DIAGS:
        if lab not in train.columns:
            continue
        a = run_probe(Xtr, train.loc[tr_ids, lab].values.astype(int),
                      Xte, test.loc[te_ids, lab].values.astype(int))
        if not np.isnan(a):
            results.setdefault(f"voicefm_whisper/{lab}", []).append(a)

out = f"{RUN}/results_v3/prospective_test_probes_{args.tag}.json"
json.dump(results, open(out, "w"), indent=2)
mean = np.mean([np.mean(results[f"voicefm_whisper/{c}"]) for c in CATS if f"voicefm_whisper/{c}" in results])
print(f"[{args.tag}] held-out category mean AUROC = {mean:.4f} -> {out}", flush=True)
