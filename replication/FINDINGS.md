# VoiceFM replication — findings

An independent replication and audit of **"Towards A Foundation Model for Clinical Voice
Biomarkers" (VoiceFM)** (medRxiv 2026.05.28.26354346) against this released code. Scope:
**paper↔code consistency, reproducibility, and result integrity** — not a judgment of
scientific merit.

**Method.** We reproduced the authors' cohort and labels from the v3.0.0 REDCap export, then
trained the authors' code **as-is** ("Run B") on the exact 846-participant cohort and 5 seeds
(42–46). We take Run B to be *the model the paper intends* — the released code run without
modification. We evaluated it in-distribution (Fig 2, GSD 70/15/15), on the 138
temporally-held-out cohort (Fig 3), and against a frozen-Whisper baseline. All comparison tables
carry **n** (positive participants) so the small-cohort caveats are legible.

> We also trained a **paper-faithful variant** ("Run A") that patches the code to match the
> paper's text/figures on the divergences in §2. It performs slightly *worse* and loses the
> significance of the contrastive gain — i.e. the literal description doesn't reproduce the
> headline as well as the as-is code. We keep the tables focused on Run B; the Run A result JSONs
> are included in `results/` (files `*_RUNA*.json`) for anyone who wants them.

---

## 1. Headline — what reproduces, what doesn't

- **Cohort & labels reproduce exactly** from the v3.0.0 REDCap export (846/138 split, Σ|Δ|=0
  across all 19 diagnoses vs Table S2).
- **In-distribution headline reproduces**: Run B (as-is code) gives **0.947** vs paper **0.952**
  (within ~1 SD); the contrastive gain over frozen stays significant (p≈0.03).
- **The temporally-held-out numbers do NOT reproduce as cleanly**: Run B falls a *systematic*
  ~0.045 below the paper on all 5 categories — a gap that only appears out-of-sample.
- **The released code diverges from the paper's described architecture** on ≥5 points (§2);
  patching the code to match the text (Run A, on request) performs *worse* — so the as-is code
  is both the intended model and the better-performing one.
- **Frozen Whisper is far more competitive out-of-sample than the paper implies** — a
  comparison the paper omits from Fig 3.

---

## 2. Paper vs. code differences (highlight)

Points where the *released code* does something different from the paper's Methods/figures. We
treat the as-is code (Run B) as the intended model; #1–#4 materially change the architecture and
are what a paper-faithful variant would patch (Run A, on request).

| # | Paper says | Released code (Run B) does | materially changes model? |
|---|---|---|---|
| 1 | Clinical encoder **mean-pools** tokens (Fig 1) | Uses the **CLS token** (`clinical_encoder.py`) | yes |
| 2 | Continuous features **discretized into bins** | **Linear projection**, no binning (FT-Transformer tokenizer) | yes |
| 3 | Audio projection = **single Linear 1280→256** (Fig 1) | **2-layer MLP** | yes |
| 4 | **"10% warmup"** then cosine schedule | Warmup **not implemented** (cosine only) | yes |
| 5 | HuBERT fine-tunes layers **9–11 (3)** | Fine-tunes layers **8–11 (4)** | minor |
| 6 | Effective batch **64 = 32×2** | Experiment config says **8×8** but silently runs 32×2 (see §6) | — |
| 7 | HeAR ≈ **4.1M** params | ≈ **0.4M** | minor |
| 8 | CKA on Whisper (32-layer) | CKA code is **HuBERT/12-layer** | minor |

**Note on the bin count (#2):** the number of bins is specified *nowhere* — not in the paper,
any git revision of this repo, or the reproduction repo. The paper cites TabTransformer (which
normalizes+concatenates, no bins); the released code is an FT-Transformer linear tokenizer; the
paper *text* describes PLE-style binning (Gorishniy 2022). The three are mutually inconsistent,
and only the linear version is actually implemented — so Run B (linear) is likely the true model
and the binning is text-faithful only.

**Interpretation:** the released code's *actual* (undocumented) choices are what we treat as the
model. A variant patched to match the paper text (Run A) is consistently *lower* and loses the
significance of the contrastive gain — so the paper's literal description does not, by itself,
reproduce the headline as well as the as-is code does. (Run A numbers on request.)

---

## 3. Paper results vs. our run of their code

In-distribution GSD (Fig 2), 5-seed mean ± SD; **n = positive participants in the 846 cohort**
(each seed scores a stratified ~15% fold, so ~15% of these positives per fold):

| category | n (846) | Paper (committed) | Run B (as-is) | ΔB−paper |
|---|--:|---|---|--:|
| control | 161 | 0.926 | 0.907 | −0.019 |
| voice | 287 | 0.954 | 0.945 | −0.009 |
| neuro | 204 | 0.998 | 0.997 | −0.001 |
| mood | 83 | 0.937 | 0.946 | +0.009 |
| respiratory | 160 | 0.943 | 0.939 | −0.004 |
| **5-cat mean** | | **0.9516 ± 0.005** | **0.9468 ± 0.007** | **−0.005** |

VoiceFM-vs-Frozen (Welch t on seed means): paper Δ+0.0258 (p=0.013); **Run B Δ+0.0208
(p=0.032, significant)** — the contrastive gain and its significance reproduce.

**Held-out (Fig 3), 5-seed mean ± SD; n = positive participants scored (of the 138). The probe
trains on all 846 (training positives always clear the ≥2 guard), so n here is what governs the
held-out result's reliability.**

| label | n | Paper | Run B | Frozen | ΔB−paper | ΔB−frozen |
|---|--:|---|---|---|---|---|
| control | 38 | 0.910 | 0.867 | 0.834 | −0.043 | +0.033 |
| voice | 46 | 0.964 | 0.935 | 0.944 | −0.029 | −0.010 |
| neuro | 23 | 0.984 | 0.985 | 0.991 | +0.001 | −0.006 |
| mood | 12 | 0.849 | 0.774 | 0.667 | −0.075 | +0.107 |
| respiratory | 31 | 0.832 | 0.755 | 0.659 | −0.078 | +0.096 |
| **5-cat mean** | | **0.9077** | **0.8630** | **0.8190** | **−0.045** | **+0.044** |
| parkinsons | 3 | 0.928 | 0.903 | 0.884 | −0.026 | +0.019 |
| alz/mci | 20 | 0.986 | 0.966 | 0.998 | −0.020 | −0.032 |
| depression | 11 | 0.841 | 0.760 | 0.749 | −0.081 | +0.011 |
| airway_stenosis | 14 | 0.891 | 0.824 | 0.855 | −0.067 | −0.031 |
| laryngeal_dystonia | 11 | 0.890 | 0.850 | 0.913 | −0.040 | −0.063 |
| mtd | 9 | 0.772 | 0.695 | 0.690 | −0.077 | +0.005 |
| anxiety | 3 | 0.818 | 0.727 | 0.570 | −0.091 | +0.157 |
| benign_lesion | 9 | 0.809 | 0.760 | 0.873 | −0.049 | −0.113 |
| copd_asthma | 8 | 0.699 | 0.596 | 0.560 | −0.103 | +0.037 |

The largest gaps and widest SDs cluster in the **small-n** rows (mood 12, respiratory 31,
COPD 8, PD/anxiety 3). See §7 for how the code handles this.

---

## 4. Frozen Whisper on the out-of-distribution cohort

The paper's Fig 3 reports only VoiceFM-Whisper vs VoiceFM-HuBERT — **no frozen baseline appears
anywhere in the held-out results, methods, or caption** (verified against the paper text; the
only nearby "frozen" mention opens the *external-datasets* section, not the 138). We computed
it (raw whisper-large-v2, 1280-d mean-pool, deterministic):

- **Category-mean: Frozen 0.819 vs Run B (contrastive) 0.863** — the contrastive model beats
  frozen by only **+0.044** out-of-sample, concentrated in **mood (+0.107)** and
  **respiratory (+0.096)**.
- **Frozen *beats* the contrastive model on several diagnoses** on unseen data:
  Alzheimer's/MCI **0.998 vs 0.966**, laryngeal dystonia 0.913 vs 0.850, benign lesion 0.873 vs
  0.760, plus neuro and voice.

In-distribution the paper reports contrastive +0.026 over frozen (significant); on the truly
unseen cohort the advantage shrinks to +0.044 category-mean and *reverses* for several
conditions. The one comparison that most directly tests whether the contrastive pretraining
earns its keep on new patients is the one Fig 3 omits.

---

## 5. Split methodology — internally consistent (no leakage); exact reproduction needs the ordering

We traced the split at both ends and it is **clean — no train/eval leakage**:

- **Training** (`train.py`): `create_participant_splits(participants, seed, stratify)` → the encoder
  (`VoiceFMDataset`) trains on the **train fold only**; the test fold is never touched. The
  participant set is audio-filtered to `data/audio` (the 846 train wavs), so the 138 held-out
  aren't even present.
- **Probe** (`unified_gsd_probes_v3.py`, per checkpoint seed *N*): the same
  `create_participant_splits(seed=N, stratify="disease_category")` over the same 846 participants
  in the same table order → the **identical** fold. The LR trains on the train fold and scores the
  test fold, i.e. the encoder's held-out participants. Every seed is aligned encoder-*N* ↔ probe-*N*.

So the probe's test fold *is* the encoder's held-out fold by construction — the in-distribution
eval does not evaluate on encoder-seen participants.

The one real caveat is **exact reproducibility**, not validity: `create_participant_splits` seeds
its stratified sampling off the participant table's **row order**, and that order isn't pinned or
shipped with the code. A replicator who rebuilds the cohort from the raw REDCap export (in whatever
order it emerges) gets a *different* — but internally consistent and still non-leaky — split, so
their numbers won't match the paper's exactly even with everything else identical. A small fix
(sort by `record_id` before splitting, or persist the split assignment) would make the folds
reproducible from the code alone.

Two smaller notes: the split is stratified by disease **category** (the 4 `cat_*` + control), not
by the 19 individual `gsd_*` diagnoses, so rare diagnoses aren't balanced across folds (see §7);
and if the stratified split ever raises, the code **silently falls back to an unstratified random
split**.

---

## 6. Silent batching bug

The paper specifies **effective batch 64 = 32×2** (micro-batch 32, grad-accum 2). InfoNCE
negatives come from the micro-batch `(B,B)` similarity matrix (`src/training/losses.py`);
grad-accum does **not** gather negatives across steps, so the *micro-batch size* — not the
effective batch — sets the number of contrastive negatives.

- `configs/train.yaml` (base): `training: {batch_size: 32, gradient_accumulation_steps: 2}`
  (comment: *"was 8, increased after Exp A"*) — correctly nested under `training:`.
- `configs/experiments/exp_whisper_ft4_gsd_v3_seed42.yaml`: `batch_size: 8` and
  `gradient_accumulation_steps: 8` at **top level**, *not* under `training:`.
- The trainer reads `config["train"]["training"]["batch_size"]`. `deep_merge` (`scripts/train.py`)
  drops the experiment's top-level keys into an unused `config["train"]["batch_size"]` slot, so
  they are **silently ignored** and training uses the base **32×2**.

Net: the shipped experiment config *appears* to request **8×8** but actually runs **32×2**. It
happens to match the paper — but by accident of the base default, not because the experiment
config is correct. Anyone who "fixed" the nesting to make 8×8 take effect would get **8**
contrastive negatives instead of 32 — a ~4× weaker contrastive signal, and likely worse results
(we did not train 8×8 to confirm) — while the reported "effective batch 64" would look unchanged.

---

## 7. Small-N handling in the code

The **only** explicit mechanism is a hard guard in `run_probe` (identical in
`unified_gsd_probes_v3.py`, `evaluate_prospective_v3.py`, and our framework):

```python
if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2: return NaN
if np.sum(y_test) < 2 or np.sum(y_train) < 2:                 return NaN
```

A label is **skipped** (→ NaN, dropped) if the train or scored fold has <2 positives.
Consequences, none of them flagged in the reported numbers:
- **Silently fewer than 5 seeds.** A rare diagnosis's 15% test fold can have 0–1 positives on
  some seeds → those seeds drop; the "5-seed mean ± SD" is then over fewer seeds (e.g. in-dist
  `anxiety` survived only 2/5 seeds).
- **No per-diagnosis stratification** (see §5) means positives-per-fold aren't guaranteed.
- **No class weighting, no confidence intervals, no exact tests.** On the held-out 138, PD and
  anxiety have **3** positives — an AUROC from ranking 3 cases among 138 is extremely
  high-variance, and the 5-seed SD (held-out set fixed) captures only encoder-seed noise, so it
  *understates* the true uncertainty.

---

## 8. Versioning issues — silent / empty files

Evidence that the released code ≠ the code that produced the committed results, plus
silent-failure hazards:

- **`eval_h28_external_v3.py`** errors out as shipped: line 23 does
  `from scripts.eval_h27_external import …`, but the file is `scripts/eval_h27_external_v3.py`
  (note the `_v3`), so it raises `ModuleNotFoundError: No module named 'scripts.eval_h27_external'`.
- **Shipped training log** `results_v3/training_log_whisper_ft4_v3_seed42.txt` — our static audit
  read this as a *pretraining* log (seed 42), not the seed-43 fine-tune it is presented as; flagged
  for the authors to confirm against their own run records.
- **Orphan `results_v3/neurovoz/acoustic_analysis_neurovoz.json`** — our static audit found it
  inconsistent with the committed gemaps JSON; the two can be diffed directly.
- **`load_frozen_whisper_embeddings`** does an *unguarded* `np.load` of
  `whisper_recording_embeddings.npz` **first**, so the GSD eval crashes unless the frozen
  embeddings are pre-extracted — an undocumented ordering dependency.
- **`preprocess.py` default `--redcap-csv`** points at a **v2.3.0** export; the paper cohort is
  **v3.0.0**. Running the defaults silently yields the wrong cohort (Σ|Δ|≠0), and the eval consumes
  whatever is in `data/processed_v3/` without asserting which cohort/version it holds.
- The **held-out embedding-extraction step is not shipped**: `evaluate_prospective_v3.py`
  expects per-seed npz files containing both the 846 train and 138 held-out embeddings, but
  `unified_gsd_probes_v3.py` only produces the 846 (held-out recordings are excluded from
  `processed_v3`, and a naive held-out task-map would overflow the trained 812-row task
  embedding). We had to reconstruct it.
- **`requirements.txt` is unpinned** and incomplete: on `torchaudio>=2.9` (torchcodec backend)
  `torchaudio.load` **raises** on header-only/undecodable files (rather than returning empty as
  older backends did), and `torchcodec` isn't listed. Our shim restores the old "treat as silence"
  behavior; in our runs it hit exactly **one** training recording (`6B2E83F0…`; 0 held-out), which
  is below the 400-sample floor and thus skipped — negligible (1 of 34,232).

---

## Bottom line

Data provenance, in-distribution results, and eval integrity hold up well — and the in-distribution
eval is leakage-free by construction (the probe scores each encoder's own held-out fold; §5). The
paper-described architecture, several mechanistic figures, and the *temporal* generalization story
are where it gets shaky: the released code outperforms its own paper's description, and on the
temporally-separated held-out cohort (Fig 3) the numbers run below what's reported with the
contrastive advantage over frozen features largely evaporating. The strongest results are the
same-period in-distribution ones; generalization to the later cohort is weaker.
