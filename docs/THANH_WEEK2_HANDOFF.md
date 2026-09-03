# Thành - Week 2 Baseline, Evaluator and Cold-Start Handoff

## Completed and verified

- Shared `Predictor` contract for Most Popular, MF, GMF and NCF checkpoints.
- Train-only K=20 nearest-neighbour embedding fold-in for new users.
- Shared Top-K evaluator using the existing candidate and ranking modules.
- Three baseline seeds: 42, 43 and 44, with per-checkpoint manifests.
- RMSE/MAE mean ± standard deviation.
- Recall@10/NDCG@10/HitRate@10/Coverage mean ± standard deviation.
- Cold-start evaluation for profiles with 5 and 10 ratings.
- CLI that accepts original MovieLens IDs and produces a new-user Top 10.
- Complete test suite: 233 tests passed after data preparation and the final
  configuration/documentation additions.

Data version:

```text
9241a56b2e544fb8ab460ea009f0a3c4edc9a76d502c58bce50a09becb29f310
```

Config hash used for the recorded three-seed baseline run:

```text
f3649204703693fa16e1606d4755d401d63727443a55df54ac444e16033885bf
```

## Evidence summary

| Model | Test RMSE | Recall@10 | NDCG@10 | HitRate@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Most Popular | 1.1115 ± 0.0000 | 0.2649 ± 0.0232 | 0.2357 ± 0.0162 | 0.5841 ± 0.0225 | 0.1793 ± 0.0087 |
| MF | 1.0270 ± 0.0042 | 0.2829 ± 0.0143 | 0.2486 ± 0.0119 | 0.6423 ± 0.0188 | 0.5351 ± 0.0131 |
| GMF | 1.0282 ± 0.0011 | 0.2927 ± 0.0150 | 0.2580 ± 0.0117 | 0.6447 ± 0.0176 | 0.5862 ± 0.0261 |

All rows use 836 evaluated users and report 107 users skipped because their
test partition has no rating >= 4. MF seed 42 scored a 1,677-item new-user
catalog in about 0.04 seconds on the local CPU run; latency must be measured
again on the official demo machine.

Two five-rating demo profiles produced different Top 10 lists. The
Action/Sci-Fi profile ranked `Raiders of the Lost Ark` first; the
Romance/Drama profile ranked `The Shawshank Redemption` first. Seven of ten
positions differed. This is integration evidence, not an offline quality
metric.

## Commands

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train_baseline.py --device cpu
python scripts/evaluate_topk.py --models most_popular mf gmf
python scripts/evaluate_cold_start.py
python -m pytest -q
```

Example demo profile:

```bash
python scripts/demo_new_user.py \
  --rating 39=5 --rating 50=5 --rating 62=5 --rating 82=5 --rating 96=5
```

## Required reviews before integration

### Sơn - protocol/evaluator review

Please confirm positive threshold 4.0, seen=train+validation, 100 negatives,
K=10, skipped-user accounting and the use of the existing metric functions.
This review is required because Thành owns the integration script while Sơn
owns metric semantics; independent definitions would invalidate comparisons.

### Chúc - Backend adapter review

Please confirm MovieLens-ID to model-index conversion, predictor error to HTTP
status mapping, manifest loading and the declared MF fallback. This prevents
database IDs from being passed into embeddings and prevents silent fallback
to an arbitrary user.

### Công Thành - NCF handoff

Please save real MovieLens NCF checkpoints with `model_config=model.config()`
and run:

```bash
python scripts/evaluate_topk.py --models ncf \
  --seeds 42 --checkpoint ncf=PATH_TO_REAL_NCF_CHECKPOINT
```

The NCF row is blocked only on a real checkpoint. Random/smoke output must not
be reported. This preserves a fair comparison and prevents fabricated RQ1
evidence.

### Hải Anh - data/mapping confirmation

Please confirm the data version and frozen ID mapping. A mapping mismatch can
return a valid numeric score for the wrong movie, so this check is mandatory.

### Hoàng Anh - release evidence

Please validate the baseline manifests against the shared manifest schema and
rerun checkpoint round-trip, full tests and clean-clone commands. Manifest
validation is needed because the final common schema is owned by testing and
must not fork into an AI-only format.

## Honest limitations

- No real NCF row is reported because the repository still contains only the
  week-one random-tensor smoke trainer.
- The cold-start result with 10 ratings was worse than with 5; it is retained
  rather than selected away.
- Three Top-K runs use a matching candidate seed per model run, so their
  variation includes both model initialization and negative sampling.
- Generated checkpoints and result JSON are ignored by Git; the documented
  summaries and reproduction commands are the committed evidence.
