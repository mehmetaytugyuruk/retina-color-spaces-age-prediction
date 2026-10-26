# Paper-to-code mapping

Reference for: M. A. Yürük and A. Memiş, "Exploring the Impact of Alternative Color
Spaces in Deep Retinal Age Prediction from Fundus Images," ATEEC 2026 (under review).

## Naming

The paper and the code use different identifiers for the three combination strategies:

| Paper name | Identifier in code and CSV outputs |
| --- | --- |
| Ensemble 1 (RGB + Lab + HSV + YCrCb, seed 42) | `color4_equal` |
| Ensemble 2 (RGB seeds 43–46) | `rgb4_equal` |
| Fusion Model (12-channel early fusion) | `fusion/rgb_lab_hsv_ycrcb_seed42` |

## Sections

| Paper section | Code |
|---|---|
| II-A. Retinal Image Dataset | `data/manifests/{train,validation,test}.csv` — patient-level split, no patient appears in more than one subset |
| II-A. Table I (age-category distribution) | Derived directly from the `age` column of the manifests; not emitted by a script |
| II-B. Image Preprocessing, Fig. 1 | `src/retinal_color_transfer/preprocessing/crop_pad_resize.py` — `crop_pad_resize_bgr()` (threshold-based bounding box, aspect-preserving resize, zero-pad to 224×224) |
| II-B. Color-space conversion | `src/retinal_color_transfer/representations/converters.py` — `convert_representation()`; single channels are replicated three times by `_repeat_channel()` |
| II-B. Representation contracts | `configs/{rgb,lab,hsv,ycrcb,grayscale}/*.yaml` — lock channel order, dtype, numeric range, cache format, tensor scaling |
| II-B. Fig. 2 (pseudo-color and per-channel visualization) | `scripts/generate_figure_assets.py` |
| II-C. Deep Learning Model | `src/retinal_color_transfer/model.py` — `build_resnet50_regressor()`; `regression_head()` replaces `fc` |
| II-C. Fusion Model 12-channel `conv1` adaptation | `src/retinal_color_transfer/model.py` — `build_resnet50_12ch_regressor()` (pretrained `conv1` weights repeated 4× and scaled by 1/4) |
| II-D. Eq. 1 (MAE) | `src/retinal_color_transfer/evaluation.py` — `regression_metrics()` |
| II-D. Eq. 2 (ensemble prediction averaging) | `src/retinal_color_transfer/workflows/results.py` — `_add_derived_predictions()`, columns `pred_color4_equal` and `pred_rgb4_equal` |
| II-D. Paired patient-level cluster bootstrap (100,000 repetitions) | `src/retinal_color_transfer/workflows/results.py` — `_paired_bootstrap()` |
| III-A. Table II (configurations and seeds) | `scripts/01_prepare_training.py` writes the 21 canonical configs; the seed list lives in `notebooks/train_all_models.ipynb` (`SEED42_REPRESENTATIONS`, `EXTRA_RGB_SEEDS`) |
| III-A. Fig. 3 (framework overview) | Diagram of the whole pipeline; drawn manually, no generating script |
| III-A. Training settings (AdamW, LR 1e-4, WD 1e-4, 80 epochs, batch 32, ReduceLROnPlateau) | `configs/resnet50_imagenet_local.yaml` and `src/retinal_color_transfer/training/engine.py` — `run_training()` |
| III-A. LDS-weighted Smooth L1 loss (β = 1.0, σ = 2.0) | `src/retinal_color_transfer/training/objectives.py` — `lds_weights()`, `weighted_smooth_l1()` |
| III-A. Data augmentation (horizontal flip p = 0.5, rotation ±10°) | `src/retinal_color_transfer/training/transforms.py` — `TrainTransform` |
| III-B. Table III, "Overall MAE" column | `analysis/final_results/model_metrics.csv`, produced by `scripts/02_evaluate_and_analyze.py` |
| III-B. Table III, per-age-category columns | **Not produced by this repository.** Computed separately by grouping `models/**/predictions.csv` on the paper's five age bands |
| III-B. Bootstrap CI `[0.013, 0.277]`, Ensemble 1 vs Ensemble 2 | `analysis/final_results/paired_bootstrap_patient_cluster.csv` |

## Reproducing the reported numbers

Every per-model `predictions.csv` is tracked in this repository, so the metric tables can
be regenerated without the raw images and without a GPU:

```bash
PYTHONPATH=src python3 scripts/02_evaluate_and_analyze.py
```

Retraining from scratch additionally requires the source image dataset; see the README.

## Notes

- The bootstrap RNG seed is `20260720` and the validation split uses `seed + 100000`.
  These are fixed in `scripts/02_evaluate_and_analyze.py` defaults.
- `dummy_train_mean` in `model_metrics.csv` is a constant-prediction baseline (the train
  split mean age). It is not reported in the paper; it exists as a sanity floor.
- Training seeds make a run reproducible under the same library and hardware stack; they
  do not guarantee bit-identical weights to the released checkpoints, which were trained
  on an NVIDIA L4 GPU.
