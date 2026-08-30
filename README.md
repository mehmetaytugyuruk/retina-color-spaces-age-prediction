# Exploring the Impact of Alternative Color Spaces in Deep Retinal Age Prediction

> [!NOTE]
> **Retinal Age Prediction research series · Study 03**
>
> [Publications overview](https://mehmetaytugyuruk.github.io/publications/) · [Study 01: ResNet baselines](https://github.com/mehmetaytugyuruk/retina-resnet-age-prediction) · [Study 02: Vision Transformers](https://github.com/mehmetaytugyuruk/retina-vit-age-prediction)

**Resources:** [Pretrained weights](https://huggingface.co/mehmetaytugyuruk/retina-color-spaces-age-prediction) · [Paper-to-code mapping](docs/paper-to-code-mapping.md) · [Citation](#citation)

## Publication

> M. A. Yürük and A. Memiş, "Exploring the Impact of Alternative Color Spaces in Deep
> Retinal Age Prediction from Fundus Images," ATEEC 2026. **Under review.**

The DOI and final publication details will be added once the review process concludes.

## Overview

This repository contains the reproducible training and analysis workflow for a study of
color representations in retinal fundus age prediction. All models share the same
patient-disjoint data splits, ImageNet-pretrained ResNet-50 architecture, optimization
protocol, and checkpoint-selection rule. Only the input representation or the RGB
training seed changes.

The primary comparison asks whether an ensemble of *different color spaces* provides
more useful predictive diversity than an equal-size ensemble of *RGB models trained with
different seeds*.

Related studies on the same dataset:
[ResNet baseline study](https://github.com/mehmetaytugyuruk/retina-resnet-age-prediction) ·
[Vision Transformer study](https://github.com/mehmetaytugyuruk/retina-vit-age-prediction)

## Naming: paper ↔ code

The paper and the code use different identifiers for the same three combination
strategies. This table is authoritative:

| Paper name | Members | Identifier in code and CSV outputs |
| --- | --- | --- |
| **Ensemble 1** | RGB, Lab, HSV, YCrCb — all seed 42 | `color4_equal` |
| **Ensemble 2** | RGB seeds 43, 44, 45, 46 | `rgb4_equal` |
| **Fusion Model** | Single ResNet-50 on a 12-channel RGB+Lab+HSV+YCrCb input | `fusion/rgb_lab_hsv_ycrcb_seed42` |

Both ensembles average four age predictions, have the same inference cost, and share no
model. The comparison concerns these two fixed ensembles; it is not an estimate of
performance over a population of arbitrary training seeds.

## Study Design

The canonical study contains **21 models**:

- 17 representations trained with seed 42:
  - full `RGB`, `Lab`, `HSV`, and `YCrCb`
  - 12 single-channel ablations: `R`, `G`, `B`, `L`, `a`, `b`, `H`, `S`, `V`, `Y`, `Cr`, `Cb`
  - one grayscale structural control
- four additional full-RGB models trained with seeds 43, 44, 45, and 46

The **Fusion Model** is trained separately, bringing the total number of released
checkpoints to **22**.

## Dataset and Splits

The tracked manifests define patient-level train, validation, and test splits. No
patient, image ID, or resolved image path appears in more than one split.

| Split | Images | Patients | Age range | Mean age |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,902 | 3,775 | 5–97 | 57.42 |
| Validation | 1,493 | 809 | 8–92 | 57.77 |
| Test | 1,462 | 809 | 7–94 | 57.30 |
| Total | 9,857 | 5,393 | 5–97 | — |

Source dataset: [Retina Age Analysis Dataset](https://huggingface.co/datasets/ramankamran/retina-age-analysis)
(Kamran, 2025), MIT-licensed.

Raw retinal images are not redistributed here. To reproduce the study, provide the same
source dataset with paths matching the manifests:

```text
<DATA_ROOT>/
└── images/
    ├── img00001.jpg
    ├── img00002.jpg
    └── ...
```

## Training Protocol

All 21 canonical models use:

- ImageNet-pretrained ResNet-50, fully fine-tuned
- 80 epochs and batch size 32
- AdamW with learning rate `1e-4` and weight decay `1e-4`
- gradient-norm clipping at `1.0`
- label-distribution-smoothing weights with Smooth L1 training loss
- ReduceLROnPlateau scheduling
- horizontal flipping with probability `0.5` and rotation within ±10°
- representation-specific normalization computed from the train split only
- best-checkpoint selection by validation MAE

The shared training template is
[`configs/resnet50_imagenet_local.yaml`](configs/resnet50_imagenet_local.yaml).
Representation contracts under `configs/` lock channel order, stored dtype, numeric
range, cache format, and tensor scaling.

The Fusion Model follows the same protocol with one architectural difference: its first
convolutional layer accepts 12 channels (RGB + Lab + HSV + YCrCb). Pretrained `conv1`
weights are adapted by repeating them four times and scaling by `1/4` to preserve the
expected activation magnitude.

## Results

MAE is reported in years. Values below are the full-precision outputs in
`analysis/final_results/model_metrics.csv`; the paper rounds them to two decimals. The
paper additionally reports a per-age-category breakdown that is not produced by this
repository's analysis workflow.

### Combination strategies

| Configuration | Validation MAE | Test MAE |
| --- | ---: | ---: |
| **Ensemble 1** (RGB+Lab+HSV+YCrCb) | **4.9658** | **4.5983** |
| **Ensemble 2** (RGB seeds 43–46) | 5.1680 | 4.7436 |
| **Fusion Model** (12-channel early fusion) | 5.6764 | 5.2039 |

On the test split, Ensemble 1 reduces MAE by 0.1453 years — 3.06% relative to the
Ensemble 2 control.

The Fusion Model outperforms neither the single RGB model (test MAE 4.9067) nor
Ensemble 1. This is consistent with the observation that the four color spaces are
mathematically invertible transformations of each other, limiting the additional
information available to a single jointly-trained model. Late fusion via independent
prediction averaging remains the more effective combination strategy.

### Statistical uncertainty

Uncertainty is estimated with a paired non-parametric patient-level cluster bootstrap.
Because both eyes from the same patient can occur in the test set, images are not treated
as independent sampling units. Each of 100,000 repetitions samples 809 test-patient IDs
with replacement, includes all images belonging to every sampled patient, and uses the
same sampled patient clusters for both ensembles. MAE is then calculated over the pooled
images in the resampled clusters.

- observed test improvement: **0.145294 years**
- 95% bootstrap interval: **[0.013264, 0.277294] years**
- proportion of bootstrap improvements greater than zero: **98.417%**
- bootstrap RNG seed: `20260720`

Full output: [`analysis/final_results/paired_bootstrap_patient_cluster.csv`](analysis/final_results/paired_bootstrap_patient_cluster.csv).

### Full representations, seed 42

| Representation | Validation MAE | Test MAE |
| --- | ---: | ---: |
| RGB | 5.4090 | **4.9067** |
| YCrCb | **5.4046** | 5.1084 |
| Lab | 5.5419 | 5.2448 |
| HSV | 5.4661 | 5.2852 |
| Grayscale | 6.1552 | 5.9976 |

No non-RGB full representation outperforms RGB on the test split.

### Single-channel ablations

Each selected channel is repeated three times to preserve the input shape expected by the
pretrained network.

| Representation | Validation MAE | Test MAE |
| --- | ---: | ---: |
| RGB-R | 7.0720 | 6.7047 |
| RGB-G | 6.1317 | 5.8333 |
| RGB-B | 6.5042 | 6.0592 |
| Lab-L | 6.4377 | 6.0677 |
| Lab-a | 6.7317 | 6.0522 |
| Lab-b | 6.5460 | 6.0295 |
| HSV-H | 6.5357 | 6.1345 |
| HSV-S | 6.3637 | 5.9151 |
| HSV-V | 7.1904 | 6.7435 |
| YCrCb-Y | **6.0825** | **5.6944** |
| YCrCb-Cr | 6.4902 | 6.1461 |
| YCrCb-Cb | 6.5453 | 5.9632 |

YCrCb-Y is the strongest single channel, followed by RGB-G. Every single-channel model is
worse than full RGB.

### Individually seeded RGB models

| Model | Validation MAE | Test MAE |
| --- | ---: | ---: |
| RGB seed 42 | 5.4090 | 4.9067 |
| RGB seed 43 | 5.5342 | 5.0659 |
| RGB seed 44 | 5.3906 | 4.9930 |
| RGB seed 45 | 5.4178 | 4.9849 |
| RGB seed 46 | 5.3535 | 5.1115 |

## Pretrained Weights

All 22 checkpoints are on Hugging Face:
[mehmetaytugyuruk/retina-color-spaces-age-prediction](https://huggingface.co/mehmetaytugyuruk/retina-color-spaces-age-prediction)

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    "mehmetaytugyuruk/retina-color-spaces-age-prediction",
    "rgb/rgb_seed42.pt",
)
```

See the model card for the full file list and a loading example.

## Repository Structure

```text
retina-color-spaces-age-prediction/
├── analysis/final_results/       Metrics and bootstrap outputs behind the paper's tables
├── configs/                      Representation contracts and training template
├── data/
│   ├── manifests/                Patient-disjoint split definitions
│   └── normalization/
│       ├── <family>/             Per-representation train-split statistics
│       └── fusion/               12-channel fusion normalization statistics
├── docs/paper-to-code-mapping.md Which script/table corresponds to which paper section
├── models/<family>/<model>/      Per-model predictions.csv and training_history.csv
├── notebooks/
│   └── train_all_models.ipynb    End-to-end Colab reproduction
├── scripts/
│   ├── 01_prepare_training.py
│   ├── 02_evaluate_and_analyze.py
│   ├── generate_figure_assets.py Color-space visualization assets (paper Fig. 2)
│   └── train_fusion12ch.py       Standalone Fusion Model training script
├── src/retinal_color_transfer/   Training and analysis package
├── pyproject.toml                Dependencies and package configuration
└── README.md
```

Model checkpoints (`best_checkpoint.pt`) are not tracked in Git — they are released on
Hugging Face. The per-model `predictions.csv` and `training_history.csv` files *are*
tracked, so every number in the paper can be verified without downloading the image
dataset or re-running inference.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Reproducing the Study

### 1. Prepare caches, normalization, and configs

```bash
PYTHONPATH=src python3 scripts/01_prepare_training.py \
  --data-root /path/to/retinal-data
```

The command:

1. builds the prepared RGB cache;
2. derives all full, single-channel, and grayscale caches;
3. reuses valid cache entries;
4. computes missing normalization statistics from the train split only;
5. validates all 9,857 entries for every representation; and
6. writes configs for the canonical 21 models under `configs/generated/`.

Generated artifacts follow the same family layout:

```text
caches/<family>/<representation>/
data/normalization/<family>/<representation>_train_stats.json
models/<family>/<representation>_seed<seed>/
```

### 2. Train all models

Use [`notebooks/train_all_models.ipynb`](notebooks/train_all_models.ipynb) in a
GPU-enabled Colab runtime. Set its repository URL, raw-data root, and persistent Drive
run root, then use **Run all**.

The notebook validates the raw data, runs preparation, verifies the 21 generated configs,
trains every canonical model, trains the Fusion Model, resumes an interrupted model from
`latest_checkpoint.pt`, skips completed models, and runs final analysis.

During training a model directory contains:

```text
best_checkpoint.pt
latest_checkpoint.pt
training_history.csv
```

After successful training, `latest_checkpoint.pt` is removed. After evaluation, the
finalized directory contains exactly:

```text
best_checkpoint.pt
predictions.csv
training_history.csv
```

### 3. Recreate predictions and analysis

```bash
PYTHONPATH=src python3 scripts/02_evaluate_and_analyze.py \
  --data-root /path/to/retinal-data
```

The workflow validates and reuses every existing `predictions.csv`. If one is missing, it
performs validation and test inference from `best_checkpoint.pt`. It then writes model
metrics and the fixed Ensemble 1 vs Ensemble 2 bootstrap comparison under
`analysis/final_results/`.

Because the tracked `predictions.csv` files are already in this repository, this step
reproduces every reported number without access to the raw images.

## Generating Figure Assets

The script `scripts/generate_figure_assets.py` reproduces the color-space visualization
in the paper (Fig. 2) from a single retinal fundus image.

```bash
python3 scripts/generate_figure_assets.py --source img00007.png
```

This writes two groups under `assets/` (git-ignored):

| Group | Directory | Contents |
| --- | --- | --- |
| Pseudo-color composites | `assets/group1_false_color/` | `rgb`, `lab`, `hsv`, `ycrcb`, `gray` |
| Single-channel grayscale | `assets/group2_channels/` | `rgb_r/g/b`, `lab_l/a/b`, `hsv_h/s/v`, `ycrcb_y/cr/cb` |

**Technique:**

- **Group 1 (pseudo-color composites):** the three encoded channels of each color space
  are mapped to the red, green, and blue display channels, respectively. Channel
  assignment:

  | Color space | Ch 0 → Red | Ch 1 → Green | Ch 2 → Blue |
  | --- | --- | --- | --- |
  | HSV | H | S | V |
  | CIELAB | L | a | b |
  | YCrCb | Y | Cr | Cb |

  The displayed colors do not represent natural retinal colors; they provide a consistent
  pseudo-color visualization of the encoded channels.

- **Group 2 (grayscale):** each channel is saved individually as a grayscale image so its
  information content can be assessed independently.

**Preprocessing applied to every output:**

1. **Background mask** — pixels where `max(B,G,R) < 10` in the original image are forced
   to black after conversion, preventing Lab/YCrCb encoding artefacts (the a/b offset of
   128 would otherwise turn black pixels green/grey in pseudo-color renders).
2. **Crop** — the bounding rectangle of non-background pixels is computed once from the
   original BGR image (threshold = 25, matching `preprocessing/crop_pad_resize.py`) and
   applied to every output. No resize is performed.

**Channel range decisions:**

All OpenCV 8-bit channels occupy `[0, 255]` except HSV-H which uses `[0, 179]`. Only
HSV-H is linearly mapped from its fixed OpenCV range to the display range:

```python
H_display = (H.astype(float) * 255.0 / 179.0).astype('uint8')
```

Per-image auto-contrast (min-max normalization) is intentionally avoided: it would
misrepresent low-variance channels (e.g. H, which carries little retinal discriminative
information) as artificially detail-rich — contradicting the quantitative MAE results.

## Reproducibility Scope

The repository tracks code, configs, split manifests, normalization statistics, per-model
predictions, analysis outputs, and the end-to-end notebook. It intentionally does not
track raw retinal images, generated caches, or model checkpoints; the checkpoints are
released on Hugging Face.

Exact numerical reproduction of the *training* run requires the same source image
dataset. The manifests validate paths and split membership but cannot verify private
source image bytes that are not distributed with the repository. Reproduction of the
*reported metrics* requires nothing beyond this repository.

The primary claim is deliberately limited to the fixed Ensemble 1 and Ensemble 2 defined
above. Only RGB was trained with the four additional control seeds; the repository does
not claim seed-population robustness for Lab, HSV, or YCrCb.

## Citation

```bibtex
@inproceedings{yuruk2026colorspaces,
  title     = {Exploring the Impact of Alternative Color Spaces in Deep Retinal
               Age Prediction from Fundus Images},
  author    = {Yürük, Mehmet Aytuğ and Memiş, Abbas},
  booktitle = {ATEEC 2026},
  year      = {2026},
  note      = {Under review}
}
```

## License

Code released under the [MIT License](LICENSE). The dataset is separately licensed by its
authors (MIT, see the [dataset card](https://huggingface.co/datasets/ramankamran/retina-age-analysis)).
