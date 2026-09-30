# WaveState

**WaveState: A parallel selective state-space attention model with class-adaptive time-frequency features for multivariate time series classification**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()

This repository is the official implementation of the paper:

> Long Li, Wanghu Chen, Jing Li. **WaveState: A parallel selective state-space attention model with class-adaptive time-frequency features for multivariate time series classification.** *Knowledge-Based Systems*, 333 (2026) 115014.
> [https://doi.org/10.1016/j.knosys.2025.115014](https://doi.org/10.1016/j.knosys.2025.115014)

---

## 📌 Overview

Multivariate time series classification (MTSC) is crucial for handling complex dynamic systems and enhancing decision-making across various fields. Existing deep learning approaches still face limitations in:

- capturing subtle categorical distinctions in the **frequency domain**;
- modeling **long-range dependencies** efficiently (quadratic complexity of attention);
- balancing feature discrimination with **computational efficiency** on high-dimensional data.

**WaveState** addresses these issues through four tightly-coupled components.

### Architecture

<div align="center">

<img src="figures/WaveState.pdf" width="800" alt="WaveState Architecture"/>

*Fig. 1 — (A) Class-adaptive wavelet transform; (B) Multi-scale convolution; (C) Parallel Selective State-space (PSSA) attention; (D) Sparse feedforward network.*

</div>

### Key Components

| Module | Paper Section | Code | Role |
|---|---|---|---|
| **CAWT** — Class-Adaptive Wavelet Transform | §3.2 (Fig. 1A) | `models/model.py` → `AdaptiveWaveletTransform` | Flexible multi-scale frequency-domain decomposition emphasizing class-discriminative bands |
| **MS Conv** — Multi-Scale Convolution | §3.3 (Fig. 1B) | `models/model.py` → `WaveState.feature1` | Parallel conv kernels (size 3/5/7) extracting local temporal patterns at multiple scales |
| **PSSA** — Parallel Selective State-space Attention | §3.4 (Fig. 2) | `models/pssa_block.py` | 4 parallel selective SSM (Mamba) branches with gating, linear complexity $O(T(C+4N))$ |
| **S-FFN** — Sparse FeedForward Network | §3.5 (Fig. 3) | `models/model.py` → `SparseFFN` | Threshold-based weight sparsification: $O(n \cdot d \cdot d_h) \to O(\alpha \cdot n \cdot d \cdot d_h)$ |

### PSSA at a glance

<div align="center">

<img src="figures/PSSA_Block(new).pdf" width="700" alt="PSSA Block"/>

*Fig. 2 — (A) PSSA architecture with normalization and residual pathways; (B) parallel selective state-space block; (C) selective state-space block.*

</div>

Each branch implements the discrete-time state-space formulation:

$$
\mathbf{h}_i(t) = \mathbf{A}_i \mathbf{h}_i(t-1) + \mathbf{B}_i \mathbf{x}_i(t), \qquad
\mathbf{y}_i(t) = \mathbf{C}_i \mathbf{h}_i(t), \qquad
\mathbf{A}_i = \mathbf{P}_i \mathbf{D}_i \mathbf{P}_i^{-1}
$$

with a gating mechanism $\mathbf{z}_i = \sigma(\text{Linear}(\mathbf{y}_i)) \odot \mathbf{y}_i$ for dynamic feature selection, and branch outputs fused via:

$$
\mathbf{Y} = \text{Linear}([\mathbf{z}_1; \mathbf{z}_2; \mathbf{z}_3; \mathbf{z}_4])
$$

Time-domain (multi-scale conv) and frequency-domain (CAWT) features are fused as
$F = \alpha \cdot F_{time} + (1-\alpha) \cdot F_{freq}$ before entering PSSA.

---

## 📊 Main Results

### UEA Archive — 30 datasets

| Model | Mean Accuracy | Std. Dev. | Params |
|---|---|---|---|
| **WaveState (ours)** | **0.836** | **0.185** | **30.5K** |
| ShapeFormer | 0.786 | — | — |
| WHEN | 0.786 | — | — |
| FCN | 0.677 | 0.237 | — |
| ResNet | 0.672 | 0.213 | — |

- Perfect accuracy (1.000) on challenging datasets: **BasicMotions**, **Cricket**, **Epilepsy**.
- Lowest accuracy standard deviation among all 17 baselines → highest robustness/adaptability.
- Total params: **30.5K** only.

### Long time-series advantages (vs. ConvTran)

| Dataset | Length | ConvTran acc / params | WaveState acc / params |
|---|---|---|---|
| Cricket | 1,197 | 1.000 / 31.3K | 1.000 / **7.8K** |
| EigenWorms | 17,984 | 0.593 / 299.8K | **0.708** / **7.7K** (−97.4%) |
| EthanolConc | 1,751 | 0.361 / 37.0K | **0.742** / **6.9K** |
| MotorImg | 3,000 | 0.500 / 119.4K | **0.930** / 22.5K |
| StandWalkJump | 2,500 | 0.333 / 50.0K | **0.667** / **7.2K** |

Inference-time reduction reaches up to **96.7%** on EigenWorms; empirical runtime scales **$O(T)$** (linear) versus $O(T^2)$ for Transformer baselines (see `figures/complexity_scaling.pdf`).

### Ablation study (§4.3.1)

| Config | CAWT | MS Conv | Attention | FFN | Accuracy | Params |
|:-:|:-:|:-:|:-:|:-:|---|---|
| **A (WaveState)** | ✓ | ✓ | PSSA | Sparse | **0.836** | 30.5K |
| B | ✓ | ✓ | PSSA | Dense | 0.761 | 46.9K |
| C | ✓ | ✓ | Simple | Sparse | 0.776 | 29.6K |
| D | ✓ | ✓ | Simple | Dense | 0.748 | 27.0K |
| E–H | ✗ | ✓ | Simple / PSSA | Sparse / Dense | 0.645–0.667 | 27.0–46.9K |

- CAWT alone contributes **+16.9 pts** (A vs. E).
- CAWT×PSSA shows **synergy**: +6.0 pts with CAWT (A vs. C) vs. only +0.3 pts without CAWT (E vs. G).
- More parameters ≠ better performance (B/F have the most params but underperform).

### PSSA branch ablation (§4.3.3)

| Dataset (length) | 1 branch | 2 branches | 4 branches |
|---|---|---|---|
| SpokenArabicDigits (2,500) | 0.822 | 0.917 | **0.975** |
| RacketSports (30) | 0.988 | 0.986 | 0.989 |
| **Avg. over 5 datasets** | 0.848 | — | **0.875** (−7.8% params) |

Multi-branch design benefits long/complex sequences at near-constant inference time.

### CAWT feature discriminability (t-SNE on imbalanced LSST, §4.4.1)

| Metric | Original | CAWT | Δ |
|---|---|---|---|
| Intra-class compactness ↓ | 0.479 | **0.410** | −14.4% |
| Inter-class mean distance ↑ | 0.282 | **0.486** | +72.3% |
| Fisher score ↑ | 0.520 | **0.700** | +34.6% |
| Separation index ↑ | 0.643 | **0.766** | +19.1% |

Visual evidence: `figures/visual_ori_tsne.pdf` (original) vs. `figures/visual_dwt_tsne.pdf` (CAWT).

---

## 📁 Repository Structure

```
WaveState/
├── main.py                     # Entry point: batch training & evaluation over datasets
├── Training.py                 # SupervisedTrainer + epoch/validation loop
├── utils.py                    # Data loading (UEA), config setup, model save/load
├── acc_params.py               # Accuracy/parameter aggregation utilities
├── cawt_visual.py              # Wavelet decomposition visualization tool
├── models/
│   ├── model.py                # WaveState + AdaptiveWaveletTransform + FFN/SparseFFN
│   ├── pssa_block.py           # Parallel Selective State-space Attention block
│   ├── transformer.py          # Transformer baseline
│   ├── convtran.py             # ConvTran / CasualConvTran baselines
│   ├── Attention.py            # Simple Attention baseline
│   ├── loss.py                 # Cross-entropy loss module + L2 regularization
│   ├── optimizers.py           # Optimizer factory (AdamW)
│   ├── analysis.py             # Classification metrics / confusion matrix
│   ├── embedding.py            # Data/type embeddings
│   ├── AbsolutePositionalEncoding.py
│   └── visual*.py              # t-SNE / attention heatmap visualizations
├── Dataset/                    # UEA archive data (auto-downloaded if missing)
├── Results/                    # Timestamped outputs: configs, checkpoints, predictions
├── figures/                    # All paper figures (PDF)
├── paper/                      # Published paper (PDF)
├── result_analysis/            # Result tables & plotting scripts
├── paper_figures/              # Long-series figure scripts
├── comparisons/                # CD diagrams & Multi-Comparison-Matrix tooling
└── LICENSE                     # Apache 2.0
```

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.8, PyTorch ≥ 2.0 (CUDA recommended)
- [mamba_ssm](https://github.com/state-spaces/mamba) (selective scan CUDA kernels)
- `pytorch_wavelets`, `timm`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `tensorboard`, `art`

```bash
pip install torch mamba_ssm pytorch_wavelets timm numpy pandas scikit-learn matplotlib seaborn tqdm tensorboard art
```

> `models/pssa_block.py` relies on `mamba_ssm` CUDA kernels — ensure your PyTorch / CUDA versions match.

### Dataset

Download the [UEA Multivariate Time Series Classification Archive (2018)](https://timeseriesclassification.com/) and place it under:

```
Dataset/UEA/<DatasetName>/
```

`Data_Verifier` in `utils.py` attempts automatic download if the directory is missing.

### Training & Evaluation

```bash
python main.py \
    --data_path Dataset/UEA/ \
    --Net_Type C-T \
    --epochs 100 \
    --train_batch_size 16 \
    --lr 0.001 \
    --device 0
```

`main.py` iterates over **every dataset** in `data_path`, trains the model, evaluates the best checkpoint, and writes:

- per-dataset accuracy CSVs → `Results/<timestamp>/predictions/`
- overall summary → `Results/<timestamp>/TMamba4MTSC_Results.csv`
- best checkpoints → `Results/<timestamp>/checkpoints/`

### Key Hyperparameters (paper setup)

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr = 0.001, weight decay = 5e-4) |
| Batch size | 16 |
| Epochs | 200 (paper) / 100 (default in `main.py`) |
| PSSA heads | 8 |
| Train/val split | 80% / 20% |
| Wavelet | db4, max decomposition level 3 |
| Hardware | NVIDIA A100-SXM4-40GB, Intel Xeon Platinum 8352S |

Model selection is driven by `--Net_Type` via `model_factory` in `models/model.py`:
`MT` → WaveState · `T` → Transformer · `C-T`/default → ConvTran family.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{li2025wavestate,
  title   = {WaveState: A parallel selective state-space attention model with class-adaptive time-frequency features for multivariate time series classification},
  author  = {Li, Long and Chen, Wanghu and Li, Jing},
  journal = {Knowledge-Based Systems},
  volume  = {333},
  pages   = {115014},
  year    = {2026},
  doi     = {10.1016/j.knosys.2025.115014}
}
```

## 🙏 Acknowledgements

- Datasets: [UEA Multivariate Time Series Classification Archive](https://timeseriesclassification.com/) curated by Bagnall et al.
- Based on [Mamba / Selective State Spaces](https://github.com/state-spaces/mamba) and [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).
- Baseline code adapted from [ConvTran](https://github.com/Naasin/ConvTran).

## 📄 License

This project is released under the [Apache License 2.0](LICENSE).
