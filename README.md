# 🎯 HiDiffRec: Hierarchical User Preference Modeling via Conditional Diffusion for Graph Recommendation

[![Python](https://img.shields.io/badge/Python-3.8.18-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **HiDiffRec**, a hierarchical user preference modeling framework that addresses hop-wise quality heterogeneity in graph-based recommendation through conditional diffusion and contrastive learning.

---

## 📖 Overview

HiDiffRec tackles a critical limitation in existing graph recommendation systems: **uniform aggregation of multi-hop collaborative signals ignores quality heterogeneity across propagation depths**. Our framework introduces:

- **🔄 CDHPE (Conditional Diffusion-based High-order Preference Extraction)**: Progressively reconstructs high-order user representations conditioned on stable low-order embeddings.

- **⚖️ CLPA (Contrastive Learning-based Preference Alignment)**: Aligns fused representations with refined explicit preferences using density-based filtering and contrastive learning.

---

## Key Features

- ✅ Hierarchical preference modeling across GNN propagation layers
- ✅ Conditional diffusion reconstruction guided by low-order interaction cues
- ✅ Density-based noise filtering using Local Outlier Factor (LOF)
- ✅ Strong performance: **5.66%**, **5.48%**, and **3.65%** improvements on Gowalla, MovieLens-1M, and Amazon-CDs (Recall@20)
- ✅ Enhanced robustness under noisy interactions (30%-80% noise ratios)

---

## 🛠️ Dependencies

- Python 3.8.18
- PyTorch 2.4.1+cu121
- numpy, scipy, scikit-learn
```bash
pip install -r requirements.txt
```

---

## 🚀 Running HiDiffRec

### 🎬 MovieLens-1M
```bash
python main.py --dataset=Movies-1M --diff_weight=1 --cl_weight=0.2 --lofk=10 --gamma=1.5 --seed=2025 --epochs=40
```

### 📍 Gowalla
```bash
python main.py --dataset=gowalla --diff_weight=0.9 --cl_weight=0.1 --lofk=15 --gamma=2 --seed=2025 --epochs=600
```

### 💿 Amazon-CDs
```bash
python main.py --dataset=amazoncds --diff_weight=0.9 --cl_weight=0.1 --lofk=15 --gamma=2 --seed=2025 --epochs=1000
```

---

## 📊 Main Results

Performance comparison on three benchmark datasets (**Recall@20**):

| Method | Gowalla | MovieLens-1M | Amazon-CDs |
|:------:|:-------:|:------------:|:----------:|
| LightGCN | 0.2069 | 0.2559 | 0.1525 |
| LayerGCN | 0.2122 | 0.2586 | 0.1546 |
| HDRM | 0.2227 | 0.2592 | 0.1542 |
| ConDiff | 0.2198 | 0.2586 | 0.1519 |
| **HiDiffRec** | **0.2332** | **0.2734** | **0.1619** |
| 📈 Improvement | **+5.66%** | **+5.48%** | **+3.65%** |

> 💡 *All improvements are statistically significant with p < 0.01*

---
