# 🧠 EEG-based Emotion Recognition

This repository contains the methodological framework and implementation code for my research on **EEG signal-based emotion recognition**, focusing on feature extraction, feature selection, domain adaptation, and transfer learning.

---

## 📌 Project Overview

This work aims to build a robust and generalizable emotion recognition system using electroencephalogram (EEG) signals.
The pipeline includes:
- High-dimensional EEG feature extraction
- MI-based feature selection (MI-IFS / MI-IFS-CANDD)
- Domain adaptation algorithms (JDA, DANN)
- Cross-subject transfer learning experiments
- Comprehensive evaluation and visualization tools

---

## 📂 Repository Structure

```text
.
├── AI/                          # Core experiment and model code
│   ├── common.py                # Common utility functions
│   ├── reporting.py             # Result reporting and metrics
│   ├── domain_adaptation.py     # Domain adaptation algorithms
│   ├── feature_selection_*.py   # Feature selection pipelines
│   ├── main_*.py                # Entry scripts for different experiments
│   └── artifacts/                # Intermediate results (e.g., feature masks)
├── else_file/                   # Extended experiments and comparisons
├── readme/                      # Project documentation
└── README.md                    # This file
