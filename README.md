# 🧠 EEG-based Emotion Recognition

This repository contains the methodological framework and implementation code for my research on **EEG signal-based emotion recognition**, focusing on feature extraction, feature selection, domain adaptation, and transfer learning.

---

## 📌 Project Overview

This work aims to build a robust and generalizable emotion recognition system using electroencephalogram (EEG) signals.
The pipeline includes:
- High-dimensional EEG feature extraction
- MI-based feature selection (交互增强互信息特征选择AEMIFS)
- Domain adaptation algorithms (MMD+DANN)
- Cross-subject transfer learning experiments
- Comprehensive evaluation and visualization tools

---
📌 Core Work
To address the problems of feature redundancy and poor cross-subject generalization in EEG emotion recognition for pattern recognition tasks, this paper proposes two effective strategies.
Firstly, aiming at the issue of feature redundancy, we propose an interaction information maximization feature selection algorithm. Specifically, we design a novel evaluation criterion for feature quality, which is defined as the relevance score minus the redundancy score. The relevance score is composed of individual independent information and synergistic interaction information between features, while the redundancy score is constructed by deducting individual information from independent redundancy. This design further strengthens the importance of the informative contribution of candidate features.
Secondly, to alleviate the poor cross-subject generalization capability, we present a multi-source domain selection strategy based on Maximum Mean Discrepancy (MMD). The strategy screens appropriate source domains before transfer learning to effectively avoid negative transfer. In the transfer stage, we adopt the Domain-Adversarial Neural Network (DANN) for domain adaptation. Relying on domain adversarial learning, DANN can simultaneously align the marginal distribution and conditional distribution between source and target domains. Compared with Transfer Component Analysis (TCA) that only adapts marginal distributions and traditional shallow Joint Distribution Adaptation (JDA), DANN is capable of mining high-order nonlinear feature correlations of EEG signals. It achieves more comprehensive domain adaptation modeling, better interpretability and superior generalization performance, making it more suitable for cross-subject EEG scenarios with obvious individual differences and distribution shifts.
## 📂 Repository Structure
1. 主程序入口        作用：一键运行整个流程（特征选择 + 迁移学习 + 训练 + 测试）
main.py 或 run.py
2. 创新算法（最重要）
AEMIFS 特征选择代码
例：AEMIFS.py / feature_selection.py
MMD 多源域选择代码
例：mmd_domain_selection.py
DANN 域对抗迁移网络代码
例：DANN_model.py / network.py
3. 数据处理与特征提取
data_loader.py（加载 EEG 数据）
preprocess.py（滤波、分段等）
feature_extract.py（DE、PSD 等脑电特征）
4. 训练、测试、评估代码
train.py
test.py
metrics.py（准确率、召回率等评价指标）
5. 对比算法（可选但推荐）
TCA、JDA、MIFS 等你论文里对比过的方法方便审稿人复现对比结果
6. 必须有的说明文件
README.md（写清楚怎么运行、环境、依赖）
requirements.txt（Python 依赖包）
