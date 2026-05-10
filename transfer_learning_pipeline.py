from __future__ import annotations

import copy
import json
import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from AI.common import (
    DEFAULT_RANDOM_STATE,
    binary_metrics,
    dataset_layout_summary,
    ensure_ai_output_path,
    ensure_artifact_dir,
    ensure_numpy_pickle_compat,
    load_deap_features,
    load_pickle,
    make_outer_folds,
    positive_class_scores,
    save_json,
    summarize_metric_list,
    to_serializable,
)
from AI.domain_adaptation import build_domain_adapter, compute_mmd, estimate_gamma
from AI.reporting import format_metric_bars, format_metric_table, write_markdown_report

CLASSIFIER_ORDER: Sequence[str] = ("LR", "KNN", "DT")
RESUME_STATE_VERSION = 1


@dataclass
class TransferLearningConfig:
    """管理监督迁移/微调阶段的核心配置。"""

    mask_path: Path = Path("AI/artifacts/miifs_mask_latest.pkl")
    feature_source_mode: str = "mask"
    feature_source_label: str = "MIIFS"
    summary_path: Path = Path("AI/artifacts/transfer_learning_summary_latest.json")
    report_path: Path = Path("AI/artifacts/transfer_learning_report_latest.md")
    random_state: int = DEFAULT_RANDOM_STATE
    evaluation_protocol: str = "loso"
    cv_splits: int = 5
    max_target_subjects: int | None = 5
    target_train_ratio: float = 0.60
    target_val_ratio: float = 0.20
    target_test_ratio: float = 0.20
    max_source_subjects: int = 6
    source_sample_cap: int = 128
    mmd_sample_cap: int = 160
    mmd_prefix_top_k: int = 0
    transfer_variant: str = "supervised"
    target_repeat_grid: Tuple[int, ...] = (1, 2, 4)
    source_target_ratio_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    jda_dim: int = 32
    jda_iterations: int = 6
    jda_n_components: int = 48
    jda_lambda: float = 1.0
    jda_reg: float = 1e-6
    jda_pseudo_labeler: str = "1nn"
    jda_pseudo_neighbors: int = 3
    jda_pseudo_change_tol: float = 1e-3
    jda_mmd_delta_tol: float = 1e-3
    jda_confidence_delta_tol: float = 5e-3
    jda_early_stop_patience: int = 2
    jda_min_iterations: int = 2
    jda_pseudo_keep_ratio_grid: Tuple[float, ...] = (0.0, 0.15, 0.25, 0.4)
    jda_pseudo_target_repeat_grid: Tuple[int, ...] = (1, 2)
    reference_summary_path: Path | None = None
    use_inverse_mmd_source_weighting: bool = True
    use_positive_ratio_source_weighting: bool = True
    positive_ratio_weight_strength: float = 4.0
    inverse_mmd_weight_epsilon: float = 1e-6
    source_positive_ratio_gap_threshold: float = 0.08
    gate_repeat_count: int = 3

    # === 新增 TwoDCNN 相关配置 ===
    twod_cnn_encoding_dim: int = 100
    twod_cnn_epochs: int = 20
    twod_cnn_batch_size: int = 32

    dann_encoding_dim: int = 64
    dann_lambda: float = 1.0
    dann_epochs: int = 20
    dann_batch_size: int = 32

    # ===========================
    def normalized_transfer_variant(self) -> str:
        """规范化迁移训练分支名称，便于在主流程里切换不同适配策略。"""
        variant = str(self.transfer_variant).lower().strip()
        if variant not in {"supervised", "enhanced_jda", "twod_cnn", "dann"}:
            raise ValueError(f"不支持的 transfer_variant: {self.transfer_variant!r}")
        return variant

        # === 新增 TwoDCNN 规范化方法 ===

    def normalized_twod_cnn_encoding_dim(self) -> int:
        """规范化 CNN 输出特征维度。"""
        return max(1, int(self.twod_cnn_encoding_dim))

    def normalized_twod_cnn_epochs(self) -> int:
        """规范化 CNN 训练轮数。"""
        return max(1, int(self.twod_cnn_epochs))

    def normalized_twod_cnn_batch_size(self) -> int:
        """规范化 CNN 批次大小。"""
        return max(1, int(self.twod_cnn_batch_size))

    # ===========================

    def normalized_protocol(self) -> str:
        """规范化迁移评估协议。"""
        protocol = str(self.evaluation_protocol).lower().strip()
        if protocol not in {"loso", "group_kfold"}:
            raise ValueError(f"不支持的 evaluation_protocol: {self.evaluation_protocol!r}")
        return protocol

    def normalized_feature_source_mode(self) -> str:
        """规范化特征入口模式。"""
        mode = str(self.feature_source_mode).lower().strip()
        if mode not in {"mask", "full"}:
            raise ValueError(f"不支持的 feature_source_mode: {self.feature_source_mode!r}")
        return mode

    def normalized_feature_source_label(self) -> str:
        """规范化特征入口标签，便于报告和日志展示。"""
        label = str(self.feature_source_label).strip()
        return label or "unnamed_feature_source"

    def normalized_target_repeat_grid(self) -> Tuple[int, ...]:
        """规范化目标域权重网格。"""
        grid = tuple(sorted({int(value) for value in self.target_repeat_grid if int(value) >= 1}))
        if not grid:
            raise ValueError("target_repeat_grid 至少要包含一个不小于 1 的整数。")
        return grid

    def normalized_transfer_variant(self) -> str:
        """规范化迁移训练分支名称，便于在主流程里切换不同适配策略。"""
        variant = str(self.transfer_variant).lower().strip()
        if variant not in {"supervised", "enhanced_jda", "twod_cnn", "dann"}:
            raise ValueError(f"不支持的 transfer_variant: {self.transfer_variant!r}")
        return variant

    def normalized_mmd_prefix_top_k(self) -> int | None:
        """规范化 MMD 前缀候选上限；小于等于 0 时表示保留全部累计前缀。"""
        normalized = int(self.mmd_prefix_top_k)
        if normalized <= 0:
            return None
        return max(1, normalized)

    def normalized_mmd_prefix_label(self) -> str:
        """把 MMD 前缀搜索上限转成便于日志和报告展示的文本。"""
        normalized = self.normalized_mmd_prefix_top_k()
        return "all_prefixes" if normalized is None else str(normalized)

    def normalized_source_target_ratio_grid(self) -> Tuple[float, ...]:
        """规范化源域/目标域强度比例网格，避免出现无效或负值。"""
        grid = tuple(sorted({float(value) for value in self.source_target_ratio_grid if float(value) > 0.0}))
        if not grid:
            raise ValueError("source_target_ratio_grid 至少要包含一个大于 0 的数值。")
        return grid

    def normalized_source_positive_ratio_gap_threshold(self) -> float:
        """规范化源域正类比例过滤阈值。"""
        return max(0.0, float(self.source_positive_ratio_gap_threshold))

    def normalized_gate_repeat_count(self) -> int:
        """规范化 gate 复核次数；最少保留 1 次主验证。"""
        return max(1, int(self.gate_repeat_count))

    def normalized_jda_dim(self) -> int:
        """规范化 JDA 潜空间维度。"""
        return max(2, int(self.jda_dim))

    def normalized_jda_iterations(self) -> int:
        """规范化 JDA 迭代次数。"""
        return max(1, int(self.jda_iterations))

    def normalized_jda_n_components(self) -> int:
        """规范化 JDA 进入线性投影前的工作维度。"""
        return max(2, int(self.jda_n_components))

    def normalized_jda_lambda(self) -> float:
        """规范化 JDA 的结构保持项权重。"""
        return max(1e-8, float(self.jda_lambda))

    def normalized_jda_reg(self) -> float:
        """规范化 JDA 广义特征值问题的数值稳定项。"""
        return max(1e-10, float(self.jda_reg))

    def normalized_jda_pseudo_labeler(self) -> str:
        """规范化 JDA 伪标签分类器名称。"""
        labeler = str(self.jda_pseudo_labeler).lower().strip()
        if labeler not in {"1nn", "knn", "lr", "svm"}:
            raise ValueError(f"不支持的 jda_pseudo_labeler: {self.jda_pseudo_labeler!r}")
        return labeler

    def normalized_jda_pseudo_neighbors(self) -> int:
        """规范化 JDA 伪标签 KNN 邻居数。"""
        return max(1, int(self.jda_pseudo_neighbors))

    def normalized_jda_pseudo_change_tol(self) -> float:
        """规范化 JDA 伪标签变化收敛阈值。"""
        return max(0.0, float(self.jda_pseudo_change_tol))

    def normalized_jda_mmd_delta_tol(self) -> float:
        """规范化 JDA 潜空间 MMD 变化收敛阈值。"""
        return max(0.0, float(self.jda_mmd_delta_tol))

    def normalized_jda_confidence_delta_tol(self) -> float:
        """规范化 JDA 置信度变化收敛阈值。"""
        return max(0.0, float(self.jda_confidence_delta_tol))

    def normalized_jda_early_stop_patience(self) -> int:
        """规范化 JDA 连续稳定轮数阈值。"""
        return max(1, int(self.jda_early_stop_patience))

    def normalized_jda_min_iterations(self) -> int:
        """规范化 JDA 允许早停前的最少迭代轮数。"""
        return max(1, int(self.jda_min_iterations))

    def normalized_jda_pseudo_keep_ratio_grid(self) -> Tuple[float, ...]:
        """规范化论文式 JDA 中高置信伪标签样本保留比例网格。"""
        grid = tuple(sorted({max(0.0, min(1.0, float(value))) for value in self.jda_pseudo_keep_ratio_grid}))
        if not grid:
            raise ValueError("jda_pseudo_keep_ratio_grid 至少要包含一个数值。")
        return grid

    def normalized_jda_pseudo_target_repeat_grid(self) -> Tuple[int, ...]:
        """规范化论文式 JDA 中伪标签目标样本重复次数网格。"""
        grid = tuple(sorted({max(1, int(value)) for value in self.jda_pseudo_target_repeat_grid}))
        if not grid:
            raise ValueError("jda_pseudo_target_repeat_grid 至少要包含一个整数。")
        return grid

    def normalized_twod_cnn_encoding_dim(self) -> int:
        """规范化 TwoD_CNN 输出特征维度。"""
        return max(1, int(self.twod_cnn_encoding_dim))

    def normalized_twod_cnn_epochs(self) -> int:
        """规范化 TwoD_CNN 训练轮数。"""
        return max(1, int(self.twod_cnn_epochs))

    def normalized_twod_cnn_batch_size(self) -> int:
        """规范化 TwoD_CNN 批次大小。"""
        return max(1, int(self.twod_cnn_batch_size))

    def validate_ratios(self) -> None:
        """检查目标域 train/val/test 划分比例是否合法。"""
        ratio_sum = float(self.target_train_ratio + self.target_val_ratio + self.target_test_ratio)
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError("target_train_ratio、target_val_ratio、target_test_ratio 之和必须为 1。")
        if min(self.target_train_ratio, self.target_val_ratio, self.target_test_ratio) <= 0:
            raise ValueError("目标域 train/val/test 比例都必须大于 0。")


@dataclass
class TransferFeaturePreprocessor:
    """对已选中的原始特征做对数变换和鲁棒标准化。"""

    positive_mask: np.ndarray | None = None
    scaler: RobustScaler | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在训练块上拟合预处理器并返回处理后的特征。"""
        X = np.asarray(X, dtype=np.float32).copy()
        self.positive_mask = np.all(X > 0, axis=0)
        if np.any(self.positive_mask):
            X[:, self.positive_mask] = np.log1p(X[:, self.positive_mask])
        self.scaler = RobustScaler()
        return np.asarray(self.scaler.fit_transform(X), dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """把训练块学到的预处理规则应用到验证或测试块。"""
        if self.positive_mask is None or self.scaler is None:
            raise ValueError("迁移预处理器尚未拟合，不能直接调用 transform。")
        X = np.asarray(X, dtype=np.float32).copy()
        if np.any(self.positive_mask):
            X[:, self.positive_mask] = np.log1p(X[:, self.positive_mask])
        return np.asarray(self.scaler.transform(X), dtype=np.float32)

    def describe(self) -> Dict[str, int]:
        """输出当前预处理器的关键统计信息。"""
        if self.positive_mask is None:
            raise ValueError("迁移预处理器尚未拟合，不能输出描述信息。")
        return {
            "selected_feature_count": int(len(self.positive_mask)),
            "log1p_feature_count": int(np.sum(self.positive_mask)),
        }


DEFAULT_CLASSIFIER_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "LR": [
        {"C": 0.125, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 0.25, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 0.5, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 2.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 4.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        {"C": 8.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
    ],
    "KNN": [
        {"n_neighbors": 3, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 5, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 7, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 9, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 11, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 13, "weights": "distance", "metric": "cosine", "n_jobs": -1},
        {"n_neighbors": 7, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        {"n_neighbors": 9, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        {"n_neighbors": 11, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        {"n_neighbors": 13, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
    ],
    "DT": [
        {"max_depth": 5, "min_samples_leaf": 4, "class_weight": "balanced"},
        {"max_depth": 8, "min_samples_leaf": 4, "class_weight": "balanced"},
        {"max_depth": 12, "min_samples_leaf": 6, "class_weight": "balanced"},
        {"max_depth": 16, "min_samples_leaf": 8, "class_weight": "balanced"},
        {"max_depth": 20, "min_samples_leaf": 10, "class_weight": "balanced"},
        {"max_depth": None, "min_samples_leaf": 8, "class_weight": "balanced"},
        {"max_depth": None, "min_samples_leaf": 12, "class_weight": "balanced"},
    ],
}

CLASSIFIER_SEARCH_PROFILES: Dict[str, Dict[str, Any]] = {
    "LR": {
        "param_grid": [
            {"C": 0.25, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 0.5, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 2.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 4.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 8.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        ],
        "target_repeat_grid": (1, 2, 4, 6, 8),
        "source_target_ratio_grid": (0.25, 0.5, 0.75, 1.0),
    },
    "KNN": {
        "param_grid": [
            {"n_neighbors": 3, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 5, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 7, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 9, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 11, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 13, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 7, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 9, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 11, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        ],
        "target_repeat_grid": (1, 2, 3, 4),
        "source_target_ratio_grid": (0.1, 0.15, 0.2, 0.25, 0.35, 0.5),
    },
    "DT": {
        "param_grid": [
            {"max_depth": 5, "min_samples_leaf": 4, "class_weight": "balanced"},
            {"max_depth": 8, "min_samples_leaf": 4, "class_weight": "balanced"},
            {"max_depth": 12, "min_samples_leaf": 6, "class_weight": "balanced"},
            {"max_depth": 16, "min_samples_leaf": 8, "class_weight": "balanced"},
            {"max_depth": None, "min_samples_leaf": 8, "class_weight": "balanced"},
            {"max_depth": None, "min_samples_leaf": 12, "class_weight": "balanced"},
        ],
        "target_repeat_grid": (1, 2, 4, 6, 8),
        "source_target_ratio_grid": (0.25, 0.5, 0.75, 1.0),
    },
}

TRANSFER_GATE_MIN_ACC_IMPROVEMENT_BY_CLASSIFIER: Dict[str, float] = {
    "LR": 0.001,
    "KNN": 0.001,
    "DT": 0.001,
}

TRANSFER_GATE_MIN_F1_IMPROVEMENT_BY_CLASSIFIER: Dict[str, float] = {
    "LR": 0.001,
    "KNN": 0.001,
    "DT": 0.001,
}


def _classifier_search_profile(classifier_name: str, config: TransferLearningConfig) -> Dict[str, Any]:
    """返回某个分类器实际使用的搜索配置，并与全局配置取交集以控制运行时间。"""
    # 中文作用: 返回当前分类器在迁移学习阶段真正使用的搜索空间配置。
    profile = dict(CLASSIFIER_SEARCH_PROFILES[classifier_name])
    global_params = DEFAULT_CLASSIFIER_GRIDS[classifier_name]
    global_repeat_grid = set(config.normalized_target_repeat_grid())
    global_ratio_grid = set(config.normalized_source_target_ratio_grid())
    profile["param_grid"] = [dict(params) for params in profile["param_grid"] if params in global_params]
    profile["target_repeat_grid"] = tuple(value for value in profile["target_repeat_grid"] if int(value) in global_repeat_grid)
    profile["source_target_ratio_grid"] = tuple(
        value for value in profile["source_target_ratio_grid"] if float(value) in global_ratio_grid
    )
    profile["search_prefix_limit"] = config.normalized_mmd_prefix_label()
    if not profile["param_grid"]:
        raise RuntimeError(f"{classifier_name} 的参数网格被裁剪为空，无法继续搜索。")
    if not profile["target_repeat_grid"]:
        raise RuntimeError(f"{classifier_name} 的 target_repeat_grid 被裁剪为空，无法继续搜索。")
    if not profile["source_target_ratio_grid"]:
        raise RuntimeError(f"{classifier_name} 的 source_target_ratio_grid 被裁剪为空，无法继续搜索。")
    return profile


def _format_threshold_preview(preview_rows: Sequence[Dict[str, float]]) -> str:
    """把阈值扫描的前几名结果压缩成一行，便于打印诊断。"""
    # 中文作用: 把阈值扫描结果压缩成单行文本，便于调试时快速比较。
    if not preview_rows:
        return "empty"
    return " | ".join(
        f"{row['threshold']:.3f}:{row['ACC']:.4f}/{row['F1']:.4f}/pos={row['PredPosRatio']:.3f}"
        for row in preview_rows
    )


def _format_metric_snapshot(metrics: Dict[str, float]) -> str:
    """把单次评估得到的常见指标压缩成一行文本。"""
    # 中文作用: 统一 ACC/F1/AUC/BACC/MCC/PRE/REC/SPE 的打印口径，便于排查哪里退化。
    return (
        f"ACC={float(metrics['ACC']):.4f} F1={float(metrics['F1']):.4f} "
        f"AUC={float(metrics['AUC']):.4f} BACC={float(metrics['BACC']):.4f} "
        f"MCC={float(metrics['MCC']):.4f} PRE={float(metrics['PRE']):.4f} "
        f"REC={float(metrics['REC']):.4f} SPE={float(metrics['SPE']):.4f}"
    )


def _format_summary_metric_snapshot(summary_block: Dict[str, Dict[str, Dict[str, float]]], classifier_name: str) -> str:
    """把汇总后的分类器均值指标压缩成一行文本。"""
    # 中文作用: 统一汇总结果的打印口径，避免最终日志只剩 ACC/F1 看不出具体问题。
    return (
        f"ACC={_metric_mean(summary_block, classifier_name, 'ACC'):.4f} "
        f"F1={_metric_mean(summary_block, classifier_name, 'F1'):.4f} "
        f"AUC={_metric_mean(summary_block, classifier_name, 'AUC'):.4f} "
        f"BACC={_metric_mean(summary_block, classifier_name, 'BACC'):.4f} "
        f"MCC={_metric_mean(summary_block, classifier_name, 'MCC'):.4f} "
        f"PRE={_metric_mean(summary_block, classifier_name, 'PRE'):.4f} "
        f"REC={_metric_mean(summary_block, classifier_name, 'REC'):.4f} "
        f"SPE={_metric_mean(summary_block, classifier_name, 'SPE'):.4f}"
    )


def _format_source_rank_preview(ranked_source_rows: Sequence[Dict[str, Any]], limit: int = 6) -> str:
    """把按 MMD 排序的源域前几名压缩成短文本，避免误读成“只选前三个”。"""
    # 中文作用: 仅用于报告预览源域排序，不参与真正的前缀搜索。
    if not ranked_source_rows:
        return "empty"
    limit = max(1, int(limit))
    preview_ids = [str(int(row["subject_id"])) for row in ranked_source_rows[:limit]]
    if len(ranked_source_rows) > limit:
        preview_ids.append("...")
    return ",".join(preview_ids)


def _classifier_gate_thresholds(classifier_name: str) -> tuple[float, float]:
    # 中文作用: 返回当前分类器触发迁移保留所需的 ACC、F1 最小增益阈值。
    min_acc_gain = float(TRANSFER_GATE_MIN_ACC_IMPROVEMENT_BY_CLASSIFIER.get(classifier_name, 0.01))
    min_f1_gain = float(TRANSFER_GATE_MIN_F1_IMPROVEMENT_BY_CLASSIFIER.get(classifier_name, min_acc_gain))
    return min_acc_gain, min_f1_gain


def _classifier_source_filter_gap_threshold(classifier_name: str, config: TransferLearningConfig) -> float:
    """返回不同分类器在源域正类比例过滤时使用的阈值。"""
    # 中文作用: KNN 对源域分布偏差更敏感，因此给它更严格的源域过滤阈值。
    base_threshold = float(config.normalized_source_positive_ratio_gap_threshold())
    if str(classifier_name).upper() == "KNN":
        return min(base_threshold, 0.10)
    return base_threshold


def _classifier_gate_threshold_text() -> str:
    """把 LR/KNN/DT 当前使用的 gate 阈值压缩成一行文本。"""
    # 中文作用: 生成便于打印和写入报告的 gate 阈值摘要，避免实验口径不清楚。
    preview_parts: List[str] = []
    for classifier_name in CLASSIFIER_ORDER:
        min_acc_gain, min_f1_gain = _classifier_gate_thresholds(classifier_name)
        preview_parts.append(
            f"{classifier_name}(ΔACC>={min_acc_gain:.3f},ΔF1>={min_f1_gain:.3f})"
        )
    return "; ".join(preview_parts)


def _candidate_key(metrics: Dict[str, float]) -> tuple[float, float, float, float, float, float]:
    """用 ACC、F1 优先排序候选结果。"""
    acc = float(metrics["ACC"])
    f1 = float(metrics["F1"])
    bacc = float(metrics["BACC"])
    mcc = float(metrics["MCC"])
    return (
        min(acc, f1),
        acc,
        f1,
        -abs(acc - f1),
        bacc,
        mcc,
    )


def _format_params(params: Dict[str, Any]) -> str:
    """把参数字典压缩成便于打印的文本。"""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _transfer_variant_label(transfer_variant: str) -> str:
    """把迁移训练分支名称转成便于日志和报告阅读的中文标签。"""
    variant = str(transfer_variant).lower().strip()
    if variant == "enhanced_jda":
        return "增强多源域选择 + JDA"
    return "监督迁移"


def _protocol_label(protocol: str) -> str:
    """把评估协议名称转成便于报告阅读的中文标签。"""
    normalized = str(protocol).lower().strip()
    if normalized == "group_kfold":
        return "按被试分组 K 折"
    if normalized == "loso":
        return "留一被试"
    return normalized


def _split_mode_label(split_mode: str) -> str:
    """把目标域切分方式转成中文说明。"""
    normalized = str(split_mode).lower().strip()
    if normalized == "subject_equal_count_stratified":
        return "单被试等量分层切分"
    return normalized or "-"


def _alignment_method_label(alignment_method: str) -> str:
    """把域对齐方法名称转成中文标签。"""
    normalized = str(alignment_method).lower().strip()
    if normalized == "jda":
        return "JDA 联合分布自适应"
    if normalized == "none":
        return "无额外域对齐"
    return normalized or "-"


def _selection_mode_label(selection_mode: str) -> str:
    """把最终方案类型转成中文标签。"""
    normalized = str(selection_mode).lower().strip()
    if normalized == "transfer":
        return "保留迁移"
    if normalized == "target_only_gate":
        return "回退到目标域基线"
    return normalized or "-"


def _gate_reason_label(gate_reason: str) -> str:
    """把 gate 原因编码翻译成中文说明。"""
    normalized = str(gate_reason).lower().strip()
    mapping = {
        "transfer_selected": "迁移候选入围",
        "transfer_kept": "验证集增益达标，保留迁移",
        "transfer_kept_repeat_gate": "重复复核通过，保留迁移",
        "gate_small_margin": "迁移增益太小，被门控回退",
        "val_acc_f1_both_worse": "验证集 ACC 和 F1 都更差",
        "val_acc_worse": "验证集 ACC 更差",
        "val_f1_worse": "验证集 F1 更差",
        "candidate_key_lower": "验证集综合排序更低",
    }
    return mapping.get(normalized, normalized or "-")


def _reason_counter_text_to_chinese(reason_text: str) -> str:
    """把回退原因计数字符串翻成中文。"""
    text = str(reason_text).strip()
    if not text:
        return "-"
    translated_parts: List[str] = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        if ":" in item:
            raw_reason, raw_count = item.split(":", 1)
            translated_parts.append(f"{_gate_reason_label(raw_reason)}:{raw_count.strip()}")
        else:
            translated_parts.append(_gate_reason_label(item))
    return ", ".join(translated_parts)


def _yes_no_label(flag_text: str) -> str:
    """把 Y/空字符串 形式的标记转成 是/否。"""
    return "是" if str(flag_text).strip().upper() == "Y" else "否"


def _jda_config_text(config: TransferLearningConfig) -> str:
    """把当前 JDA 配置压缩成一行，方便调试输出和 Markdown 展示。"""
    return (
        f"dim={config.normalized_jda_dim()} "
        f"T={config.normalized_jda_iterations()} "
        f"n_components={config.normalized_jda_n_components()} "
        f"lambda={config.normalized_jda_lambda():.4f} "
        f"reg={config.normalized_jda_reg():.1e} "
        f"pseudo={config.normalized_jda_pseudo_labeler()}(k={config.normalized_jda_pseudo_neighbors()}) "
        f"pseudo_keep={list(config.normalized_jda_pseudo_keep_ratio_grid())} "
        f"pseudo_repeat={list(config.normalized_jda_pseudo_target_repeat_grid())} "
        f"pc_tol={config.normalized_jda_pseudo_change_tol():.4f} "
        f"mmd_tol={config.normalized_jda_mmd_delta_tol():.4f} "
        f"conf_tol={config.normalized_jda_confidence_delta_tol():.4f} "
        f"patience={config.normalized_jda_early_stop_patience()} "
        f"min_iter={config.normalized_jda_min_iterations()}"
    )


def _summarize_jda_iteration_history(iteration_history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """把 JDA 逐轮迭代日志压缩成便于排序和写报告的摘要。"""
    if not iteration_history:
        return {
            "iterations": 0.0,
            "start_mmd": 0.0,
            "end_mmd": 0.0,
            "delta_mmd": 0.0,
            "last_pseudo_change_ratio": 0.0,
            "last_confidence_mean": 0.0,
            "last_pseudo_class_ratio": 0.0,
            "best_mmd": 0.0,
            "last_stable_rounds": 0.0,
            "selected_iteration": 0.0,
            "stop_iteration": 0.0,
        }

    first_row = iteration_history[0]
    last_row = iteration_history[-1]
    selected_rows = [row for row in iteration_history if float(row.get("selected", 0.0)) > 0.5]
    selected_row = selected_rows[0] if selected_rows else last_row
    best_mmd = min(float(row.get("mmd", 0.0)) for row in iteration_history)
    return {
        "iterations": float(len(iteration_history)),
        "start_mmd": float(first_row.get("mmd", 0.0)),
        "end_mmd": float(selected_row.get("mmd", 0.0)),
        "delta_mmd": float(selected_row.get("mmd", 0.0)) - float(first_row.get("mmd", 0.0)),
        "last_pseudo_change_ratio": float(selected_row.get("pseudo_change_ratio", 0.0)),
        "last_confidence_mean": float(selected_row.get("confidence_mean", 0.0)),
        "last_pseudo_class_ratio": float(selected_row.get("pseudo_class_ratio", 0.0)),
        "best_mmd": float(best_mmd),
        "last_stable_rounds": float(selected_row.get("stable_rounds", 0.0)),
        "selected_iteration": float(selected_row.get("iteration", 0.0)),
        "stop_iteration": float(last_row.get("iteration", 0.0)),
    }


def _run_jda_alignment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target_train: np.ndarray,
    eval_blocks: Dict[str, np.ndarray],
    config: TransferLearningConfig,
    random_state: int,
    debug_target_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
    """对当前源域前缀和目标训练集执行 JDA，并把验证/测试块映射到同一潜空间。"""
    adapter = build_domain_adapter(
        method="jda",
        dim=config.normalized_jda_dim(),
        T=config.normalized_jda_iterations(),
        gamma=None,
        n_components=config.normalized_jda_n_components(),
        kernel_type="linear",
        lamb=config.normalized_jda_lambda(),
        reg=config.normalized_jda_reg(),
        pseudo_labeler=config.normalized_jda_pseudo_labeler(),
        pseudo_neighbors=config.normalized_jda_pseudo_neighbors(),
        pseudo_change_tol=config.normalized_jda_pseudo_change_tol(),
        mmd_delta_tol=config.normalized_jda_mmd_delta_tol(),
        confidence_delta_tol=config.normalized_jda_confidence_delta_tol(),
        early_stop_patience=config.normalized_jda_early_stop_patience(),
        min_iterations=config.normalized_jda_min_iterations(),
        random_state=int(random_state),
        verbose=True,
    )
    Z_source, Z_target_train = adapter.fit_transform(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target_train,
        Yt_target=debug_target_y,
    )
    aligned_eval_blocks = {
        block_name: np.asarray(adapter.transform(block_X), dtype=np.float32)
        for block_name, block_X in eval_blocks.items()
    }
    jda_diag = _summarize_jda_iteration_history(getattr(adapter, "iteration_history", []))
    pseudo_labels = getattr(adapter, "selected_target_pseudo_labels_", None)
    pseudo_confidence = getattr(adapter, "selected_target_confidence_", None)
    pseudo_bundle = {
        "pseudo_labels": None if pseudo_labels is None else np.asarray(pseudo_labels, dtype=np.int32),
        "confidence": None if pseudo_confidence is None else np.asarray(pseudo_confidence, dtype=np.float32),
        "selected_iteration": int(getattr(adapter, "selected_iteration_", 0)),
        "pseudo_labeler": str(config.normalized_jda_pseudo_labeler()),
    }
    print(
        f"[Transfer][JDAAlign] source_samples={len(y_source)} target_train={len(X_target_train)} "
        f"iterations={int(jda_diag['iterations'])} selected_iter={int(jda_diag.get('selected_iteration', 0.0))} "
        f"stop_iter={int(jda_diag.get('stop_iteration', 0.0))} start_mmd={jda_diag['start_mmd']:.4f} "
        f"end_mmd={jda_diag['end_mmd']:.4f} delta_mmd={jda_diag['delta_mmd']:+.4f} "
        f"pseudo_change={jda_diag['last_pseudo_change_ratio']:.4f} "
        f"conf_mean={jda_diag['last_confidence_mean']:.4f} "
        f"stable_rounds={int(jda_diag.get('last_stable_rounds', 0.0))}"
    )
    return Z_source, Z_target_train, aligned_eval_blocks, jda_diag, pseudo_bundle


def _strength_ratio(source_strength: float, target_strength: float) -> float:
    """计算源域与目标域的有效训练强度比例。"""
    target_strength = max(float(target_strength), 1e-6)
    return float(source_strength) / target_strength


def _resolve_source_weight_scale(
    base_source_strength: float,
    target_strength: float,
    desired_ratio: float,
) -> float:
    """根据目标比例把源域基准强度换算成训练时要使用的缩放系数。"""
    if float(base_source_strength) <= 0.0:
        return 0.0
    desired_source_strength = float(desired_ratio) * max(float(target_strength), 1e-6)
    return max(desired_source_strength / max(float(base_source_strength), 1e-6), 1e-6)


def _paper52_source_only_weight_info(source_sample_count: int) -> Dict[str, Any]:
    """返回论文 5.2 风格 JDA 分支的固定权重诊断信息。"""
    # 中文作用: 论文式 JDA 分支不再做源/目标混合加权，因此这里显式返回“源域单独训练”的说明。
    source_sample_count = max(int(source_sample_count), 0)
    return {
        "mode": "paper52_jda_source_only",
        "preview": "source_only_classifier_training",
        "min_weight": 1.0,
        "max_weight": 1.0,
        "mean_weight": 1.0,
        "total_weight": float(source_sample_count),
    }


def _select_high_confidence_pseudo_samples(
    X_target_train: np.ndarray,
    pseudo_labels: np.ndarray | None,
    pseudo_confidence: np.ndarray | None,
    keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """从目标训练块中挑选高置信伪标签样本，供 paper52 JDA 做半监督微调。"""
    # 中文作用: 按置信度从高到低选一部分目标域伪标签样本，既保留目标域信息，又尽量控制噪声扩散。
    X_target_train = np.asarray(X_target_train, dtype=np.float32)
    if pseudo_labels is None or pseudo_confidence is None or len(X_target_train) == 0:
        return (
            _empty_block(X_target_train.shape[1] if X_target_train.ndim == 2 and X_target_train.shape[1] > 0 else 0),
            np.zeros(0, dtype=np.int32),
            {
                "selected_count": 0.0,
                "selected_ratio": 0.0,
                "selected_conf_mean": 0.0,
                "selected_conf_min": 0.0,
                "selected_pos_ratio": 0.0,
            },
        )

    pseudo_labels = np.asarray(pseudo_labels, dtype=np.int32).reshape(-1)
    pseudo_confidence = np.asarray(pseudo_confidence, dtype=np.float32).reshape(-1)
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    feature_dim = int(X_target_train.shape[1])
    if keep_ratio <= 0.0:
        return (
            _empty_block(feature_dim),
            np.zeros(0, dtype=np.int32),
            {
                "selected_count": 0.0,
                "selected_ratio": 0.0,
                "selected_conf_mean": 0.0,
                "selected_conf_min": 0.0,
                "selected_pos_ratio": 0.0,
            },
        )

    keep_count = max(1, min(len(pseudo_labels), int(round(len(pseudo_labels) * keep_ratio))))
    order = np.argsort(-pseudo_confidence)
    selected = list(order[:keep_count])
    unique_all = np.unique(pseudo_labels)
    unique_selected = set(int(pseudo_labels[idx]) for idx in selected)
    for class_id in unique_all:
        if int(class_id) in unique_selected:
            continue
        class_indices = [int(idx) for idx in order if int(pseudo_labels[idx]) == int(class_id)]
        if class_indices:
            selected.append(class_indices[0])
    selected_indices = np.asarray(sorted(set(selected)), dtype=np.int32)
    selected_X = np.asarray(X_target_train[selected_indices], dtype=np.float32)
    selected_y = np.asarray(pseudo_labels[selected_indices], dtype=np.int32)
    selected_conf = np.asarray(pseudo_confidence[selected_indices], dtype=np.float32)
    diag = {
        "selected_count": float(len(selected_indices)),
        "selected_ratio": float(len(selected_indices) / max(len(pseudo_labels), 1)),
        "selected_conf_mean": float(np.mean(selected_conf)) if len(selected_conf) else 0.0,
        "selected_conf_min": float(np.min(selected_conf)) if len(selected_conf) else 0.0,
        "selected_pos_ratio": float(np.mean(selected_y)) if len(selected_y) else 0.0,
    }
    return selected_X, selected_y, diag


def _gate_reason_text(transfer_metrics: Dict[str, float], target_only_metrics: Dict[str, float]) -> str:
    """根据验证集 ACC/F1 的相对关系输出更可读的 gate 原因。"""
    transfer_acc = float(transfer_metrics["ACC"])
    transfer_f1 = float(transfer_metrics["F1"])
    target_acc = float(target_only_metrics["ACC"])
    target_f1 = float(target_only_metrics["F1"])
    if transfer_acc < target_acc and transfer_f1 < target_f1:
        return "val_acc_f1_both_worse"
    if transfer_acc < target_acc:
        return "val_acc_worse"
    if transfer_f1 < target_f1:
        return "val_f1_worse"
    return "candidate_key_lower"


def _resume_pair_key(target_unit: str, classifier_name: str) -> str:
    """为单个目标单元/分类器组合生成稳定的断点键。"""
    return f"{str(target_unit)}::{str(classifier_name)}"


def _parse_metric_snapshot_text(text: str) -> Dict[str, float]:
    """把日志中的 `(ACC=... F1=...)` 片段解析成指标字典。"""
    metrics: Dict[str, float] = {}
    for name, raw_value in re.findall(r"([A-Z0-9]+)=([+-]?\d+(?:\.\d+)?)", str(text)):
        metrics[str(name)] = float(raw_value)
    return metrics


def _resume_float(text: str, key: str, default: float = 0.0) -> float:
    """从日志文本中读取一个浮点字段。"""
    match = re.search(rf"{re.escape(key)}=([+-]?\d+(?:\.\d+)?)", str(text))
    if match is None:
        return float(default)
    return float(match.group(1))


def _resume_int(text: str, key: str, default: int = 0) -> int:
    """从日志文本中读取一个整数字段。"""
    match = re.search(rf"{re.escape(key)}=(\d+)", str(text))
    if match is None:
        return int(default)
    return int(match.group(1))


def _resume_text(text: str, pattern: str, default: str = "") -> str:
    """用正则从日志文本中抽取一个字符串片段。"""
    match = re.search(pattern, str(text))
    if match is None:
        return str(default)
    return str(match.group(1))


def _resume_close(left: float, right: float, tol: float = 1e-4) -> bool:
    """判断两个日志浮点值是否可以视为同一个候选。"""
    return abs(float(left) - float(right)) <= float(tol)


def _load_transfer_resume_pairs(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """读取迁移阶段的运行中 checkpoint。"""
    resolved_path = ensure_ai_output_path(summary_path)
    if not resolved_path.exists():
        return {}
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or str(payload.get("checkpoint_status", "")) != "running":
        return {}
    resume_pairs = payload.get("resume_pairs", {})
    if not isinstance(resume_pairs, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in resume_pairs.items()
        if isinstance(value, dict)
    }


def _save_transfer_resume_pairs(summary_path: Path, config: TransferLearningConfig, resume_pairs: Dict[str, Dict[str, Any]]) -> None:
    """把已完成目标单元写入运行中 checkpoint，便于中断后续跑。"""
    resolved_path = ensure_ai_output_path(summary_path)
    payload = {
        "checkpoint_status": "running",
        "resume_state_version": RESUME_STATE_VERSION,
        "config": asdict(config),
        "resume_pair_count": int(len(resume_pairs)),
        "resume_pairs": resume_pairs,
    }
    save_json(to_serializable(payload), resolved_path)


def _match_supervised_resume_candidate(
    candidates: Sequence[Dict[str, Any]],
    prefix_rank: int,
    prefix_size: int,
    repeat: int,
    desired_ratio: float,
    threshold: float,
    val_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """从日志里的 `SupervisedBest` 候选中找到最终 `Supervised` 所对应的那一条。"""
    for candidate in reversed(list(candidates)):
        if int(candidate.get("prefix_rank", 0)) != int(prefix_rank):
            continue
        if int(candidate.get("prefix_size", 0)) != int(prefix_size):
            continue
        if int(candidate.get("repeat", 0)) != int(repeat):
            continue
        if not _resume_close(float(candidate.get("desired_ratio", 0.0)), float(desired_ratio)):
            continue
        if not _resume_close(float(candidate.get("threshold", 0.0)), float(threshold)):
            continue
        candidate_val = dict(candidate.get("val_metrics", {}))
        if not _resume_close(float(candidate_val.get("ACC", 0.0)), float(val_metrics.get("ACC", 0.0))):
            continue
        if not _resume_close(float(candidate_val.get("F1", 0.0)), float(val_metrics.get("F1", 0.0))):
            continue
        return dict(candidate)
    return {}


def build_transfer_resume_checkpoint_from_console_log(log_path: Path, config: TransferLearningConfig) -> int:
    """把旧控制台日志恢复成迁移阶段 checkpoint，便于直接继续跑。"""
    resolved_log_path = ensure_ai_output_path(log_path)
    if not resolved_log_path.exists():
        return 0
    try:
        log_text = resolved_log_path.read_text(encoding="utf-16")
    except UnicodeError:
        log_text = resolved_log_path.read_text(encoding="utf-8", errors="ignore")

    transfer_variant = config.normalized_transfer_variant()
    alignment_method = "jda" if transfer_variant == "enhanced_jda" else "none"
    resume_pairs: Dict[str, Dict[str, Any]] = {}

    current_unit_label = ""
    current_target_subjects = ""
    current_mask_feature_counts: Dict[str, int] = {}
    current_pair: Dict[str, Any] | None = None

    def _start_pair(classifier_name: str) -> Dict[str, Any]:
        nonlocal current_pair
        current_pair = {
            "TargetUnit": current_unit_label,
            "TargetSubjects": current_target_subjects,
            "Classifier": str(classifier_name),
            "MaskFeatureCount": int(current_mask_feature_counts.get(str(classifier_name), 0)),
            "SourceFilterKept": "",
            "SourceRankPreview": "",
            "BestMMDPrefix": 0,
            "BestMMD": 0.0,
            "PrefixRankPreview": "",
            "BestMMDSources": [],
            "BestMMDWeightPreview": "",
            "mmd_prefix_sources": {},
            "mmd_prefix_mmd": {},
            "source_weight_preview_by_rank": {},
            "source_weight_mode": "inverse_mmd_ratio_align_mean1",
            "target_only_params": "",
            "target_only_threshold": 0.0,
            "target_only_val_metrics": {},
            "target_only_test_metrics": {},
            "source_only_test_metrics": {},
            "supervised_best_candidates": [],
            "raw_transfer_params": "",
            "raw_transfer_threshold": 0.0,
            "raw_transfer_val_metrics": {},
            "raw_transfer_test_metrics": {},
            "raw_transfer_prefix_rank": 0,
            "raw_transfer_prefix_size": 0,
            "raw_transfer_prefix_mmd": 0.0,
            "raw_transfer_target_repeat": 1,
            "raw_transfer_desired_ratio": 0.0,
            "raw_transfer_actual_ratio": 0.0,
            "raw_transfer_top_sources": [],
            "gate_repeat_summary": {
                "repeat_gate_passed": False,
                "mean_gain_acc": 0.0,
                "mean_gain_f1": 0.0,
            },
        }
        return current_pair

    def _active_pair_from_classifier_line(line: str) -> Dict[str, Any] | None:
        nonlocal current_pair
        classifier_match = re.search(r"classifier=([A-Z]+)", line)
        if classifier_match is None or not current_unit_label:
            return current_pair
        classifier_name = str(classifier_match.group(1))
        if current_pair is None or str(current_pair.get("Classifier", "")) != classifier_name:
            return _start_pair(classifier_name)
        return current_pair

    def _finalize_current_pair(line: str) -> None:
        nonlocal current_pair
        if current_pair is None:
            return
        if not current_pair.get("source_only_test_metrics"):
            current_pair = None
            return
        if not current_pair.get("target_only_test_metrics"):
            current_pair = None
            return
        if not current_pair.get("raw_transfer_test_metrics"):
            current_pair = None
            return

        mode = _resume_text(line, r"mode=([A-Za-z_]+)", "transfer")
        acc_gain = _resume_float(line, "gain_ACC", 0.0)
        f1_gain = _resume_float(line, "gain_F1", 0.0)
        min_acc_gain = _resume_float(line, "min_acc_gain", 0.0)
        min_f1_gain = _resume_float(line, "min_f1_gain", 0.0)
        raw_transfer_val_metrics = dict(current_pair.get("raw_transfer_val_metrics", {}))
        target_only_val_metrics = dict(current_pair.get("target_only_val_metrics", {}))

        if mode == "transfer":
            selection_mode = "transfer"
            gate_reason = "transfer_kept"
            final_metrics = dict(current_pair["raw_transfer_test_metrics"])
            final_val_metrics = raw_transfer_val_metrics
            final_threshold = float(current_pair.get("raw_transfer_threshold", 0.0))
            final_params = str(current_pair.get("raw_transfer_params", ""))
            final_prefix_rank = int(current_pair.get("raw_transfer_prefix_rank", 0))
            final_prefix_size = int(current_pair.get("raw_transfer_prefix_size", 0))
            final_desired_ratio = float(current_pair.get("raw_transfer_desired_ratio", 0.0))
            final_actual_ratio = float(current_pair.get("raw_transfer_actual_ratio", 0.0))
            final_top_sources = list(current_pair.get("raw_transfer_top_sources", []))
            final_source_weight_mode = str(current_pair.get("source_weight_mode", "inverse_mmd_ratio_align_mean1"))
            final_target_repeat = int(current_pair.get("raw_transfer_target_repeat", 1))
        elif mode == "transfer_repeat_gate":
            selection_mode = "transfer"
            gate_reason = "transfer_kept_repeat_gate"
            final_metrics = dict(current_pair["raw_transfer_test_metrics"])
            final_val_metrics = raw_transfer_val_metrics
            final_threshold = float(current_pair.get("raw_transfer_threshold", 0.0))
            final_params = str(current_pair.get("raw_transfer_params", ""))
            final_prefix_rank = int(current_pair.get("raw_transfer_prefix_rank", 0))
            final_prefix_size = int(current_pair.get("raw_transfer_prefix_size", 0))
            final_desired_ratio = float(current_pair.get("raw_transfer_desired_ratio", 0.0))
            final_actual_ratio = float(current_pair.get("raw_transfer_actual_ratio", 0.0))
            final_top_sources = list(current_pair.get("raw_transfer_top_sources", []))
            final_source_weight_mode = str(current_pair.get("source_weight_mode", "inverse_mmd_ratio_align_mean1"))
            final_target_repeat = int(current_pair.get("raw_transfer_target_repeat", 1))
        else:
            transfer_key = _candidate_key(raw_transfer_val_metrics)
            target_key = _candidate_key(target_only_val_metrics)
            if transfer_key >= target_key and (acc_gain < min_acc_gain or f1_gain < min_f1_gain):
                gate_reason = "gate_small_margin"
            else:
                gate_reason = _gate_reason_text(raw_transfer_val_metrics, target_only_val_metrics)
            selection_mode = "target_only_gate"
            final_metrics = dict(current_pair["target_only_test_metrics"])
            final_val_metrics = target_only_val_metrics
            final_threshold = float(current_pair.get("target_only_threshold", 0.0))
            final_params = str(current_pair.get("target_only_params", ""))
            final_prefix_rank = 0
            final_prefix_size = 0
            final_desired_ratio = 0.0
            final_actual_ratio = 0.0
            final_top_sources = []
            final_source_weight_mode = "target_only_gate"
            final_target_repeat = 1

        raw_transfer_metrics = dict(current_pair["raw_transfer_test_metrics"])
        source_only_metrics = dict(current_pair["source_only_test_metrics"])
        target_only_metrics = dict(current_pair["target_only_test_metrics"])
        raw_prefix_mmd = float(current_pair.get("raw_transfer_prefix_mmd", 0.0))
        repeat_summary = dict(current_pair.get("gate_repeat_summary", {}))
        subject_row = {
            "TargetUnit": str(current_pair["TargetUnit"]),
            "Classifier": str(current_pair["Classifier"]),
            "MaskFeatureCount": int(current_pair.get("MaskFeatureCount", 0)),
            "SourceOnly_ACC": f"{float(source_only_metrics.get('ACC', 0.0)):.4f}",
            "TargetOnly_ACC": f"{float(target_only_metrics.get('ACC', 0.0)):.4f}",
            "RawTransfer_ACC": f"{float(raw_transfer_metrics.get('ACC', 0.0)):.4f}",
            "RawTransfer_F1": f"{float(raw_transfer_metrics.get('F1', 0.0)):.4f}",
            "Val_ACC": f"{float(final_val_metrics.get('ACC', 0.0)):.4f}",
            "Val_F1": f"{float(final_val_metrics.get('F1', 0.0)):.4f}",
            "Transfer_ACC": f"{float(final_metrics.get('ACC', 0.0)):.4f}",
            "Transfer_F1": f"{float(final_metrics.get('F1', 0.0)):.4f}",
            "ValTestGap_ACC": f"{float(final_metrics.get('ACC', 0.0)) - float(final_val_metrics.get('ACC', 0.0)):+.4f}",
            "ValTestGap_F1": f"{float(final_metrics.get('F1', 0.0)) - float(final_val_metrics.get('F1', 0.0)):+.4f}",
            "DeltaVsSource_ACC": f"{float(final_metrics.get('ACC', 0.0)) - float(source_only_metrics.get('ACC', 0.0)):+.4f}",
            "DeltaVsTarget_ACC": f"{float(final_metrics.get('ACC', 0.0)) - float(target_only_metrics.get('ACC', 0.0)):+.4f}",
            "RawDeltaVsTarget_ACC": f"{float(raw_transfer_metrics.get('ACC', 0.0)) - float(target_only_metrics.get('ACC', 0.0)):+.4f}",
            "RawDeltaVsTarget_F1": f"{float(raw_transfer_metrics.get('F1', 0.0)) - float(target_only_metrics.get('F1', 0.0)):+.4f}",
            "PrefixSize": int(final_prefix_size),
            "PrefixRank": int(final_prefix_rank),
            "PrefixMMD": f"{raw_prefix_mmd:.4f}",
            "TargetRepeat": int(final_target_repeat),
            "PseudoKeepRatio": "0.00",
            "PseudoSelected": 0,
            "PseudoConfMean": "0.0000",
            "DesiredRatio": f"{float(final_desired_ratio):.2f}",
            "ActualRatio": f"{float(final_actual_ratio):.2f}",
            "Threshold": f"{float(final_threshold):.3f}",
            "TopSources": ",".join(str(source_id) for source_id in final_top_sources),
            "SourceWeightMode": str(final_source_weight_mode),
            "TransferVariant": str(transfer_variant),
            "AlignmentMethod": alignment_method,
            "JDAIterations": 0,
            "JDALastMMD": "0.0000",
            "JDAMMDDelta": "+0.0000",
            "JDAPseudoChange": "0.0000",
            "JDAConfidence": "0.0000",
            "SelectionMode": str(selection_mode),
            "GateReason": str(gate_reason),
            "GateRepeatPassed": "Y" if bool(repeat_summary.get("repeat_gate_passed", False)) else "",
            "GateRepeatMeanGain_ACC": f"{float(repeat_summary.get('mean_gain_acc', 0.0)):+.4f}",
            "GateRepeatMeanGain_F1": f"{float(repeat_summary.get('mean_gain_f1', 0.0)):+.4f}",
            "Params": str(final_params),
        }
        resume_pairs[_resume_pair_key(str(current_pair["TargetUnit"]), str(current_pair["Classifier"]))] = {
            "source_only_metrics": source_only_metrics,
            "target_only_metrics": target_only_metrics,
            "raw_transfer_metrics": raw_transfer_metrics,
            "final_transfer_metrics": final_metrics,
            "subject_row": subject_row,
        }
        current_pair = None

    for raw_line in log_text.splitlines():
        line = str(raw_line).strip()
        if not line:
            continue

        unit_match = re.search(
            r"^\[Transfer\] target_unit=(\S+) target_subject_ids=\[([^\]]*)\] classifier_feature_counts=(.+)$",
            line,
        )
        if unit_match is not None:
            current_unit_label = str(unit_match.group(1))
            current_target_subjects = ",".join(
                item.strip()
                for item in str(unit_match.group(2)).split(",")
                if item.strip()
            )
            current_mask_feature_counts = {
                str(name): int(raw_count)
                for name, raw_count in re.findall(r"([A-Z]+)=(\d+)", str(unit_match.group(3)))
            }
            current_pair = None
            continue

        if current_unit_label:
            if line.startswith("[Transfer][SourceFilter]") or line.startswith("[Transfer][UnitDetail]") or line.startswith("[Transfer][SourceOnlyBest]") or line.startswith("[Transfer][SourceOnly]") or line.startswith("[Transfer][TargetOnlyBest]") or line.startswith("[Transfer][TargetOnly]") or line.startswith("[Transfer][SearchPlan]") or line.startswith("[Transfer][SupervisedBest]") or line.startswith("[Transfer][Supervised]") or line.startswith("[Transfer][GateRepeatSummary]") or line.startswith("[Transfer][Gate]"):
                _active_pair_from_classifier_line(line)

        if current_pair is None:
            continue

        if line.startswith("[Transfer][SourceFilter]"):
            current_pair["SourceFilterKept"] = _resume_text(line, r"kept=(\d+/\d+)", str(current_pair.get("SourceFilterKept", "")))
            continue

        if line.startswith("[Transfer][MMDPrefix]"):
            prefix_size = _resume_int(line, "prefix", 0)
            source_ids_text = _resume_text(line, r"source_ids=(\[[^\]]*\])", "[]")
            try:
                source_ids = [int(value) for value in ast.literal_eval(source_ids_text)]
            except (ValueError, SyntaxError):
                source_ids = []
            current_pair["mmd_prefix_sources"][int(prefix_size)] = source_ids
            current_pair["mmd_prefix_mmd"][int(prefix_size)] = _resume_float(line, "prefix_mmd", 0.0)
            continue

        if line.startswith("[Transfer][MMDPrefixSearch]"):
            current_pair["PrefixRankPreview"] = _resume_text(line, r"preview=(.+)$", "")
            continue

        if line.startswith("[Transfer][SourceWeight]") and "prefix_rank=" in line:
            prefix_rank = _resume_int(line, "prefix_rank", 0)
            current_pair["source_weight_mode"] = _resume_text(line, r"mode=([A-Za-z0-9_]+)", str(current_pair.get("source_weight_mode", "")))
            current_pair["source_weight_preview_by_rank"][int(prefix_rank)] = _resume_text(line, r"preview=(.+)$", "")
            continue

        if line.startswith("[Transfer][UnitDetail]"):
            current_pair["MaskFeatureCount"] = _resume_int(line, "mask_feature_count", int(current_pair.get("MaskFeatureCount", 0)))
            current_pair["SourceFilterKept"] = _resume_text(line, r"source_filter_kept=(\d+/\d+)", str(current_pair.get("SourceFilterKept", "")))
            current_pair["SourceRankPreview"] = _resume_text(line, r"source_rank_preview=(.+?) best_prefix=", str(current_pair.get("SourceRankPreview", "")))
            current_pair["BestMMDPrefix"] = _resume_int(line, "best_prefix", int(current_pair.get("BestMMDPrefix", 0)))
            current_pair["BestMMD"] = _resume_float(line, "best_mmd", float(current_pair.get("BestMMD", 0.0)))
            best_prefix = int(current_pair.get("BestMMDPrefix", 0))
            current_pair["BestMMDSources"] = list(current_pair.get("mmd_prefix_sources", {}).get(best_prefix, []))
            current_pair["BestMMDWeightPreview"] = str(current_pair.get("source_weight_preview_by_rank", {}).get(1, ""))
            continue

        if line.startswith("[Transfer][SourceOnly]"):
            current_pair["source_only_test_metrics"] = _parse_metric_snapshot_text(_resume_text(line, r"test_metrics=\((.*?)\)", ""))
            continue

        if line.startswith("[Transfer][TargetOnlyBest]"):
            current_pair["target_only_params"] = _resume_text(line, r"params=\((.*?)\)\s+threshold=", str(current_pair.get("target_only_params", "")))
            continue

        if line.startswith("[Transfer][TargetOnly]"):
            current_pair["target_only_threshold"] = _resume_float(line, "threshold", float(current_pair.get("target_only_threshold", 0.0)))
            current_pair["target_only_test_metrics"] = _parse_metric_snapshot_text(_resume_text(line, r"test_metrics=\((.*?)\)", ""))
            current_pair["target_only_val_metrics"] = _parse_metric_snapshot_text(_resume_text(line, r"val_metrics=\((.*?)\)", ""))
            continue

        if line.startswith("[Transfer][SupervisedBest]"):
            current_pair["supervised_best_candidates"].append(
                {
                    "prefix_rank": _resume_int(line, "prefix_rank", 0),
                    "prefix_size": _resume_int(line, "prefix", 0),
                    "repeat": _resume_int(line, "repeat", 1),
                    "desired_ratio": _resume_float(line, "desired_ratio", 0.0),
                    "actual_ratio": _resume_float(line, "actual_ratio", 0.0),
                    "threshold": _resume_float(line, "threshold", 0.0),
                    "mmd": _resume_float(line, "mmd", 0.0),
                    "params": _resume_text(line, r"params=\((.*?)\)\s+val_ACC=", ""),
                    "val_metrics": {
                        "ACC": _resume_float(line, "val_ACC", 0.0),
                        "F1": _resume_float(line, "val_F1", 0.0),
                    },
                }
            )
            continue

        if line.startswith("[Transfer][Supervised]"):
            current_pair["raw_transfer_prefix_rank"] = _resume_int(line, "chosen_prefix_rank", 0)
            current_pair["raw_transfer_prefix_size"] = _resume_int(line, "chosen_prefix", 0)
            current_pair["raw_transfer_target_repeat"] = _resume_int(line, "repeat", 1)
            current_pair["raw_transfer_desired_ratio"] = _resume_float(line, "desired_ratio", 0.0)
            current_pair["raw_transfer_actual_ratio"] = _resume_float(line, "actual_ratio", 0.0)
            current_pair["raw_transfer_threshold"] = _resume_float(line, "threshold", 0.0)
            current_pair["raw_transfer_test_metrics"] = _parse_metric_snapshot_text(_resume_text(line, r"test_metrics=\((.*?)\)", ""))
            current_pair["raw_transfer_val_metrics"] = _parse_metric_snapshot_text(_resume_text(line, r"val_metrics=\((.*?)\)", ""))
            current_pair["raw_transfer_prefix_mmd"] = float(
                current_pair.get("mmd_prefix_mmd", {}).get(int(current_pair["raw_transfer_prefix_size"]), 0.0)
            )
            current_pair["raw_transfer_top_sources"] = list(
                current_pair.get("mmd_prefix_sources", {}).get(int(current_pair["raw_transfer_prefix_size"]), [])
            )
            matched_candidate = _match_supervised_resume_candidate(
                candidates=current_pair.get("supervised_best_candidates", []),
                prefix_rank=int(current_pair["raw_transfer_prefix_rank"]),
                prefix_size=int(current_pair["raw_transfer_prefix_size"]),
                repeat=int(current_pair["raw_transfer_target_repeat"]),
                desired_ratio=float(current_pair["raw_transfer_desired_ratio"]),
                threshold=float(current_pair["raw_transfer_threshold"]),
                val_metrics=dict(current_pair["raw_transfer_val_metrics"]),
            )
            current_pair["raw_transfer_params"] = str(matched_candidate.get("params", ""))
            if matched_candidate.get("mmd") is not None:
                current_pair["raw_transfer_prefix_mmd"] = float(matched_candidate.get("mmd", current_pair["raw_transfer_prefix_mmd"]))
            continue

        if line.startswith("[Transfer][GateRepeatSummary]"):
            current_pair["gate_repeat_summary"] = {
                "repeat_gate_passed": str(_resume_text(line, r"passed=(True|False)", "False")).lower() == "true",
                "mean_gain_acc": _resume_float(line, "mean_gain_ACC", 0.0),
                "mean_gain_f1": _resume_float(line, "mean_gain_F1", 0.0),
            }
            continue

        if line.startswith("[Transfer][Gate]"):
            _finalize_current_pair(line)

    if not resume_pairs:
        return 0
    _save_transfer_resume_pairs(config.summary_path, config=config, resume_pairs=resume_pairs)
    return int(len(resume_pairs))


def _empty_block(feature_dim: int) -> np.ndarray:
    """构造一个特征维度正确的空样本块。"""
    return np.zeros((0, int(feature_dim)), dtype=np.float32)


def _build_classifier(name: str, params: Dict[str, Any], random_state: int) -> Any:
    """按分类器名称和参数字典实例化模型。"""
    if name == "LR":
        return LogisticRegression(random_state=random_state, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "DT":
        return DecisionTreeClassifier(random_state=random_state, **params)
    raise ValueError(f"不支持的分类器: {name}")


def _safe_split_once(y: np.ndarray, test_size: float | int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """尽量使用分层切分；支持按比例或按精确样本数切分。"""
    y = np.asarray(y, dtype=np.int32)
    if len(y) < 2:
        raise ValueError("样本数不足，无法继续切分。")

    if isinstance(test_size, (int, np.integer)):
        test_count = int(test_size)
        test_count = min(max(test_count, 1), len(y) - 1)
        normalized_test_size: float | int = int(test_count)
    else:
        normalized_test_size = float(test_size)
        test_count = int(round(len(y) * float(normalized_test_size)))
        test_count = min(max(test_count, 1), len(y) - 1)
        normalized_test_size = float(normalized_test_size)

    class_counts = np.bincount(y)
    nonzero_counts = class_counts[class_counts > 0]
    if np.unique(y).size >= 2 and nonzero_counts.size and int(np.min(nonzero_counts)) >= 2:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=normalized_test_size,
                random_state=int(random_state),
            )
            train_idx, test_idx = next(splitter.split(np.zeros((len(y), 1), dtype=np.float32), y))
            return np.asarray(train_idx, dtype=np.int32), np.asarray(test_idx, dtype=np.int32)
        except ValueError as exc:
            print(
                f"[Transfer][SplitFallback] reason=stratified_failed "
                f"test_size={normalized_test_size} error={exc}"
            )

    rng = np.random.default_rng(int(random_state))
    permutation = rng.permutation(len(y))
    test_idx = np.sort(permutation[:test_count]).astype(np.int32)
    train_idx = np.sort(permutation[test_count:]).astype(np.int32)
    return train_idx, test_idx


def _sample_preserve_class_ratio_indices(y: np.ndarray, max_samples: int | None, random_state: int) -> np.ndarray:
    """在单个数据块内按类别比例做轻量采样。"""
    y = np.asarray(y, dtype=np.int32)
    if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y), dtype=np.int32)

    rng = np.random.default_rng(int(random_state))
    chosen: List[int] = []
    class_ids, class_counts = np.unique(y, return_counts=True)
    total_count = int(np.sum(class_counts))

    for class_id, class_count in zip(class_ids.tolist(), class_counts.tolist()):
        class_indices = np.flatnonzero(y == int(class_id))
        take = int(round(int(max_samples) * class_count / max(total_count, 1)))
        take = min(len(class_indices), max(1, take))
        chosen.extend(rng.choice(class_indices, size=take, replace=False).tolist())

    if len(chosen) > int(max_samples):
        chosen = rng.choice(np.asarray(chosen, dtype=np.int32), size=int(max_samples), replace=False).tolist()

    if len(chosen) < int(max_samples):
        remaining = np.setdiff1d(np.arange(len(y), dtype=np.int32), np.asarray(chosen, dtype=np.int32), assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(int(max_samples) - len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())

    return np.asarray(sorted(set(chosen)), dtype=np.int32)


def _sample_subject_block(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对单个被试数据块做采样并返回采样后的特征与标签。"""
    keep = _sample_preserve_class_ratio_indices(y=y, max_samples=max_samples, random_state=random_state)
    return np.asarray(X[keep], dtype=np.float32), np.asarray(y[keep], dtype=np.int32)


def _build_subject_feature_blocks(
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    selected_raw_indices: np.ndarray,
) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """把全量数据按被试切成仅保留已选特征的原始块。"""
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id in np.unique(groups).astype(np.int32).tolist():
        subject_mask = groups == int(subject_id)
        subject_blocks[int(subject_id)] = (
            np.asarray(X_raw[subject_mask][:, selected_raw_indices], dtype=np.float32),
            np.asarray(y[subject_mask], dtype=np.int32),
        )
    return subject_blocks


def _build_evaluation_units(
    y: np.ndarray,
    groups: np.ndarray,
    evaluation_protocol: str,
    cv_splits: int,
    random_state: int,
    max_units: int | None,
) -> List[Dict[str, Any]]:
    """根据协议构造目标域评估单元。"""
    protocol = str(evaluation_protocol).lower().strip()
    unique_groups = np.unique(groups).astype(np.int32)
    units: List[Dict[str, Any]] = []

    if protocol == "loso":
        for subject_id in unique_groups.tolist():
            units.append(
                {
                    "unit_id": int(subject_id),
                    "unit_label": f"subject_{int(subject_id)}",
                    "target_subject_ids": [int(subject_id)],
                }
            )
    elif protocol == "group_kfold":
        n_splits = min(int(cv_splits), int(len(unique_groups)))
        for fold_id, _train_idx, test_idx in make_outer_folds(
            y=y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
        ):
            units.append(
                {
                    "unit_id": int(fold_id),
                    "unit_label": f"fold_{int(fold_id)}",
                    "target_subject_ids": np.unique(groups[test_idx]).astype(np.int32).tolist(),
                }
            )
    else:
        raise ValueError(f"不支持的 evaluation_protocol: {evaluation_protocol!r}")

    if max_units is not None:
        units = units[: int(max_units)]

    for unit in units:
        print(
            f"[Transfer][EvalUnit] protocol={protocol} unit={unit['unit_label']} "
            f"target_subject_ids={unit['target_subject_ids']}"
        )
    return units


def _resolve_target_split_counts(total_count: int, config: TransferLearningConfig) -> tuple[int, int, int]:
    """把目标域比例换算成按被试等量切分的整数 train/val/test 样本数。"""
    # 中文作用: 先在单个目标被试内部固定 train/val/test 个数，再做分层切分，避免不同被试贡献不均。
    total_count = int(total_count)
    if total_count < 3:
        raise ValueError("单个目标被试样本数不足 3，无法同时切出 train/val/test。")

    test_count = int(round(total_count * float(config.target_test_ratio)))
    val_count = int(round(total_count * float(config.target_val_ratio)))
    test_count = min(max(test_count, 1), total_count - 2)
    val_count = min(max(val_count, 1), total_count - test_count - 1)
    train_count = total_count - test_count - val_count
    if train_count <= 0:
        val_count = max(1, val_count - 1)
        train_count = total_count - test_count - val_count
    if train_count <= 0:
        test_count = max(1, test_count - 1)
        train_count = total_count - test_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "目标域按被试等量切分失败，请检查 target_train_ratio/target_val_ratio/target_test_ratio 配置。"
        )
    return int(train_count), int(val_count), int(test_count)


def _per_subject_split_text(split_rows: Sequence[Dict[str, Any]]) -> str:
    """把按被试切分后的 train/val/test 计数压缩成便于报告展示的文本。"""
    # 中文作用: 便于快速确认每个目标被试是否按等量整数切分。
    if not split_rows:
        return "empty"
    train_values = sorted({int(row["Train"]) for row in split_rows})
    val_values = sorted({int(row["Val"]) for row in split_rows})
    test_values = sorted({int(row["Test"]) for row in split_rows})
    if len(train_values) == len(val_values) == len(test_values) == 1:
        return f"{train_values[0]}/{val_values[0]}/{test_values[0]} per_subject"
    return (
        f"train={train_values[0]}-{train_values[-1]},"
        f"val={val_values[0]}-{val_values[-1]},"
        f"test={test_values[0]}-{test_values[-1]}"
    )


def _split_target_subjects(
    target_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """把目标域按被试等量切成 train/val/test，再拼成统一的 train/val/test。"""
    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    y_train_blocks: List[np.ndarray] = []
    y_val_blocks: List[np.ndarray] = []
    y_test_blocks: List[np.ndarray] = []
    split_rows: List[Dict[str, Any]] = []

    for offset, subject_id in enumerate(target_subject_ids):
        X_subject, y_subject = subject_blocks[int(subject_id)]
        subject_train_count, subject_val_count, subject_test_count = _resolve_target_split_counts(len(y_subject), config)
        train_val_idx, test_idx = _safe_split_once(
            y=y_subject,
            test_size=int(subject_test_count),
            random_state=int(random_state + offset * 17),
        )
        train_idx, val_idx_rel = _safe_split_once(
            y=y_subject[train_val_idx],
            test_size=int(subject_val_count),
            random_state=int(random_state + offset * 17 + 1),
        )
        val_idx = train_val_idx[val_idx_rel]
        train_idx = train_val_idx[train_idx]

        X_train_subject = np.asarray(X_subject[train_idx], dtype=np.float32)
        X_val_subject = np.asarray(X_subject[val_idx], dtype=np.float32)
        X_test_subject = np.asarray(X_subject[test_idx], dtype=np.float32)
        y_train_subject = np.asarray(y_subject[train_idx], dtype=np.int32)
        y_val_subject = np.asarray(y_subject[val_idx], dtype=np.int32)
        y_test_subject = np.asarray(y_subject[test_idx], dtype=np.int32)

        train_blocks.append(X_train_subject)
        val_blocks.append(X_val_subject)
        test_blocks.append(X_test_subject)
        y_train_blocks.append(y_train_subject)
        y_val_blocks.append(y_val_subject)
        y_test_blocks.append(y_test_subject)
        split_rows.append(
            {
                "Subject": int(subject_id),
                "Train": int(len(train_idx)),
                "Val": int(len(val_idx)),
                "Test": int(len(test_idx)),
                "TrainPosRatio": f"{float(np.mean(y_train_subject)):.4f}",
                "ValPosRatio": f"{float(np.mean(y_val_subject)):.4f}",
                "TestPosRatio": f"{float(np.mean(y_test_subject)):.4f}",
            }
        )
        print(
            f"[Transfer][TargetSplit] subject={subject_id} split_mode=subject_equal_count_stratified "
            f"planned={subject_train_count}/{subject_val_count}/{subject_test_count} "
            f"actual={len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
            f"train_pos_ratio={float(np.mean(y_train_subject)):.4f} "
            f"val_pos_ratio={float(np.mean(y_val_subject)):.4f} test_pos_ratio={float(np.mean(y_test_subject)):.4f}"
        )

    per_subject_split_text = _per_subject_split_text(split_rows)
    print(
        f"[Transfer][TargetSplitSummary] subjects={list(target_subject_ids)} split_mode=subject_equal_count_stratified "
        f"per_subject_split={per_subject_split_text}"
    )
    return {
        "X_train": np.vstack(train_blocks).astype(np.float32),
        "y_train": np.hstack(y_train_blocks).astype(np.int32),
        "X_val": np.vstack(val_blocks).astype(np.float32),
        "y_val": np.hstack(y_val_blocks).astype(np.int32),
        "X_test": np.vstack(test_blocks).astype(np.float32),
        "y_test": np.hstack(y_test_blocks).astype(np.int32),
        "split_rows": split_rows,
        "split_mode": "subject_equal_count_stratified",
        "per_subject_split_text": per_subject_split_text,
    }


def _stack_source_blocks(
    source_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    sample_cap: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """按给定被试列表堆叠源域训练块，并记录采样明细。"""
    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    sample_rows: List[Dict[str, Any]] = []

    for offset, subject_id in enumerate(source_subject_ids):
        X_subject, y_subject = subject_blocks[int(subject_id)]
        X_used, y_used = _sample_subject_block(
            X=X_subject,
            y=y_subject,
            max_samples=sample_cap,
            random_state=int(random_state + offset * 19),
        )
        X_blocks.append(X_used)
        y_blocks.append(y_used)
        sample_rows.append(
            {
                "subject_id": int(subject_id),
                "total_samples": int(len(y_subject)),
                "used_samples": int(len(y_used)),
                "positive_ratio": float(np.mean(y_used)),
            }
        )

    if not X_blocks:
        raise ValueError("源域被试列表为空，无法继续迁移训练。")
    return np.vstack(X_blocks).astype(np.float32), np.hstack(y_blocks).astype(np.int32), sample_rows


def _estimate_pair_mmd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    sample_cap: int | None,
    random_state: int,
) -> float:
    """估计单个源域块与目标域训练块之间的 MMD。"""
    source_keep = _sample_preserve_class_ratio_indices(
        y=np.zeros(len(X_source), dtype=np.int32),
        max_samples=sample_cap,
        random_state=int(random_state),
    )
    target_keep = _sample_preserve_class_ratio_indices(
        y=np.zeros(len(X_target), dtype=np.int32),
        max_samples=sample_cap,
        random_state=int(random_state + 1),
    )
    Xs = np.asarray(X_source[source_keep], dtype=np.float32)
    Xt = np.asarray(X_target[target_keep], dtype=np.float32)
    gamma = estimate_gamma(
        np.vstack([Xs, Xt]),
        sample=min(160, len(Xs) + len(Xt)),
        random_state=int(random_state + 2),
    )
    return float(compute_mmd(Xs, Xt, gamma=gamma, chunk_size=min(128, max(32, len(Xs)))))


def _select_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    random_state: int,
    context_label: str,
    classifier_name: str,
    emit_debug: bool = True,
) -> tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """在验证集上扫描分类阈值，并返回 ACC/F1 更优的阈值。"""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    clipped_prob = np.clip(y_prob, 0.0, 1.0)
    positive_ratio = float(np.mean(y_true)) if len(y_true) else 0.5

    unique_scores = np.unique(np.round(clipped_prob, 6))
    candidate_values: List[float] = [0.5]
    candidate_values.extend(np.linspace(0.2, 0.8, 13).tolist())
    if str(classifier_name).upper() in {"LR", "DT"}:
        candidate_values.extend(np.linspace(0.1, 0.9, 33).tolist())
        local_low = max(0.05, positive_ratio - 0.20)
        local_high = min(0.95, positive_ratio + 0.20)
        candidate_values.extend(np.linspace(local_low, local_high, 21).tolist())
    if unique_scores.size == 1:
        candidate_values.append(float(unique_scores[0]))
    elif unique_scores.size > 1:
        candidate_values.extend(unique_scores.tolist())
        candidate_values.extend(((unique_scores[:-1] + unique_scores[1:]) / 2.0).tolist())

    thresholds = np.unique(np.clip(np.asarray(candidate_values, dtype=np.float32), 0.05, 0.95))
    if len(thresholds) > 81:
        quantiles = np.linspace(0.0, 1.0, 81)
        thresholds = np.unique(np.quantile(thresholds, quantiles).astype(np.float32))

    best_threshold = 0.5
    best_metrics: Dict[str, float] | None = None
    threshold_rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(int(random_state))
    tie_break_noise = rng.uniform(0.0, 1e-8, size=len(thresholds))

    def threshold_key(metrics: Dict[str, float], threshold: float, pred_pos_ratio: float) -> tuple[float, ...]:
        """给单个阈值候选打分，LR/DT 会额外惩罚极端正类比例漂移。"""
        acc = float(metrics["ACC"])
        f1 = float(metrics["F1"])
        bacc = float(metrics["BACC"])
        mcc = float(metrics["MCC"])
        base_score = min(acc, f1)
        ratio_gap = abs(float(pred_pos_ratio) - positive_ratio)
        if str(classifier_name).upper() in {"LR", "DT"}:
            objective = (
                0.45 * base_score
                + 0.30 * bacc
                + 0.15 * ((mcc + 1.0) / 2.0)
                + 0.10 * (1.0 - ratio_gap)
            )
            return (
                float(objective),
                base_score,
                bacc,
                mcc,
                -ratio_gap,
                -abs(acc - f1),
                -abs(float(threshold) - 0.5),
            )
        return (
            base_score,
            acc,
            f1,
            -abs(acc - f1),
            bacc,
            mcc,
            -ratio_gap,
            -abs(float(threshold) - 0.5),
        )

    for offset, threshold in enumerate(thresholds.tolist()):
        y_pred = (clipped_prob >= threshold).astype(np.int32)
        metrics = binary_metrics(y_true, y_pred, clipped_prob)
        pred_pos_ratio = float(np.mean(y_pred)) if len(y_pred) else 0.0
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "ACC": float(metrics["ACC"]),
                "F1": float(metrics["F1"]),
                "BACC": float(metrics["BACC"]),
                "MCC": float(metrics["MCC"]),
                "PredPosRatio": pred_pos_ratio,
            }
        )
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue

        current_key = threshold_key(metrics=metrics, threshold=float(threshold), pred_pos_ratio=pred_pos_ratio)
        best_pred_pos_ratio = (
            float(best_metrics["REC"]) * positive_ratio
            + (1.0 - float(best_metrics["SPE"])) * (1.0 - positive_ratio)
        )
        best_key = threshold_key(metrics=best_metrics, threshold=float(best_threshold), pred_pos_ratio=best_pred_pos_ratio)
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if current_key == best_key:
            current_margin = abs(float(threshold) - 0.5) + float(tie_break_noise[offset])
            best_margin = abs(float(best_threshold) - 0.5)
            if current_margin < best_margin:
                best_threshold = float(threshold)
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = binary_metrics(y_true, (clipped_prob >= 0.5).astype(np.int32), clipped_prob)

    preview_rows = sorted(
        threshold_rows,
        key=lambda row: threshold_key(
            metrics=row,
            threshold=float(row["threshold"]),
            pred_pos_ratio=float(row["PredPosRatio"]),
        ),
        reverse=True,
    )[:3]
    if emit_debug:
        preview_text = _format_threshold_preview(preview_rows)
        chosen_pred = (clipped_prob >= float(best_threshold)).astype(np.int32)
        chosen_pos_ratio = float(np.mean(chosen_pred)) if len(chosen_pred) else 0.0
        print(
            f"[Transfer][Threshold] context={context_label} classifier={classifier_name} chosen={best_threshold:.3f} "
            f"val_ACC={best_metrics['ACC']:.4f} val_F1={best_metrics['F1']:.4f} "
            f"val_BACC={best_metrics['BACC']:.4f} val_MCC={best_metrics['MCC']:.4f} "
            f"pred_pos_ratio={chosen_pos_ratio:.3f} true_pos_ratio={positive_ratio:.3f} top={preview_text}"
        )
    return best_threshold, best_metrics, preview_rows


def _filter_source_subject_ids_by_positive_ratio(
    classifier_name: str,
    source_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_train_y: np.ndarray,
    config: TransferLearningConfig,
) -> tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """先按目标域训练集正类比例过滤源被试，减少明显不匹配的源域噪声。"""
    # 中文作用: 在做 MMD 排序前先剔除正类比例差得太远的源被试；若过滤过严，则回补到至少可搜索 max_source_subjects 个。
    target_positive_ratio = float(np.mean(np.asarray(target_train_y, dtype=np.int32))) if len(target_train_y) else 0.0
    gap_threshold = _classifier_source_filter_gap_threshold(classifier_name=classifier_name, config=config)
    candidate_rows: List[Dict[str, Any]] = []
    for subject_id in source_subject_ids:
        _X_source, y_source = subject_blocks[int(subject_id)]
        source_positive_ratio = float(np.mean(y_source)) if len(y_source) else 0.0
        candidate_rows.append(
            {
                "SourceId": int(subject_id),
                "SourcePosRatio": float(source_positive_ratio),
                "PositiveRatioGap": abs(float(source_positive_ratio) - float(target_positive_ratio)),
                "Samples": int(len(y_source)),
            }
        )

    candidate_rows.sort(key=lambda row: (float(row["PositiveRatioGap"]), int(row["SourceId"])))
    kept_rows = [
        dict(row)
        for row in candidate_rows
        if float(row["PositiveRatioGap"]) <= gap_threshold
    ]
    min_keep = min(len(candidate_rows), max(1, int(config.max_source_subjects)))
    kept_ids = {int(row["SourceId"]) for row in kept_rows}
    if len(kept_rows) < min_keep:
        for row in candidate_rows:
            if int(row["SourceId"]) in kept_ids:
                continue
            kept_rows.append(dict(row))
            kept_ids.add(int(row["SourceId"]))
            if len(kept_rows) >= min_keep:
                break

    filtered_source_subject_ids = [int(row["SourceId"]) for row in kept_rows]
    preview = " ; ".join(
        f"{int(row['SourceId'])}:{float(row['SourcePosRatio']):.3f}/gap={float(row['PositiveRatioGap']):.3f}"
        for row in kept_rows[: min(8, len(kept_rows))]
    )
    if len(kept_rows) > 8:
        preview += " ; ..."
    print(
        f"[Transfer][SourceFilter] classifier={classifier_name} target_pos_ratio={target_positive_ratio:.4f} "
        f"gap_threshold={gap_threshold:.4f} kept={len(filtered_source_subject_ids)}/{len(candidate_rows)} "
        f"preview={preview or 'empty'}"
    )
    return filtered_source_subject_ids, candidate_rows, kept_rows


def _rank_source_subjects_by_mmd(
    source_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_train_raw: np.ndarray,
    sample_cap: int | None,
    random_state: int,
) -> List[Dict[str, Any]]:
    """按目标域训练集上的 MMD 从小到大排序源域被试。"""
    all_source_raw = np.vstack([subject_blocks[int(subject_id)][0] for subject_id in source_subject_ids]).astype(np.float32)
    search_preprocessor = TransferFeaturePreprocessor()
    search_preprocessor.fit_transform(np.vstack([all_source_raw, target_train_raw]).astype(np.float32))
    target_train = search_preprocessor.transform(target_train_raw)

    ranking_rows: List[Dict[str, Any]] = []
    for offset, subject_id in enumerate(source_subject_ids):
        X_source_raw, y_source = subject_blocks[int(subject_id)]
        X_source = search_preprocessor.transform(X_source_raw)
        score = _estimate_pair_mmd(
            X_source=X_source,
            X_target=target_train,
            sample_cap=sample_cap,
            random_state=int(random_state + offset * 23),
        )
        ranking_rows.append(
            {
                "subject_id": int(subject_id),
                "mmd": float(score),
                "samples": int(len(y_source)),
                "positive_ratio": float(np.mean(y_source)),
            }
        )

    ranking_rows.sort(key=lambda row: (row["mmd"], row["subject_id"]))
    top_preview = " | ".join(
        f"{row['subject_id']}:{row['mmd']:.4f}"
        for row in ranking_rows[: min(6, len(ranking_rows))]
    )
    print(f"[Transfer][MMDRank] top_sources={top_preview}")
    return ranking_rows


def _build_prefix_weight_diagnostics(
    prefix_candidate: Dict[str, Any],
    ranked_source_rows: Sequence[Dict[str, Any]],
    config: TransferLearningConfig,
    target_positive_ratio: float,
) -> Dict[str, Any]:
    """为单个前缀候选补充按源域 MMD 反比展开的权重诊断信息。"""
    rank_row_by_subject = {int(row["subject_id"]): row for row in ranked_source_rows}
    raw_inverse_weights: List[float] = []
    for subject_id in prefix_candidate["chosen_source_ids"]:
        subject_mmd = float(rank_row_by_subject[int(subject_id)]["mmd"])
        raw_inverse_weights.append(1.0 / max(subject_mmd, float(config.inverse_mmd_weight_epsilon)))
    inverse_weight_mean = float(np.mean(raw_inverse_weights)) if raw_inverse_weights else 1.0

    source_mmd_rows: List[Dict[str, Any]] = []
    for subject_id, raw_weight in zip(prefix_candidate["chosen_source_ids"], raw_inverse_weights):
        rank_row = rank_row_by_subject[int(subject_id)]
        inverse_weight_mean_safe = max(inverse_weight_mean, float(config.inverse_mmd_weight_epsilon))
        normalized_inverse_weight = float(raw_weight / inverse_weight_mean_safe)
        positive_ratio_gap = abs(float(rank_row["positive_ratio"]) - float(target_positive_ratio))
        ratio_alignment_weight = 1.0
        if bool(config.use_positive_ratio_source_weighting):
            ratio_alignment_weight = 1.0 / (
                1.0 + float(config.positive_ratio_weight_strength) * float(positive_ratio_gap)
            )
        source_mmd_rows.append(
            {
                "subject_id": int(subject_id),
                "mmd": float(rank_row["mmd"]),
                "samples": int(rank_row["samples"]),
                "positive_ratio": float(rank_row["positive_ratio"]),
                "positive_ratio_gap": float(positive_ratio_gap),
                "ratio_alignment_weight": float(ratio_alignment_weight),
                "inverse_mmd_weight": float(raw_weight),
                "normalized_inverse_weight": normalized_inverse_weight,
                "source_weight_mode": (
                    "inverse_mmd_ratio_align_mean1"
                    if config.use_inverse_mmd_source_weighting and config.use_positive_ratio_source_weighting
                    else "inverse_mmd_normalized_mean1"
                    if config.use_inverse_mmd_source_weighting
                    else "disabled"
                ),
            }
        )

    if source_mmd_rows:
        combined_raw_weights = [
            float(row["normalized_inverse_weight"]) * float(row["ratio_alignment_weight"])
            for row in source_mmd_rows
        ]
        combined_weight_mean = float(np.mean(combined_raw_weights)) if combined_raw_weights else 1.0
        combined_weight_mean = max(combined_weight_mean, float(config.inverse_mmd_weight_epsilon))
        for row, combined_raw_weight in zip(source_mmd_rows, combined_raw_weights):
            row["combined_subject_weight"] = float(combined_raw_weight)
            row["normalized_dynamic_weight"] = float(combined_raw_weight / combined_weight_mean)

    weight_preview = " ; ".join(
        f"{row['subject_id']}:{row.get('normalized_dynamic_weight', row['normalized_inverse_weight']):.3f}"
        for row in source_mmd_rows[: min(6, len(source_mmd_rows))]
    )
    if len(source_mmd_rows) > 6:
        weight_preview += " ; ..."

    return {
        **prefix_candidate,
        "source_mmd_rows": source_mmd_rows,
        "source_weight_preview": weight_preview or "empty",
        "source_weight_mode": (
            "inverse_mmd_ratio_align_mean1"
            if config.use_inverse_mmd_source_weighting and config.use_positive_ratio_source_weighting
            else "inverse_mmd_normalized_mean1"
            if config.use_inverse_mmd_source_weighting
            else "disabled"
        ),
    }


def _select_mmd_prefix_locally(
    ranked_source_rows: Sequence[Dict[str, Any]],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_train_raw: np.ndarray,
    target_train_y: np.ndarray,
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """按累计前缀 MMD 构造候选；JDA 分支按论文 5.2 直接取最小前缀 MMD。"""
    transfer_variant = config.normalized_transfer_variant()
    max_prefix = (
        len(ranked_source_rows)
        if transfer_variant == "enhanced_jda"
        else min(int(config.max_source_subjects), len(ranked_source_rows))
    )
    if max_prefix <= 0:
        raise ValueError("可用于累计前缀 MMD 初筛的源域数量必须大于 0。")

    prefix_candidates: List[Dict[str, Any]] = []
    prefix_rows: List[Dict[str, Any]] = []
    for prefix_size in range(1, max_prefix + 1):
        chosen_ids = [int(row["subject_id"]) for row in ranked_source_rows[:prefix_size]]
        X_source_raw, y_source, sample_rows = _stack_source_blocks(
            source_subject_ids=chosen_ids,
            subject_blocks=subject_blocks,
            sample_cap=config.source_sample_cap,
            random_state=random_state + prefix_size * 41,
        )
        preprocessor = TransferFeaturePreprocessor()
        transformed = preprocessor.fit_transform(np.vstack([X_source_raw, target_train_raw]).astype(np.float32))
        X_source = transformed[: len(X_source_raw)]
        X_target_train = transformed[len(X_source_raw):]
        prefix_mmd = _estimate_pair_mmd(
            X_source=X_source,
            X_target=X_target_train,
            sample_cap=config.mmd_sample_cap,
            random_state=random_state + prefix_size * 43,
        )
        source_pos_ratio = float(np.mean(y_source)) if len(y_source) else 0.0
        prefix_rows.append(
            {
                "PrefixSize": int(prefix_size),
                "SourceIds": ",".join(str(x) for x in chosen_ids),
                "PrefixMMD": float(prefix_mmd),
                "SourceSamples": int(len(y_source)),
                "SourcePosRatio": source_pos_ratio,
            }
        )
        print(
            f"[Transfer][MMDPrefix] prefix={prefix_size} source_ids={chosen_ids} "
            f"prefix_mmd={prefix_mmd:.4f} source_samples={len(y_source)} "
            f"source_pos_ratio={source_pos_ratio:.4f}"
        )

        prefix_candidates.append(
            {
                "prefix_size": int(prefix_size),
                "chosen_source_ids": chosen_ids,
                "prefix_mmd": float(prefix_mmd),
                "sample_rows": sample_rows,
                "preprocessor_info": preprocessor.describe(),
            }
        )

    sorted_candidates = sorted(
        prefix_candidates,
        key=lambda row: (float(row["prefix_mmd"]), int(row["prefix_size"])),
    )
    prefix_limit = config.normalized_mmd_prefix_top_k()
    if transfer_variant == "enhanced_jda":
        limited_candidates = [dict(sorted_candidates[0])]
    else:
        limited_candidates = (
            list(sorted_candidates)
            if prefix_limit is None
            else list(sorted_candidates[: min(int(prefix_limit), len(sorted_candidates))])
        )
    candidate_prefixes: List[Dict[str, Any]] = []
    for prefix_rank, candidate in enumerate(limited_candidates, start=1):
        candidate_with_diag = _build_prefix_weight_diagnostics(
            prefix_candidate=dict(candidate),
            ranked_source_rows=ranked_source_rows,
            config=config,
            target_positive_ratio=float(np.mean(target_train_y)),
        )
        candidate_with_diag["prefix_rank"] = int(prefix_rank)
        candidate_prefixes.append(candidate_with_diag)

    if not candidate_prefixes:
        raise RuntimeError("累计前缀 MMD 初筛没有得到有效候选。")

    search_prefix_sizes = {int(candidate["prefix_size"]) for candidate in candidate_prefixes}
    formatted_prefix_rows = [
        {
            "PrefixSize": int(row["PrefixSize"]),
            "SourceIds": str(row["SourceIds"]),
            "PrefixMMD": f"{float(row['PrefixMMD']):.4f}",
            "SourceSamples": int(row["SourceSamples"]),
            "SourcePosRatio": f"{float(row['SourcePosRatio']):.4f}",
            "InSearchPool": "Y" if int(row["PrefixSize"]) in search_prefix_sizes else "",
        }
        for row in prefix_rows
    ]
    prefix_rank_preview = " ; ".join(
        f"rank{int(candidate['prefix_rank'])}:p{int(candidate['prefix_size'])}/mmd={float(candidate['prefix_mmd']):.4f}"
        for candidate in candidate_prefixes
    )
    if transfer_variant == "enhanced_jda":
        search_pool_text = "paper52_local_min_prefix"
        search_note = "JDA 分支按论文 5.2 直接选累计前缀里 MMD 最小的源域组合"
    else:
        search_pool_text = "all_prefixes" if prefix_limit is None else f"top_{int(prefix_limit)}_prefixes"
        search_note = "后续将由每个分类器在验证集上分别复排这些累计前缀"
    print(
        f"[Transfer][MMDPrefixSearch] search_pool={search_pool_text} "
        f"candidate_count={len(candidate_prefixes)}/{len(sorted_candidates)} "
        f"preview={prefix_rank_preview} note={search_note}"
    )
    for candidate in candidate_prefixes:
        print(
            f"[Transfer][SourceWeight] prefix_rank={candidate['prefix_rank']} "
            f"prefix={candidate['prefix_size']} mode={candidate['source_weight_mode']} "
            f"preview={candidate['source_weight_preview']}"
        )

    best_mmd_candidate = candidate_prefixes[0]
    return {
        "best_mmd_prefix": best_mmd_candidate,
        "candidate_prefixes": candidate_prefixes,
        "prefix_rows": formatted_prefix_rows,
        "prefix_rank_preview": prefix_rank_preview,
        "candidate_prefix_limit": None if prefix_limit is None else int(prefix_limit),
    }

def _prepare_training_payload(
    classifier_name: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    target_repeat: int,
    source_sample_weight: np.ndarray | None = None,
    source_weight_scale: float = 1.0,
    sampling_random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict[str, float]]:
    """按分类器类型构造训练输入、样本权重和实际生效的强度统计。"""
    feature_dim = int(X_source.shape[1] if len(X_source) else X_target.shape[1])
    X_source = np.asarray(X_source, dtype=np.float32).reshape(-1, feature_dim)
    y_source = np.asarray(y_source, dtype=np.int32).reshape(-1)
    X_target = np.asarray(X_target, dtype=np.float32).reshape(-1, feature_dim)
    y_target = np.asarray(y_target, dtype=np.int32).reshape(-1)
    source_sample_weight = None if source_sample_weight is None else np.asarray(source_sample_weight, dtype=np.float32).reshape(-1)
    target_repeat = max(int(target_repeat), 1)
    source_weight_scale = float(max(source_weight_scale, 1e-6))

    if classifier_name == "KNN":
        rng = np.random.default_rng(int(sampling_random_state))
        if len(X_source):
            if source_sample_weight is not None and np.sum(source_sample_weight) > 0:
                weighted_source_count = max(1, int(round(len(y_source) * source_weight_scale)))
                sampling_prob = np.asarray(source_sample_weight, dtype=np.float32)
                sampling_prob = sampling_prob / np.sum(sampling_prob)
                take = rng.choice(len(y_source), size=weighted_source_count, replace=True, p=sampling_prob)
                X_source = np.asarray(X_source[take], dtype=np.float32)
                y_source = np.asarray(y_source[take], dtype=np.int32)
            elif source_weight_scale < 0.999:
                keep_count = max(1, min(len(y_source), int(round(len(y_source) * source_weight_scale))))
                keep = _sample_preserve_class_ratio_indices(
                    y=y_source,
                    max_samples=keep_count,
                    random_state=int(sampling_random_state),
                )
                X_source = np.asarray(X_source[keep], dtype=np.float32)
                y_source = np.asarray(y_source[keep], dtype=np.int32)
            elif source_weight_scale > 1.001:
                full_repeats = max(1, int(np.floor(source_weight_scale)))
                residual_scale = float(source_weight_scale - full_repeats)
                X_source_blocks = [X_source.copy() for _ in range(full_repeats)]
                y_source_blocks = [y_source.copy() for _ in range(full_repeats)]
                residual_count = int(round(len(y_source) * residual_scale))
                if residual_count > 0:
                    take = rng.choice(len(y_source), size=residual_count, replace=True)
                    X_source_blocks.append(np.asarray(X_source[take], dtype=np.float32))
                    y_source_blocks.append(np.asarray(y_source[take], dtype=np.int32))
                X_source = np.vstack(X_source_blocks).astype(np.float32)
                y_source = np.hstack(y_source_blocks).astype(np.int32)
        X_blocks: List[np.ndarray] = []
        y_blocks: List[np.ndarray] = []
        if len(X_source):
            X_blocks.append(X_source)
            y_blocks.append(y_source)
        if len(X_target):
            for _ in range(target_repeat):
                X_blocks.append(X_target)
                y_blocks.append(y_target)
        target_strength = float(len(y_target) * target_repeat)
        source_strength = float(len(y_source))
        balance_info = {
            "effective_source_strength": source_strength,
            "effective_target_strength": target_strength,
            "effective_source_target_ratio": _strength_ratio(source_strength, target_strength) if target_strength > 0 else 0.0,
            "effective_source_samples": float(len(y_source)),
            "effective_target_samples": target_strength,
        }
        return (
            np.vstack(X_blocks).astype(np.float32),
            np.hstack(y_blocks).astype(np.int32),
            None,
            balance_info,
        )

    X_train = np.vstack([block for block in [X_source, X_target] if len(block)]).astype(np.float32)
    y_train = np.hstack([block for block in [y_source, y_target] if len(block)]).astype(np.int32)
    sample_weight_parts: List[np.ndarray] = []
    if len(y_source):
        if source_sample_weight is not None:
            if len(source_sample_weight) != len(y_source):
                raise ValueError("源域样本权重长度与源域样本数不一致。")
            sample_weight_parts.append((source_sample_weight.astype(np.float32) * source_weight_scale).astype(np.float32))
        else:
            sample_weight_parts.append(np.full(len(y_source), source_weight_scale, dtype=np.float32))
    if len(y_target):
        sample_weight_parts.append(np.full(len(y_target), float(target_repeat), dtype=np.float32))
    sample_weight = np.hstack(sample_weight_parts).astype(np.float32) if sample_weight_parts else None
    source_strength = 0.0
    target_strength = 0.0
    if len(y_source):
        if source_sample_weight is not None:
            source_strength = float(np.sum(source_sample_weight.astype(np.float32) * source_weight_scale))
        else:
            source_strength = float(len(y_source) * source_weight_scale)
    if len(y_target):
        target_strength = float(len(y_target) * target_repeat)
    balance_info = {
        "effective_source_strength": source_strength,
        "effective_target_strength": target_strength,
        "effective_source_target_ratio": _strength_ratio(source_strength, target_strength) if target_strength > 0 else 0.0,
        "effective_source_samples": float(len(y_source)),
        "effective_target_samples": float(len(y_target) * target_repeat),
    }
    return X_train, y_train, sample_weight, balance_info


def _build_source_sample_weights(
    sample_rows: Sequence[Dict[str, Any]],
    source_mmd_rows: Sequence[Dict[str, Any]],
    enabled: bool,
) -> tuple[np.ndarray | None, Dict[str, Any]]:
    """把按子源域定义的权重展开成逐样本权重，并输出诊断摘要。"""
    if not enabled or not source_mmd_rows:
        info = {
            "mode": "disabled",
            "preview": "disabled",
            "min_weight": 1.0,
            "max_weight": 1.0,
            "mean_weight": 1.0,
            "total_weight": float(sum(int(row["used_samples"]) for row in sample_rows)),
        }
        return None, info

    weight_by_subject = {
        int(row["subject_id"]): float(row.get("normalized_dynamic_weight", row.get("normalized_inverse_weight", 1.0)))
        for row in source_mmd_rows
    }
    weight_mode = str(source_mmd_rows[0].get("source_weight_mode", "")) if source_mmd_rows else ""
    if not weight_mode:
        has_dynamic_weight = any("normalized_dynamic_weight" in row for row in source_mmd_rows)
        weight_mode = "inverse_mmd_ratio_align_mean1" if has_dynamic_weight else "inverse_mmd_normalized_mean1"
    weight_blocks: List[np.ndarray] = []
    preview_items: List[str] = []
    for sample_row in sample_rows:
        subject_id = int(sample_row["subject_id"])
        used_samples = int(sample_row["used_samples"])
        subject_weight = float(weight_by_subject.get(subject_id, 1.0))
        preview_items.append(f"{subject_id}:{subject_weight:.3f}")
        weight_blocks.append(np.full(used_samples, subject_weight, dtype=np.float32))

    if not weight_blocks:
        info = {
            "mode": weight_mode,
            "preview": "empty",
            "min_weight": 1.0,
            "max_weight": 1.0,
            "mean_weight": 1.0,
            "total_weight": 0.0,
        }
        return None, info

    sample_weight = np.hstack(weight_blocks).astype(np.float32)
    preview = " | ".join(preview_items[: min(6, len(preview_items))])
    if len(preview_items) > 6:
        preview += " | ..."
    info = {
        "mode": weight_mode,
        "preview": preview,
        "min_weight": float(np.min(sample_weight)),
        "max_weight": float(np.max(sample_weight)),
        "mean_weight": float(np.mean(sample_weight)),
        "total_weight": float(np.sum(sample_weight)),
    }
    print(
        f"[Transfer][SourceWeight] mode={info['mode']} total={info['total_weight']:.2f} "
        f"min={info['min_weight']:.3f} max={info['max_weight']:.3f} "
        f"mean={info['mean_weight']:.3f} preview={info['preview']}"
    )
    return sample_weight, info


def _fit_predict_probability(
    classifier_name: str,
    params: Dict[str, Any],
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_eval: np.ndarray,
    target_repeat: int,
    random_state: int,
    source_sample_weight: np.ndarray | None = None,
    source_weight_scale: float = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    """拟合分类器并输出验证/测试概率与训练强度统计。"""
    model = _build_classifier(classifier_name, params, random_state=random_state)
    X_train, y_train, sample_weight, balance_info = _prepare_training_payload(
        classifier_name=classifier_name,
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        y_target=y_target,
        target_repeat=target_repeat,
        source_sample_weight=source_sample_weight,
        source_weight_scale=source_weight_scale,
        sampling_random_state=int(random_state + 17),
    )
    if classifier_name in {"LR", "DT"} and sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return np.asarray(positive_class_scores(model, X_eval), dtype=np.float32), balance_info


def _run_source_only_baseline(
    classifier_name: str,
    source_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """运行只使用源域训练的 baseline，并在目标域验证集上调参。"""
    X_source_raw, y_source, sample_rows = _stack_source_blocks(
        source_subject_ids=source_subject_ids,
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state,
    )
    preprocessor = TransferFeaturePreprocessor()
    X_source = preprocessor.fit_transform(X_source_raw)
    X_val = preprocessor.transform(target_split["X_val"])
    X_test = preprocessor.transform(target_split["X_test"])
    feature_dim = int(X_source.shape[1])
    search_profile = _classifier_search_profile(classifier_name, config)
    best_candidate: Dict[str, Any] | None = None

    for candidate_id, params in enumerate(search_profile["param_grid"], start=1):
        prob_val, balance_info = _fit_predict_probability(
            classifier_name=classifier_name,
            params=params,
            X_source=X_source,
            y_source=y_source,
            X_target=_empty_block(feature_dim),
            y_target=np.zeros(0, dtype=np.int32),
            X_eval=X_val,
            target_repeat=1,
            random_state=random_state + candidate_id * 31,
        )
        threshold, val_metrics, _ = _select_best_threshold(
            y_true=target_split["y_val"],
            y_prob=prob_val,
            random_state=random_state + candidate_id * 31 + 5,
            context_label=f"source_only:{classifier_name}:{candidate_id}",
            classifier_name=classifier_name,
            emit_debug=False,
        )
        candidate = {
            "params": dict(params),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "balance_info": balance_info,
        }
        if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
            best_candidate = candidate
            print(
                f"[Transfer][SourceOnlyBest] classifier={classifier_name} params=({_format_params(params)}) "
                f"threshold={threshold:.3f} val_ACC={val_metrics['ACC']:.4f} val_F1={val_metrics['F1']:.4f} "
                f"src_tgt_ratio={balance_info['effective_source_target_ratio']:.3f}"
            )

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} ? source-only baseline ?????????")

    prob_test, balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=X_source,
        y_source=y_source,
        X_target=_empty_block(feature_dim),
        y_target=np.zeros(0, dtype=np.int32),
        X_eval=X_test,
        target_repeat=1,
        random_state=random_state + 701,
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    print(
        f"[Transfer][SourceOnly] classifier={classifier_name} threshold={best_candidate['threshold']:.3f} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])}) "
        f"src_tgt_ratio={balance_info['effective_source_target_ratio']:.3f}"
    )
    return {
        "params": best_candidate["params"],
        "threshold": float(best_candidate["threshold"]),
        "val_metrics": best_candidate["val_metrics"],
        "test_metrics": test_metrics,
        "balance_info": balance_info,
        "sample_rows": sample_rows,
        "preprocessor_info": preprocessor.describe(),
    }

def _run_target_only_baseline(
    classifier_name: str,
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """运行只使用目标域训练的 baseline，并在目标域验证集上调参。"""
    search_preprocessor = TransferFeaturePreprocessor()
    X_train = search_preprocessor.fit_transform(target_split["X_train"])
    X_val = search_preprocessor.transform(target_split["X_val"])
    feature_dim = int(X_train.shape[1])
    search_profile = _classifier_search_profile(classifier_name, config)
    best_candidate: Dict[str, Any] | None = None

    for candidate_id, params in enumerate(search_profile["param_grid"], start=1):
        prob_val, balance_info = _fit_predict_probability(
            classifier_name=classifier_name,
            params=params,
            X_source=_empty_block(feature_dim),
            y_source=np.zeros(0, dtype=np.int32),
            X_target=X_train,
            y_target=target_split["y_train"],
            X_eval=X_val,
            target_repeat=1,
            random_state=random_state + candidate_id * 37,
        )
        threshold, val_metrics, _ = _select_best_threshold(
            y_true=target_split["y_val"],
            y_prob=prob_val,
            random_state=random_state + candidate_id * 37 + 5,
            context_label=f"target_only:{classifier_name}:{candidate_id}",
            classifier_name=classifier_name,
            emit_debug=False,
        )
        candidate = {
            "params": dict(params),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "balance_info": balance_info,
        }
        if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
            best_candidate = candidate
            print(
                f"[Transfer][TargetOnlyBest] classifier={classifier_name} params=({_format_params(params)}) "
                f"threshold={threshold:.3f} val_ACC={val_metrics['ACC']:.4f} val_F1={val_metrics['F1']:.4f} "
                f"src_tgt_ratio={balance_info['effective_source_target_ratio']:.3f}"
            )

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} ? target-only baseline ?????????")

    final_preprocessor = TransferFeaturePreprocessor()
    X_train_val = np.vstack([target_split["X_train"], target_split["X_val"]]).astype(np.float32)
    y_train_val = np.hstack([target_split["y_train"], target_split["y_val"]]).astype(np.int32)
    X_train_val_processed = final_preprocessor.fit_transform(X_train_val)
    X_test = final_preprocessor.transform(target_split["X_test"])
    prob_test, balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=_empty_block(feature_dim),
        y_source=np.zeros(0, dtype=np.int32),
        X_target=X_train_val_processed,
        y_target=y_train_val,
        X_eval=X_test,
        target_repeat=1,
        random_state=random_state + 709,
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    print(
        f"[Transfer][TargetOnly] classifier={classifier_name} threshold={best_candidate['threshold']:.3f} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])}) "
        f"src_tgt_ratio={balance_info['effective_source_target_ratio']:.3f}"
    )
    return {
        "params": best_candidate["params"],
        "threshold": float(best_candidate["threshold"]),
        "val_metrics": best_candidate["val_metrics"],
        "test_metrics": test_metrics,
        "balance_info": balance_info,
        "preprocessor_info": final_preprocessor.describe(),
    }

def _run_supervised_transfer(
    classifier_name: str,
    mmd_prefix_selection: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
    target_only_val_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """在累计前缀候选池中搜索各分类器自己的最优迁移配置。"""
    search_profile = _classifier_search_profile(classifier_name, config)
    repeat_grid = tuple(int(value) for value in search_profile["target_repeat_grid"])
    source_target_ratio_grid = tuple(float(value) for value in search_profile["source_target_ratio_grid"])
    best_candidate: Dict[str, Any] | None = None
    best_non_dominated_candidate: Dict[str, Any] | None = None
    candidate_prefixes = [dict(candidate) for candidate in mmd_prefix_selection.get("candidate_prefixes", [])]
    if not candidate_prefixes:
        raise RuntimeError("累计前缀候选池为空，无法继续监督迁移搜索。")
    total_candidates = 0
    nondominated_candidates = 0
    print(
        f"[Transfer][SearchPlan] classifier={classifier_name} prefix_candidates={len(candidate_prefixes)} "
        f"prefix_search={search_profile['search_prefix_limit']} "
        f"params={len(search_profile['param_grid'])} repeats={list(repeat_grid)} "
        f"ratios={list(source_target_ratio_grid)}"
    )

    for prefix_candidate in candidate_prefixes:
        chosen_ids = [int(subject_id) for subject_id in prefix_candidate["chosen_source_ids"]]
        prefix_size = int(prefix_candidate["prefix_size"])
        prefix_rank = int(prefix_candidate.get("prefix_rank", 0))
        prefix_mmd = float(prefix_candidate["prefix_mmd"])

        X_source_raw, y_source, sample_rows = _stack_source_blocks(
            source_subject_ids=chosen_ids,
            subject_blocks=subject_blocks,
            sample_cap=config.source_sample_cap,
            random_state=random_state + prefix_size * 41 + prefix_rank * 131,
        )
        preprocessor = TransferFeaturePreprocessor()
        X_source = preprocessor.fit_transform(np.vstack([X_source_raw, target_split["X_train"]]).astype(np.float32))
        X_source = X_source[: len(X_source_raw)]
        X_target_train = preprocessor.transform(target_split["X_train"])
        X_target_val = preprocessor.transform(target_split["X_val"])
        source_sample_weight, source_weight_info = _build_source_sample_weights(
            sample_rows=sample_rows,
            source_mmd_rows=prefix_candidate.get("source_mmd_rows", []),
            enabled=bool(config.use_inverse_mmd_source_weighting),
        )
        if source_sample_weight is not None:
            base_source_strength = float(source_weight_info["total_weight"])
        else:
            base_source_strength = float(len(y_source))
        print(
            f"[Transfer][WeightBalance] classifier={classifier_name} prefix_rank={prefix_rank} "
            f"prefix={prefix_size} base_source_strength={base_source_strength:.2f} "
            f"source_weight_mode={source_weight_info['mode']} source_preview={source_weight_info['preview']}"
        )

        for params_id, params in enumerate(search_profile["param_grid"], start=1):
            for repeat in repeat_grid:
                target_strength = float(len(target_split["y_train"]) * int(repeat))
                for desired_ratio in source_target_ratio_grid:
                    total_candidates += 1
                    source_scale = _resolve_source_weight_scale(
                        base_source_strength=base_source_strength,
                        target_strength=target_strength,
                        desired_ratio=float(desired_ratio),
                    )
                    fit_random_state = (
                        random_state
                        + prefix_size * 101
                        + prefix_rank * 29
                        + params_id * 17
                        + int(repeat) * 11
                        + int(round(float(desired_ratio) * 100))
                    )
                    prob_val, balance_info = _fit_predict_probability(
                        classifier_name=classifier_name,
                        params=params,
                        X_source=X_source,
                        y_source=y_source,
                        X_target=X_target_train,
                        y_target=target_split["y_train"],
                        X_eval=X_target_val,
                        target_repeat=int(repeat),
                        random_state=fit_random_state,
                        source_sample_weight=source_sample_weight,
                        source_weight_scale=float(source_scale),
                    )
                    threshold, val_metrics, _ = _select_best_threshold(
                        y_true=target_split["y_val"],
                        y_prob=prob_val,
                        random_state=fit_random_state + 3,
                        context_label=(
                            f"supervised:{classifier_name}:rank{prefix_rank}:prefix{prefix_size}:repeat{repeat}:"
                            f"ratio{float(desired_ratio):.2f}:params{params_id}"
                        ),
                        classifier_name=classifier_name,
                        emit_debug=False,
                    )
                    val_vs_target_acc = (
                        float(val_metrics["ACC"]) - float(target_only_val_metrics["ACC"])
                        if target_only_val_metrics is not None
                        else 0.0
                    )
                    val_vs_target_f1 = (
                        float(val_metrics["F1"]) - float(target_only_val_metrics["F1"])
                        if target_only_val_metrics is not None
                        else 0.0
                    )
                    candidate = {
                        "classifier_name": classifier_name,
                        "prefix_rank": prefix_rank,
                        "prefix_size": int(prefix_size),
                        "chosen_source_ids": chosen_ids,
                        "params": dict(params),
                        "target_repeat": int(repeat),
                        "desired_source_target_ratio": float(desired_ratio),
                        "effective_source_target_ratio": float(balance_info["effective_source_target_ratio"]),
                        "source_weight_scale": float(source_scale),
                        "threshold": float(threshold),
                        "val_metrics": val_metrics,
                        "prefix_mmd": float(prefix_mmd),
                        "sample_rows": sample_rows,
                        "preprocessor_info": preprocessor.describe(),
                        "balance_info": dict(balance_info),
                        "source_weight_info": dict(source_weight_info),
                        "source_mmd_rows": list(prefix_candidate.get("source_mmd_rows", [])),
                        "prefix_rank_preview": str(mmd_prefix_selection.get("prefix_rank_preview", "")),
                    }
                    if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
                        best_candidate = candidate
                        print(
                            f"[Transfer][SupervisedBest] classifier={classifier_name} prefix_rank={prefix_rank} "
                            f"prefix={prefix_size} repeat={repeat} desired_ratio={float(desired_ratio):.2f} "
                            f"source_scale={source_scale:.3f} actual_ratio={balance_info['effective_source_target_ratio']:.3f} "
                            f"mmd={prefix_mmd:.4f} threshold={threshold:.3f} "
                            f"srcW={balance_info['effective_source_strength']:.2f} "
                            f"tgtW={balance_info['effective_target_strength']:.2f} "
                            f"params=({_format_params(params)}) val_ACC={val_metrics['ACC']:.4f} "
                            f"val_F1={val_metrics['F1']:.4f} dValTarget_ACC={val_vs_target_acc:+.4f} "
                            f"dValTarget_F1={val_vs_target_f1:+.4f}"
                        )
                    if target_only_val_metrics is not None:
                        dominated_by_target = (
                            float(val_metrics["ACC"]) < float(target_only_val_metrics["ACC"])
                            and float(val_metrics["F1"]) < float(target_only_val_metrics["F1"])
                        )
                        if not dominated_by_target and (
                            best_non_dominated_candidate is None
                            or _candidate_key(val_metrics) > _candidate_key(best_non_dominated_candidate["val_metrics"])
                        ):
                            best_non_dominated_candidate = candidate
                            nondominated_candidates += 1

    print(
        f"[Transfer][SearchSummary] classifier={classifier_name} total_candidates={total_candidates} "
        f"nondominated_updates={nondominated_candidates}"
    )

    if best_non_dominated_candidate is not None:
        best_candidate = best_non_dominated_candidate

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} 的监督迁移没有得到有效候选。")

    X_source_final_raw, y_source_final, sample_rows = _stack_source_blocks(
        source_subject_ids=best_candidate["chosen_source_ids"],
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state + 911,
    )
    source_sample_weight_final, source_weight_info_final = _build_source_sample_weights(
        sample_rows=sample_rows,
        source_mmd_rows=best_candidate.get("source_mmd_rows", []),
        enabled=bool(config.use_inverse_mmd_source_weighting),
    )
    X_target_train_val_raw = np.vstack([target_split["X_train"], target_split["X_val"]]).astype(np.float32)
    y_target_train_val = np.hstack([target_split["y_train"], target_split["y_val"]]).astype(np.int32)
    final_preprocessor = TransferFeaturePreprocessor()
    X_source_and_target = final_preprocessor.fit_transform(
        np.vstack([X_source_final_raw, X_target_train_val_raw]).astype(np.float32)
    )
    X_source_final = X_source_and_target[: len(X_source_final_raw)]
    X_target_train_val = X_source_and_target[len(X_source_final_raw):]
    X_test = final_preprocessor.transform(target_split["X_test"])
    if source_sample_weight_final is not None:
        base_source_strength_final = float(source_weight_info_final["total_weight"])
    else:
        base_source_strength_final = float(len(y_source_final))
    target_strength_final = float(len(y_target_train_val) * int(best_candidate["target_repeat"]))
    final_source_scale = _resolve_source_weight_scale(
        base_source_strength=base_source_strength_final,
        target_strength=target_strength_final,
        desired_ratio=float(best_candidate.get("desired_source_target_ratio", 0.0)),
    )
    prob_test, final_balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=X_source_final,
        y_source=y_source_final,
        X_target=X_target_train_val,
        y_target=y_target_train_val,
        X_eval=X_test,
        target_repeat=int(best_candidate["target_repeat"]),
        random_state=random_state + 919,
        source_sample_weight=source_sample_weight_final,
        source_weight_scale=float(final_source_scale),
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    val_test_gap_acc = float(test_metrics["ACC"]) - float(best_candidate["val_metrics"]["ACC"])
    val_test_gap_f1 = float(test_metrics["F1"]) - float(best_candidate["val_metrics"]["F1"])
    print(
        f"[Transfer][Supervised] classifier={classifier_name} chosen_prefix_rank={best_candidate.get('prefix_rank', 0)} "
        f"chosen_prefix={best_candidate['prefix_size']} repeat={best_candidate['target_repeat']} "
        f"desired_ratio={best_candidate.get('desired_source_target_ratio', 0.0):.2f} "
        f"final_source_scale={final_source_scale:.3f} "
        f"actual_ratio={final_balance_info['effective_source_target_ratio']:.3f} "
        f"threshold={best_candidate['threshold']:.3f} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])}) "
        f"gap_ACC={val_test_gap_acc:+.4f} gap_F1={val_test_gap_f1:+.4f}"
    )
    return {
        **best_candidate,
        "transfer_variant": "supervised",
        "alignment_method": "none",
        "search_source_weight_scale": float(best_candidate.get("source_weight_scale", 1.0)),
        "source_weight_scale": float(final_source_scale),
        "test_metrics": test_metrics,
        "final_preprocessor_info": final_preprocessor.describe(),
        "final_sample_rows": sample_rows,
        "prefix_rows": list(mmd_prefix_selection["prefix_rows"]),
        "candidate_prefix_count": int(len(candidate_prefixes)),
        "final_balance_info": dict(final_balance_info),
        "source_weight_info": dict(source_weight_info_final),
        "jda_diag": {},
        "final_jda_diag": {},
        "selection_mode": "transfer",
        "gate_reason": "transfer_selected",
    }

def _run_twod_cnn_transfer(
    classifier_name: str,
    mmd_prefix_selection: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
    target_only_val_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """使用 TwoD_CNN 作为特征提取器的迁移学习分支。"""
    from AI.domain_adaptation import build_domain_adapter

    search_profile = _classifier_search_profile(classifier_name, config)
    repeat_grid = tuple(int(value) for value in search_profile["target_repeat_grid"])
    source_target_ratio_grid = tuple(float(value) for value in search_profile["source_target_ratio_grid"])
    candidate_prefixes = [dict(candidate) for candidate in mmd_prefix_selection.get("candidate_prefixes", [])]
    if not candidate_prefixes:
        raise RuntimeError("累计前缀候选池为空，无法继续监督迁移搜索。")

    best_candidate: Dict[str, Any] | None = None
    total_candidates = 0

    print(
        f"[Transfer][SearchPlan] classifier={classifier_name} variant=twod_cnn "
        f"prefix_candidates={len(candidate_prefixes)} "
        f"encoding_dim={config.normalized_twod_cnn_encoding_dim()} "
        f"epochs={config.normalized_twod_cnn_epochs()} batch_size={config.normalized_twod_cnn_batch_size()}"
    )

    for prefix_candidate in candidate_prefixes:
        chosen_ids = [int(subject_id) for subject_id in prefix_candidate["chosen_source_ids"]]
        prefix_size = int(prefix_candidate["prefix_size"])
        prefix_rank = int(prefix_candidate.get("prefix_rank", 0))
        prefix_mmd = float(prefix_candidate["prefix_mmd"])

        # 堆叠源域数据
        X_source_raw, y_source, sample_rows = _stack_source_blocks(
            source_subject_ids=chosen_ids,
            subject_blocks=subject_blocks,
            sample_cap=config.source_sample_cap,
            random_state=random_state + prefix_size * 41 + prefix_rank * 131,
        )
        # 预处理
        preprocessor = TransferFeaturePreprocessor()
        X_source_and_target = preprocessor.fit_transform(
            np.vstack([X_source_raw, target_split["X_train"]]).astype(np.float32)
        )
        X_source = X_source_and_target[: len(X_source_raw)]
        X_target_train = X_source_and_target[len(X_source_raw):]
        X_target_val = preprocessor.transform(target_split["X_val"])

        # ========== 核心：调用 TwoDCNN 进行特征提取 ==========
        adapter = build_domain_adapter(
            method='twod_cnn',
            input_dim=X_source.shape[1],
            encoding_dim=config.normalized_twod_cnn_encoding_dim(),
            epochs=config.normalized_twod_cnn_epochs(),
            batch_size=config.normalized_twod_cnn_batch_size(),
            verbose=True,
        )
        Z_source, Z_target_train = adapter.fit_transform(X_source, y_source, X_target_train)
        Z_target_val = adapter.transform(X_target_val)
        # =================================================

        # 在提取后的特征上搜索分类器超参数
        for params in search_profile["param_grid"]:
            for repeat in repeat_grid:
                for desired_ratio in source_target_ratio_grid:
                    total_candidates += 1
                    prob_val, balance_info = _fit_predict_probability(
                        classifier_name=classifier_name,
                        params=params,
                        X_source=Z_source,
                        y_source=y_source,
                        X_target=Z_target_train,
                        y_target=target_split["y_train"],
                        X_eval=Z_target_val,
                        target_repeat=int(repeat),
                        random_state=random_state + prefix_size * 101 + prefix_rank * 29,
                        source_sample_weight=None,
                        source_weight_scale=1.0,
                    )
                    threshold, val_metrics, _ = _select_best_threshold(
                        y_true=target_split["y_val"],
                        y_prob=prob_val,
                        random_state=random_state + 3,
                        context_label=f"twod_cnn:{classifier_name}:prefix{prefix_size}",
                        classifier_name=classifier_name,
                        emit_debug=False,
                    )
                    candidate = {
                        "classifier_name": classifier_name,
                        "transfer_variant": "twod_cnn",
                        "alignment_method": "twod_cnn",
                        "prefix_rank": prefix_rank,
                        "prefix_size": prefix_size,
                        "chosen_source_ids": chosen_ids,
                        "params": dict(params),
                        "target_repeat": int(repeat),
                        "desired_source_target_ratio": float(desired_ratio),
                        "threshold": float(threshold),
                        "val_metrics": val_metrics,
                        "prefix_mmd": float(prefix_mmd),
                        "sample_rows": sample_rows,
                        "preprocessor_info": preprocessor.describe(),
                        "balance_info": dict(balance_info),
                    }
                    if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
                        best_candidate = candidate
                        print(
                            f"[Transfer][TwoDCNNBest] classifier={classifier_name} prefix_rank={prefix_rank} "
                            f"prefix={prefix_size} repeat={repeat} desired_ratio={desired_ratio:.2f} "
                            f"threshold={threshold:.3f} val_ACC={val_metrics['ACC']:.4f} val_F1={val_metrics['F1']:.4f}"
                        )

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} 的 TwoDCNN 迁移没有得到有效候选。")

    # 最终在测试集上评估（类似于 supervised 分支的尾部逻辑）
    X_source_final_raw, y_source_final, sample_rows = _stack_source_blocks(
        source_subject_ids=best_candidate["chosen_source_ids"],
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state + 911,
    )
    X_target_train_val_raw = np.vstack([target_split["X_train"], target_split["X_val"]]).astype(np.float32)
    y_target_train_val = np.hstack([target_split["y_train"], target_split["y_val"]]).astype(np.int32)
    final_preprocessor = TransferFeaturePreprocessor()
    X_source_and_target = final_preprocessor.fit_transform(
        np.vstack([X_source_final_raw, X_target_train_val_raw]).astype(np.float32)
    )
    X_source_final = X_source_and_target[: len(X_source_final_raw)]
    X_target_train_val = X_source_and_target[len(X_source_final_raw):]
    X_test = final_preprocessor.transform(target_split["X_test"])

    # 重新用 TwoDCNN 提取最终特征
    adapter = build_domain_adapter(
        method='twod_cnn',
        input_dim=X_source_final.shape[1],
        encoding_dim=config.normalized_twod_cnn_encoding_dim(),
        epochs=config.normalized_twod_cnn_epochs(),
        batch_size=config.normalized_twod_cnn_batch_size(),
        verbose=True,
    )
    Z_source_final, Z_target_train_val = adapter.fit_transform(X_source_final, y_source_final, X_target_train_val)
    Z_test = adapter.transform(X_test)

    prob_test, final_balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=Z_source_final,
        y_source=y_source_final,
        X_target=Z_target_train_val,
        y_target=y_target_train_val,
        X_eval=Z_test,
        target_repeat=int(best_candidate["target_repeat"]),
        random_state=random_state + 919,
        source_sample_weight=None,
        source_weight_scale=1.0,
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    print(
        f"[Transfer][TwoDCNNFinal] classifier={classifier_name} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])})"
    )
    return {
        **best_candidate,
        "source_weight_scale": 1.0,
        "test_metrics": test_metrics,
        "final_preprocessor_info": final_preprocessor.describe(),
        "final_sample_rows": sample_rows,
        "prefix_rows": list(mmd_prefix_selection["prefix_rows"]),
        "candidate_prefix_count": len(candidate_prefixes),
        "final_balance_info": dict(final_balance_info),
        "selection_mode": "transfer",
        "gate_reason": "transfer_selected",
    }

def _run_dann_transfer(
    classifier_name: str,
    mmd_prefix_selection: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
    target_only_val_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """使用 DANN 作为域自适应特征提取器的迁移学习分支。"""
    search_profile = _classifier_search_profile(classifier_name, config)
    repeat_grid = tuple(int(value) for value in search_profile["target_repeat_grid"])
    source_target_ratio_grid = tuple(float(value) for value in search_profile["source_target_ratio_grid"])
    candidate_prefixes = [dict(candidate) for candidate in mmd_prefix_selection.get("candidate_prefixes", [])]
    if not candidate_prefixes:
        raise RuntimeError("累计前缀候选池为空，无法继续 DANN 迁移搜索。")

    best_candidate: Dict[str, Any] | None = None
    total_candidates = 0

    print(
        f"[Transfer][SearchPlan] classifier={classifier_name} variant=dann "
        f"prefix_candidates={len(candidate_prefixes)} "
        f"encoding_dim={config.dann_encoding_dim} lambda={config.dann_lambda} "
        f"epochs={config.dann_epochs} batch_size={config.dann_batch_size}"
    )

    for prefix_candidate in candidate_prefixes:
        chosen_ids = [int(subject_id) for subject_id in prefix_candidate["chosen_source_ids"]]
        prefix_size = int(prefix_candidate["prefix_size"])
        prefix_rank = int(prefix_candidate.get("prefix_rank", 0))
        prefix_mmd = float(prefix_candidate["prefix_mmd"])

        # 1. 堆叠源域数据
        X_source_raw, y_source, sample_rows = _stack_source_blocks(
            source_subject_ids=chosen_ids,
            subject_blocks=subject_blocks,
            sample_cap=config.source_sample_cap,
            random_state=random_state + prefix_size * 41 + prefix_rank * 131,
        )
        # 2. 预处理（对数变换+标准化）
        preprocessor = TransferFeaturePreprocessor()
        X_source_and_target = preprocessor.fit_transform(
            np.vstack([X_source_raw, target_split["X_train"]]).astype(np.float32)
        )
        X_source = X_source_and_target[: len(X_source_raw)]
        X_target_train = X_source_and_target[len(X_source_raw):]
        X_target_val = preprocessor.transform(target_split["X_val"])

        # 3. 调用 DANN 适配器进行特征提取
        from AI.domain_adaptation import build_domain_adapter
        adapter = build_domain_adapter(
            method='dann',
            input_dim=X_source.shape[1],
            encoding_dim=config.dann_encoding_dim,
            lambda_=config.dann_lambda,
            epochs=config.dann_epochs,
            batch_size=config.dann_batch_size,
            verbose=True,
        )
        Z_source, Z_target_train = adapter.fit_transform(X_source, y_source, X_target_train)
        Z_target_val = adapter.transform(X_target_val)

        # 4. 在提取后的特征上搜索分类器超参数（与 supervised 分支相同）
        for params in search_profile["param_grid"]:
            for repeat in repeat_grid:
                for desired_ratio in source_target_ratio_grid:
                    total_candidates += 1
                    prob_val, balance_info = _fit_predict_probability(
                        classifier_name=classifier_name,
                        params=params,
                        X_source=Z_source,
                        y_source=y_source,
                        X_target=Z_target_train,
                        y_target=target_split["y_train"],
                        X_eval=Z_target_val,
                        target_repeat=int(repeat),
                        random_state=random_state + prefix_size * 101 + prefix_rank * 29,
                        source_sample_weight=None,
                        source_weight_scale=1.0,
                    )
                    threshold, val_metrics, _ = _select_best_threshold(
                        y_true=target_split["y_val"],
                        y_prob=prob_val,
                        random_state=random_state + 3,
                        context_label=f"dann:{classifier_name}:prefix{prefix_size}",
                        classifier_name=classifier_name,
                        emit_debug=False,
                    )
                    candidate = {
                        "classifier_name": classifier_name,
                        "transfer_variant": "dann",
                        "alignment_method": "dann",
                        "prefix_rank": prefix_rank,
                        "prefix_size": prefix_size,
                        "chosen_source_ids": chosen_ids,
                        "params": dict(params),
                        "target_repeat": int(repeat),
                        "desired_source_target_ratio": float(desired_ratio),
                        "threshold": float(threshold),
                        "val_metrics": val_metrics,
                        "prefix_mmd": float(prefix_mmd),
                        "sample_rows": sample_rows,
                        "preprocessor_info": preprocessor.describe(),
                    }
                    if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
                        best_candidate = candidate
                        print(
                            f"[Transfer][DANNBest] classifier={classifier_name} prefix_rank={prefix_rank} "
                            f"prefix={prefix_size} repeat={repeat} desired_ratio={desired_ratio:.2f} "
                            f"threshold={threshold:.3f} val_ACC={val_metrics['ACC']:.4f} val_F1={val_metrics['F1']:.4f}"
                        )

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} 的 DANN 迁移没有得到有效候选。")

    # 最终在测试集上评估（使用最佳候选的配置）
    chosen_ids = best_candidate["chosen_source_ids"]
    X_source_final_raw, y_source_final, sample_rows = _stack_source_blocks(
        source_subject_ids=chosen_ids,
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state + 911,
    )
    X_target_train_val_raw = np.vstack([target_split["X_train"], target_split["X_val"]]).astype(np.float32)
    y_target_train_val = np.hstack([target_split["y_train"], target_split["y_val"]]).astype(np.int32)

    final_preprocessor = TransferFeaturePreprocessor()
    X_source_and_target_final = final_preprocessor.fit_transform(
        np.vstack([X_source_final_raw, X_target_train_val_raw]).astype(np.float32)
    )
    X_source_final = X_source_and_target_final[: len(X_source_final_raw)]
    X_target_train_val = X_source_and_target_final[len(X_source_final_raw):]
    X_test = final_preprocessor.transform(target_split["X_test"])

    # 再次调用 DANN 适配器（用训练+验证集作为目标域）
    adapter_final = build_domain_adapter(
        method='dann',
        input_dim=X_source_final.shape[1],
        encoding_dim=config.dann_encoding_dim,
        lambda_=config.dann_lambda,
        epochs=config.dann_epochs,
        batch_size=config.dann_batch_size,
        verbose=True,
    )
    Z_source_final, Z_target_train_val = adapter_final.fit_transform(
        X_source_final, y_source_final, X_target_train_val
    )
    Z_test = adapter_final.transform(X_test)

    prob_test, final_balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=Z_source_final,
        y_source=y_source_final,
        X_target=Z_target_train_val,
        y_target=y_target_train_val,
        X_eval=Z_test,
        target_repeat=int(best_candidate["target_repeat"]),
        random_state=random_state + 919,
        source_sample_weight=None,
        source_weight_scale=1.0,
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    val_test_gap_acc = float(test_metrics["ACC"]) - float(best_candidate["val_metrics"]["ACC"])
    val_test_gap_f1 = float(test_metrics["F1"]) - float(best_candidate["val_metrics"]["F1"])
    print(
        f"[Transfer][DANNFinal] classifier={classifier_name} chosen_prefix_rank={best_candidate.get('prefix_rank', 0)} "
        f"chosen_prefix={best_candidate['prefix_size']} repeat={best_candidate['target_repeat']} "
        f"desired_ratio={best_candidate.get('desired_source_target_ratio', 0.0):.2f} "
        f"threshold={best_candidate['threshold']:.3f} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])}) "
        f"gap_ACC={val_test_gap_acc:+.4f} gap_F1={val_test_gap_f1:+.4f}"
    )

    return {
        **best_candidate,
        "search_source_weight_scale": 1.0,
        "source_weight_scale": 1.0,
        "test_metrics": test_metrics,
        "final_preprocessor_info": final_preprocessor.describe(),
        "final_sample_rows": sample_rows,
        "prefix_rows": list(mmd_prefix_selection["prefix_rows"]),
        "candidate_prefix_count": len(candidate_prefixes),
        "final_balance_info": dict(final_balance_info),
        "source_weight_info": {"mode": "dann", "preview": "dann_feature_extraction"},
        "jda_diag": {},
        "final_jda_diag": {},
        "selection_mode": "transfer",
        "gate_reason": "transfer_selected",
    }

def _run_enhanced_jda_transfer(
    classifier_name: str,
    mmd_prefix_selection: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
    target_only_val_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """按论文 5.2 运行“多源域 MMD 选源 + JDA + 高置信伪标签自训练”迁移。"""
    search_profile = _classifier_search_profile(classifier_name, config)
    pseudo_keep_ratio_grid = tuple(float(value) for value in config.normalized_jda_pseudo_keep_ratio_grid())
    pseudo_target_repeat_grid = tuple(int(value) for value in config.normalized_jda_pseudo_target_repeat_grid())
    best_candidate: Dict[str, Any] | None = None
    best_non_dominated_candidate: Dict[str, Any] | None = None
    best_mmd_prefix = dict(mmd_prefix_selection.get("best_mmd_prefix", {}))
    if not best_mmd_prefix:
        raise RuntimeError("缺少论文 5.2 所需的最小前缀 MMD 源域组合。")

    total_candidates = 0
    nondominated_candidates = 0
    print(
        f"[Transfer][SearchPlan] classifier={classifier_name} variant=enhanced_jda "
        f"prefix_candidates=1 prefix_search=paper52_local_min_prefix "
        f"params={len(search_profile['param_grid'])} pseudo_keep={list(pseudo_keep_ratio_grid)} "
        f"pseudo_repeat={list(pseudo_target_repeat_grid)} classifier_fit=source_plus_high_conf_pseudo_after_jda "
        f"jda=({_jda_config_text(config)})"
    )
    chosen_ids = [int(subject_id) for subject_id in best_mmd_prefix["chosen_source_ids"]]
    prefix_size = int(best_mmd_prefix["prefix_size"])
    prefix_rank = int(best_mmd_prefix.get("prefix_rank", 1))
    prefix_mmd = float(best_mmd_prefix["prefix_mmd"])

    X_source_raw, y_source, sample_rows = _stack_source_blocks(
        source_subject_ids=chosen_ids,
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state + prefix_size * 41 + prefix_rank * 131,
    )
    preprocessor = TransferFeaturePreprocessor()
    X_source_and_target = preprocessor.fit_transform(np.vstack([X_source_raw, target_split["X_train"]]).astype(np.float32))
    X_source = X_source_and_target[: len(X_source_raw)]
    X_target_train = X_source_and_target[len(X_source_raw):]
    X_target_val = preprocessor.transform(target_split["X_val"])
    source_weight_info = {
        **_paper52_source_only_weight_info(len(y_source)),
        "mode": "paper52_jda_pseudo_self_train",
        "preview": "source_plus_high_confidence_pseudo_target",
    }
    print(
        f"[Transfer][Paper52] classifier={classifier_name} chosen_prefix_rank={prefix_rank} "
        f"chosen_prefix={prefix_size} prefix_mmd={prefix_mmd:.4f} "
        f"source_training_mode={source_weight_info['mode']} chosen_sources={chosen_ids}"
    )

    Z_source, Z_target_train, aligned_eval_blocks, jda_diag, pseudo_bundle = _run_jda_alignment(
        X_source=X_source,
        y_source=y_source,
        X_target_train=X_target_train,
        eval_blocks={"val": X_target_val},
        config=config,
        random_state=random_state + prefix_size * 173 + prefix_rank * 37,
        debug_target_y=target_split["y_train"],
    )
    Z_target_val = aligned_eval_blocks["val"]
    print(
        f"[Transfer][JDAPrefix] classifier={classifier_name} prefix_rank={prefix_rank} prefix={prefix_size} "
        f"mmd_before={prefix_mmd:.4f} jda_end_mmd={jda_diag['end_mmd']:.4f} "
        f"jda_delta_mmd={jda_diag['delta_mmd']:+.4f} pseudo_change={jda_diag['last_pseudo_change_ratio']:.4f}"
    )

    feature_dim = int(Z_source.shape[1] if len(Z_source) else Z_target_train.shape[1])
    for params_id, params in enumerate(search_profile["param_grid"], start=1):
        for keep_ratio in pseudo_keep_ratio_grid:
            pseudo_X_train, pseudo_y_train, pseudo_diag = _select_high_confidence_pseudo_samples(
                X_target_train=Z_target_train,
                pseudo_labels=pseudo_bundle.get("pseudo_labels"),
                pseudo_confidence=pseudo_bundle.get("confidence"),
                keep_ratio=float(keep_ratio),
            )
            for pseudo_repeat in pseudo_target_repeat_grid:
                total_candidates += 1
                fit_random_state = (
                    random_state
                    + prefix_size * 101
                    + prefix_rank * 29
                    + params_id * 17
                    + int(round(float(keep_ratio) * 100))
                    + int(pseudo_repeat) * 19
                )
                prob_val, balance_info = _fit_predict_probability(
                    classifier_name=classifier_name,
                    params=params,
                    X_source=Z_source,
                    y_source=y_source,
                    X_target=pseudo_X_train,
                    y_target=pseudo_y_train,
                    X_eval=Z_target_val,
                    target_repeat=int(pseudo_repeat),
                    random_state=fit_random_state,
                    source_sample_weight=None,
                    source_weight_scale=1.0,
                )
                threshold, val_metrics, _ = _select_best_threshold(
                    y_true=target_split["y_val"],
                    y_prob=prob_val,
                    random_state=fit_random_state + 3,
                    context_label=(
                        f"enhanced_jda_paper52:{classifier_name}:prefix{prefix_size}:params{params_id}:"
                        f"keep{float(keep_ratio):.2f}:repeat{int(pseudo_repeat)}"
                    ),
                    classifier_name=classifier_name,
                    emit_debug=False,
                )
                val_vs_target_acc = (
                    float(val_metrics["ACC"]) - float(target_only_val_metrics["ACC"])
                    if target_only_val_metrics is not None
                    else 0.0
                )
                val_vs_target_f1 = (
                    float(val_metrics["F1"]) - float(target_only_val_metrics["F1"])
                    if target_only_val_metrics is not None
                    else 0.0
                )
                candidate = {
                    "classifier_name": classifier_name,
                    "transfer_variant": "enhanced_jda",
                    "alignment_method": "jda",
                    "prefix_rank": prefix_rank,
                    "prefix_size": int(prefix_size),
                    "chosen_source_ids": chosen_ids,
                    "params": dict(params),
                    "target_repeat": int(pseudo_repeat),
                    "pseudo_keep_ratio": float(keep_ratio),
                    "pseudo_selected_count": int(round(float(pseudo_diag["selected_count"]))),
                    "pseudo_selected_ratio": float(pseudo_diag["selected_ratio"]),
                    "pseudo_selected_conf_mean": float(pseudo_diag["selected_conf_mean"]),
                    "pseudo_selected_pos_ratio": float(pseudo_diag["selected_pos_ratio"]),
                    "desired_source_target_ratio": 0.0,
                    "effective_source_target_ratio": float(balance_info["effective_source_target_ratio"]),
                    "source_weight_scale": 1.0,
                    "threshold": float(threshold),
                    "val_metrics": val_metrics,
                    "prefix_mmd": float(prefix_mmd),
                    "sample_rows": sample_rows,
                    "preprocessor_info": preprocessor.describe(),
                    "balance_info": dict(balance_info),
                    "source_weight_info": dict(source_weight_info),
                    "source_mmd_rows": list(best_mmd_prefix.get("source_mmd_rows", [])),
                    "prefix_rank_preview": str(mmd_prefix_selection.get("prefix_rank_preview", "")),
                    "jda_diag": dict(jda_diag),
                }
                if best_candidate is None or _candidate_key(val_metrics) > _candidate_key(best_candidate["val_metrics"]):
                    best_candidate = candidate
                    print(
                        f"[Transfer][JDABest] classifier={classifier_name} prefix_rank={prefix_rank} "
                        f"prefix={prefix_size} pseudo_keep={float(keep_ratio):.2f} pseudo_repeat={int(pseudo_repeat)} "
                        f"pseudo_selected={int(round(float(pseudo_diag['selected_count'])))} "
                        f"pseudo_conf_mean={pseudo_diag['selected_conf_mean']:.4f} threshold={threshold:.3f} "
                        f"actual_ratio={balance_info['effective_source_target_ratio']:.3f} "
                        f"mmd={prefix_mmd:.4f} jda_end_mmd={jda_diag['end_mmd']:.4f} "
                        f"srcW={balance_info['effective_source_strength']:.2f} tgtW={balance_info['effective_target_strength']:.2f} "
                        f"params=({_format_params(params)}) val_ACC={val_metrics['ACC']:.4f} val_F1={val_metrics['F1']:.4f} "
                        f"dValTarget_ACC={val_vs_target_acc:+.4f} dValTarget_F1={val_vs_target_f1:+.4f}"
                    )
                if target_only_val_metrics is not None:
                    dominated_by_target = (
                        float(val_metrics["ACC"]) < float(target_only_val_metrics["ACC"])
                        and float(val_metrics["F1"]) < float(target_only_val_metrics["F1"])
                    )
                    if not dominated_by_target and (
                        best_non_dominated_candidate is None
                        or _candidate_key(val_metrics) > _candidate_key(best_non_dominated_candidate["val_metrics"])
                    ):
                        best_non_dominated_candidate = candidate
                        nondominated_candidates += 1

    print(
        f"[Transfer][SearchSummary] classifier={classifier_name} variant=enhanced_jda "
        f"total_candidates={total_candidates} nondominated_updates={nondominated_candidates}"
    )

    if best_non_dominated_candidate is not None:
        best_candidate = best_non_dominated_candidate

    if best_candidate is None:
        raise RuntimeError(f"{classifier_name} 的增强 JDA 迁移没有得到有效候选。")

    X_source_final_raw, y_source_final, sample_rows = _stack_source_blocks(
        source_subject_ids=best_candidate["chosen_source_ids"],
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=random_state + 911,
    )
    source_weight_info_final = {
        **_paper52_source_only_weight_info(len(y_source_final)),
        "mode": "paper52_jda_pseudo_self_train",
        "preview": "source_plus_high_confidence_pseudo_target",
    }
    X_target_train_val_raw = np.vstack([target_split["X_train"], target_split["X_val"]]).astype(np.float32)
    y_target_train_val = np.hstack([target_split["y_train"], target_split["y_val"]]).astype(np.int32)
    final_preprocessor = TransferFeaturePreprocessor()
    X_source_and_target = final_preprocessor.fit_transform(
        np.vstack([X_source_final_raw, X_target_train_val_raw]).astype(np.float32)
    )
    X_source_final = X_source_and_target[: len(X_source_final_raw)]
    X_target_train_val = X_source_and_target[len(X_source_final_raw):]
    X_test = final_preprocessor.transform(target_split["X_test"])
    Z_source_final, Z_target_train_val, aligned_eval_blocks, jda_diag_final, pseudo_bundle_final = _run_jda_alignment(
        X_source=X_source_final,
        y_source=y_source_final,
        X_target_train=X_target_train_val,
        eval_blocks={"test": X_test},
        config=config,
        random_state=random_state + 919,
        debug_target_y=y_target_train_val,
    )
    pseudo_X_train_val, pseudo_y_train_val, pseudo_diag_final = _select_high_confidence_pseudo_samples(
        X_target_train=Z_target_train_val,
        pseudo_labels=pseudo_bundle_final.get("pseudo_labels"),
        pseudo_confidence=pseudo_bundle_final.get("confidence"),
        keep_ratio=float(best_candidate.get("pseudo_keep_ratio", 0.0)),
    )
    prob_test, final_balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=best_candidate["params"],
        X_source=Z_source_final,
        y_source=y_source_final,
        X_target=pseudo_X_train_val,
        y_target=pseudo_y_train_val,
        X_eval=aligned_eval_blocks["test"],
        target_repeat=int(best_candidate.get("target_repeat", 1)),
        random_state=random_state + 937,
        source_sample_weight=None,
        source_weight_scale=1.0,
    )
    test_metrics = binary_metrics(
        target_split["y_test"],
        (prob_test >= float(best_candidate["threshold"])).astype(np.int32),
        prob_test,
    )
    val_test_gap_acc = float(test_metrics["ACC"]) - float(best_candidate["val_metrics"]["ACC"])
    val_test_gap_f1 = float(test_metrics["F1"]) - float(best_candidate["val_metrics"]["F1"])
    print(
        f"[Transfer][JDAFinal] classifier={classifier_name} chosen_prefix_rank={best_candidate.get('prefix_rank', 0)} "
        f"chosen_prefix={best_candidate['prefix_size']} pseudo_keep={float(best_candidate.get('pseudo_keep_ratio', 0.0)):.2f} "
        f"pseudo_repeat={int(best_candidate.get('target_repeat', 1))} "
        f"pseudo_selected={int(round(float(pseudo_diag_final['selected_count'])))} "
        f"actual_ratio={final_balance_info['effective_source_target_ratio']:.3f} "
        f"threshold={best_candidate['threshold']:.3f} jda_end_mmd={jda_diag_final['end_mmd']:.4f} "
        f"test_metrics=({_format_metric_snapshot(test_metrics)}) "
        f"val_metrics=({_format_metric_snapshot(best_candidate['val_metrics'])}) "
        f"gap_ACC={val_test_gap_acc:+.4f} gap_F1={val_test_gap_f1:+.4f}"
    )
    return {
        **best_candidate,
        "search_source_weight_scale": 1.0,
        "source_weight_scale": 1.0,
        "test_metrics": test_metrics,
        "final_preprocessor_info": final_preprocessor.describe(),
        "final_sample_rows": sample_rows,
        "prefix_rows": list(mmd_prefix_selection["prefix_rows"]),
        "candidate_prefix_count": 1,
        "final_balance_info": dict(final_balance_info),
        "source_weight_info": dict(source_weight_info_final),
        "pseudo_selected_count": int(round(float(pseudo_diag_final["selected_count"]))),
        "pseudo_selected_ratio": float(pseudo_diag_final["selected_ratio"]),
        "pseudo_selected_conf_mean": float(pseudo_diag_final["selected_conf_mean"]),
        "pseudo_selected_pos_ratio": float(pseudo_diag_final["selected_pos_ratio"]),
        "final_jda_diag": dict(jda_diag_final),
        "selection_mode": "transfer",
        "gate_reason": "transfer_selected",
    }


def _run_selected_transfer_variant(
    classifier_name: str,
    mmd_prefix_selection: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
    target_only_val_metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """根据配置切换监督迁移或增强 JDA 分支，统一主流程入口。"""
    transfer_variant = config.normalized_transfer_variant()
    if transfer_variant == "enhanced_jda":
        return _run_enhanced_jda_transfer(
            classifier_name=classifier_name,
            mmd_prefix_selection=mmd_prefix_selection,
            subject_blocks=subject_blocks,
            target_split=target_split,
            config=config,
            random_state=random_state,
            target_only_val_metrics=target_only_val_metrics,
        )
    elif transfer_variant == "twod_cnn":
        return _run_twod_cnn_transfer(
            classifier_name=classifier_name,
            mmd_prefix_selection=mmd_prefix_selection,
            subject_blocks=subject_blocks,
            target_split=target_split,
            config=config,
            random_state=random_state,
            target_only_val_metrics=target_only_val_metrics,
        )
    elif transfer_variant == "dann":
        return _run_dann_transfer(
            classifier_name=classifier_name,
            mmd_prefix_selection=mmd_prefix_selection,
            subject_blocks=subject_blocks,
            target_split=target_split,
            config=config,
            random_state=random_state,
            target_only_val_metrics=target_only_val_metrics,
        )
    return _run_supervised_transfer(
        classifier_name=classifier_name,
        mmd_prefix_selection=mmd_prefix_selection,
        subject_blocks=subject_blocks,
        target_split=target_split,
        config=config,
        random_state=random_state,
        target_only_val_metrics=target_only_val_metrics,
    )


def _evaluate_fixed_target_only_candidate_on_split(
    classifier_name: str,
    params: Dict[str, Any],
    target_split: Dict[str, Any],
    random_state: int,
) -> Dict[str, Any]:
    """在新的目标域切分上复核固定 target-only 候选的验证集表现。"""
    # 中文作用: gate 复核时，不再重新搜完整网格，而是用主搜索选出的 target-only 参数在新切分上复测。
    search_preprocessor = TransferFeaturePreprocessor()
    X_train = search_preprocessor.fit_transform(target_split["X_train"])
    X_val = search_preprocessor.transform(target_split["X_val"])
    feature_dim = int(X_train.shape[1])
    prob_val, balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=params,
        X_source=_empty_block(feature_dim),
        y_source=np.zeros(0, dtype=np.int32),
        X_target=X_train,
        y_target=target_split["y_train"],
        X_eval=X_val,
        target_repeat=1,
        random_state=int(random_state),
    )
    threshold, val_metrics, _ = _select_best_threshold(
        y_true=target_split["y_val"],
        y_prob=prob_val,
        random_state=int(random_state + 5),
        context_label=f"gate_repeat_target_only:{classifier_name}",
        classifier_name=classifier_name,
        emit_debug=False,
    )
    return {
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "balance_info": balance_info,
    }


def _evaluate_fixed_transfer_candidate_on_split(
    classifier_name: str,
    transfer_result: Dict[str, Any],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_split: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """在新的目标域切分上复核固定迁移候选的验证集表现。"""
    # 中文作用: gate 复核时固定源前缀、参数和目标重复次数，只重新在新切分上测 transfer 与 target-only 的相对优劣。
    chosen_source_ids = [int(subject_id) for subject_id in transfer_result.get("chosen_source_ids", [])]
    if not chosen_source_ids:
        raise ValueError("固定迁移候选缺少 chosen_source_ids，无法执行 gate 复核。")

    X_source_raw, y_source, sample_rows = _stack_source_blocks(
        source_subject_ids=chosen_source_ids,
        subject_blocks=subject_blocks,
        sample_cap=config.source_sample_cap,
        random_state=int(random_state + 11),
    )
    preprocessor = TransferFeaturePreprocessor()
    X_source_and_target = preprocessor.fit_transform(
        np.vstack([X_source_raw, target_split["X_train"]]).astype(np.float32)
    )
    X_source = X_source_and_target[: len(X_source_raw)]
    X_target_train = X_source_and_target[len(X_source_raw):]
    X_val = preprocessor.transform(target_split["X_val"])
    transfer_variant = str(transfer_result.get("transfer_variant", "supervised"))
    jda_diag: Dict[str, float] = {}
    if transfer_variant == "enhanced_jda":
        source_sample_weight = None
        source_weight_info = {
            **_paper52_source_only_weight_info(len(y_source)),
            "mode": "paper52_jda_pseudo_self_train",
            "preview": "source_plus_high_confidence_pseudo_target",
        }
        source_scale = 1.0
        X_source, X_target_train, aligned_eval_blocks, jda_diag, pseudo_bundle = _run_jda_alignment(
            X_source=X_source,
            y_source=y_source,
            X_target_train=X_target_train,
            eval_blocks={"val": X_val},
            config=config,
            random_state=int(random_state + 13),
            debug_target_y=target_split["y_train"],
        )
        X_val = aligned_eval_blocks["val"]
        X_fit_target, y_fit_target, _pseudo_diag = _select_high_confidence_pseudo_samples(
            X_target_train=X_target_train,
            pseudo_labels=pseudo_bundle.get("pseudo_labels"),
            pseudo_confidence=pseudo_bundle.get("confidence"),
            keep_ratio=float(transfer_result.get("pseudo_keep_ratio", 0.0)),
        )
        fit_target_repeat = int(transfer_result.get("target_repeat", 1))
    else:
        source_sample_weight, source_weight_info = _build_source_sample_weights(
            sample_rows=sample_rows,
            source_mmd_rows=transfer_result.get("source_mmd_rows", []),
            enabled=bool(config.use_inverse_mmd_source_weighting),
        )
        if source_sample_weight is not None:
            base_source_strength = float(source_weight_info["total_weight"])
        else:
            base_source_strength = float(len(y_source))
        target_strength = float(len(target_split["y_train"]) * int(transfer_result["target_repeat"]))
        source_scale = _resolve_source_weight_scale(
            base_source_strength=base_source_strength,
            target_strength=target_strength,
            desired_ratio=float(transfer_result.get("desired_source_target_ratio", 0.0)),
        )
        X_fit_target = X_target_train
        y_fit_target = target_split["y_train"]
        fit_target_repeat = int(transfer_result["target_repeat"])
    prob_val, balance_info = _fit_predict_probability(
        classifier_name=classifier_name,
        params=dict(transfer_result["params"]),
        X_source=X_source,
        y_source=y_source,
        X_target=X_fit_target,
        y_target=y_fit_target,
        X_eval=X_val,
        target_repeat=fit_target_repeat,
        random_state=int(random_state + 17),
        source_sample_weight=source_sample_weight,
        source_weight_scale=float(source_scale),
    )
    threshold, val_metrics, _ = _select_best_threshold(
        y_true=target_split["y_val"],
        y_prob=prob_val,
        random_state=int(random_state + 23),
        context_label=f"gate_repeat_transfer:{classifier_name}",
        classifier_name=classifier_name,
        emit_debug=False,
    )
    return {
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "balance_info": balance_info,
        "source_weight_info": source_weight_info,
        "source_weight_scale": float(source_scale),
        "jda_diag": dict(jda_diag),
    }


def _run_repeat_gate_check(
    classifier_name: str,
    target_subject_ids: Sequence[int],
    subject_blocks: Dict[int, tuple[np.ndarray, np.ndarray]],
    target_only_result: Dict[str, Any],
    transfer_result: Dict[str, Any],
    config: TransferLearningConfig,
    random_state: int,
) -> Dict[str, Any]:
    """对原本会被回退的迁移候选做多次重复切分复核，减少单次验证切分的误杀。"""
    # 中文作用: 只对已选中的迁移候选做固定参数复测，用多次等量分层切分判断它是否被单次验证切分误杀。
    total_gate_evals = int(config.normalized_gate_repeat_count())
    extra_repeat_count = max(0, total_gate_evals - 1)
    if extra_repeat_count <= 0 or int(transfer_result.get("prefix_size", 0)) <= 0:
        return {
            "repeat_gate_enabled": False,
            "repeat_gate_passed": False,
            "repeat_rows": [],
            "repeat_keep_votes": 0,
            "repeat_total": 0,
            "mean_gain_acc": 0.0,
            "mean_gain_f1": 0.0,
        }

    repeat_rows: List[Dict[str, Any]] = []
    keep_votes = 0
    for repeat_idx in range(extra_repeat_count):
        split_seed = int(random_state + 1009 + repeat_idx * 211)
        repeat_split = _split_target_subjects(
            target_subject_ids=target_subject_ids,
            subject_blocks=subject_blocks,
            config=config,
            random_state=split_seed,
        )
        repeat_target_only = _evaluate_fixed_target_only_candidate_on_split(
            classifier_name=classifier_name,
            params=dict(target_only_result["params"]),
            target_split=repeat_split,
            random_state=split_seed + 31,
        )
        repeat_transfer = _evaluate_fixed_transfer_candidate_on_split(
            classifier_name=classifier_name,
            transfer_result=transfer_result,
            subject_blocks=subject_blocks,
            target_split=repeat_split,
            config=config,
            random_state=split_seed + 67,
        )
        acc_gain = float(repeat_transfer["val_metrics"]["ACC"]) - float(repeat_target_only["val_metrics"]["ACC"])
        f1_gain = float(repeat_transfer["val_metrics"]["F1"]) - float(repeat_target_only["val_metrics"]["F1"])
        keep_flag = (
            _candidate_key(repeat_transfer["val_metrics"]) >= _candidate_key(repeat_target_only["val_metrics"])
            and acc_gain >= 0.0
            and f1_gain >= 0.0
        )
        keep_votes += int(keep_flag)
        repeat_rows.append(
            {
                "repeat_id": int(repeat_idx + 1),
                "target_only_acc": float(repeat_target_only["val_metrics"]["ACC"]),
                "target_only_f1": float(repeat_target_only["val_metrics"]["F1"]),
                "transfer_acc": float(repeat_transfer["val_metrics"]["ACC"]),
                "transfer_f1": float(repeat_transfer["val_metrics"]["F1"]),
                "gain_acc": float(acc_gain),
                "gain_f1": float(f1_gain),
                "keep": bool(keep_flag),
            }
        )
        print(
            f"[Transfer][GateRepeat] classifier={classifier_name} repeat={repeat_idx + 1}/{extra_repeat_count} "
            f"transfer_ACC={repeat_transfer['val_metrics']['ACC']:.4f} transfer_F1={repeat_transfer['val_metrics']['F1']:.4f} "
            f"target_ACC={repeat_target_only['val_metrics']['ACC']:.4f} target_F1={repeat_target_only['val_metrics']['F1']:.4f} "
            f"gain_ACC={acc_gain:+.4f} gain_F1={f1_gain:+.4f} keep={keep_flag}"
        )

    mean_gain_acc = float(np.mean([row["gain_acc"] for row in repeat_rows])) if repeat_rows else 0.0
    mean_gain_f1 = float(np.mean([row["gain_f1"] for row in repeat_rows])) if repeat_rows else 0.0
    repeat_gate_passed = (
        bool(repeat_rows)
        and keep_votes >= max(1, (len(repeat_rows) + 1) // 2)
        and mean_gain_acc >= 0.0
        and mean_gain_f1 >= 0.0
    )
    print(
        f"[Transfer][GateRepeatSummary] classifier={classifier_name} repeats={len(repeat_rows)} "
        f"keep_votes={keep_votes} mean_gain_ACC={mean_gain_acc:+.4f} mean_gain_F1={mean_gain_f1:+.4f} "
        f"passed={repeat_gate_passed}"
    )
    return {
        "repeat_gate_enabled": True,
        "repeat_gate_passed": bool(repeat_gate_passed),
        "repeat_rows": repeat_rows,
        "repeat_keep_votes": int(keep_votes),
        "repeat_total": int(len(repeat_rows)),
        "mean_gain_acc": float(mean_gain_acc),
        "mean_gain_f1": float(mean_gain_f1),
    }


def _select_transfer_or_target_only(
    classifier_name: str,
    target_only_result: Dict[str, Any],
    transfer_result: Dict[str, Any],
) -> Dict[str, Any]:
    """按验证集表现决定保留迁移结果，还是启用 target-only 保护门控。"""
    # 中文作用: 根据验证集增益决定保留迁移结果，还是回退到 target-only。
    transfer_acc = float(transfer_result["val_metrics"]["ACC"])
    transfer_f1 = float(transfer_result["val_metrics"]["F1"])
    target_acc = float(target_only_result["val_metrics"]["ACC"])
    target_f1 = float(target_only_result["val_metrics"]["F1"])
    acc_gain = transfer_acc - target_acc
    f1_gain = transfer_f1 - target_f1
    min_acc_gain, min_f1_gain = _classifier_gate_thresholds(classifier_name)
    transfer_key = _candidate_key(transfer_result["val_metrics"])
    target_only_key = _candidate_key(target_only_result["val_metrics"])
    if (
        transfer_key >= target_only_key
        and acc_gain >= min_acc_gain
        and f1_gain >= min_f1_gain
    ):
        transfer_result["gate_reason"] = "transfer_kept"
        print(
            f"[Transfer][Gate] classifier={classifier_name} mode=transfer "
            f"val_transfer_ACC={transfer_acc:.4f} "
            f"val_transfer_F1={transfer_f1:.4f} "
            f"val_target_ACC={target_acc:.4f} "
            f"val_target_F1={target_f1:.4f} "
            f"gain_ACC={acc_gain:+.4f} gain_F1={f1_gain:+.4f} "
            f"min_acc_gain={min_acc_gain:.4f} min_f1_gain={min_f1_gain:.4f}"
        )
        return transfer_result

    repeat_gate_summary = transfer_result.get("gate_repeat_summary", {})
    if bool(repeat_gate_summary.get("repeat_gate_passed", False)):
        transfer_result["gate_reason"] = "transfer_kept_repeat_gate"
        print(
            f"[Transfer][Gate] classifier={classifier_name} mode=transfer_repeat_gate "
            f"val_transfer_ACC={transfer_acc:.4f} val_transfer_F1={transfer_f1:.4f} "
            f"val_target_ACC={target_acc:.4f} val_target_F1={target_f1:.4f} "
            f"repeat_keep_votes={int(repeat_gate_summary.get('repeat_keep_votes', 0))}/"
            f"{int(repeat_gate_summary.get('repeat_total', 0))} "
            f"mean_repeat_gain_ACC={float(repeat_gate_summary.get('mean_gain_acc', 0.0)):+.4f} "
            f"mean_repeat_gain_F1={float(repeat_gate_summary.get('mean_gain_f1', 0.0)):+.4f}"
        )
        return transfer_result

    gated_result = {
        **transfer_result,
        "params": dict(target_only_result["params"]),
        "threshold": float(target_only_result["threshold"]),
        "val_metrics": dict(target_only_result["val_metrics"]),
        "test_metrics": dict(target_only_result["test_metrics"]),
        "target_repeat": 1,
        "source_weight_scale": 0.0,
        "desired_source_target_ratio": 0.0,
        "effective_source_target_ratio": 0.0,
        "chosen_source_ids": [],
        "prefix_size": 0,
        "prefix_rank": 0,
        "source_weight_info": {
            "mode": "target_only_gate",
            "preview": "target_only_gate",
            "min_weight": 0.0,
            "max_weight": 0.0,
            "mean_weight": 0.0,
            "total_weight": 0.0,
        },
        "selection_mode": "target_only_gate",
        "gate_reason": (
            "gate_small_margin"
            if transfer_key >= target_only_key
            and (acc_gain < min_acc_gain or f1_gain < min_f1_gain)
            else _gate_reason_text(transfer_result["val_metrics"], target_only_result["val_metrics"])
        ),
        "final_balance_info": dict(target_only_result.get("balance_info", {})),
    }
    print(
        f"[Transfer][Gate] classifier={classifier_name} mode=target_only_gate "
        f"val_transfer_ACC={transfer_acc:.4f} "
        f"val_transfer_F1={transfer_f1:.4f} "
        f"val_target_ACC={target_acc:.4f} "
        f"val_target_F1={target_f1:.4f} "
        f"gain_ACC={acc_gain:+.4f} gain_F1={f1_gain:+.4f} "
        f"min_acc_gain={min_acc_gain:.4f} min_f1_gain={min_f1_gain:.4f}"
    )
    return gated_result


def _metric_mean(summary_block: Dict[str, Dict[str, Dict[str, float]]], classifier_name: str, metric_name: str) -> float:
    """从 mean/std 风格的汇总结果中抽取均值。"""
    metric_value = summary_block[classifier_name][metric_name]
    if isinstance(metric_value, dict):
        return float(metric_value.get("mean", 0.0))
    return float(metric_value)


def _summary_bar_rows(summary_block: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    """把 mean/std 风格的汇总结果转成柱状图行。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in [name for name in CLASSIFIER_ORDER if name in summary_block]:
        rows.append(
            {
                "Classifier": classifier_name,
                "ACC": _metric_mean(summary_block, classifier_name, "ACC"),
                "F1": _metric_mean(summary_block, classifier_name, "F1"),
            }
        )
    return rows


def _gain_rows(
    reference_summary: Dict[str, Dict[str, Dict[str, float]]],
    current_summary: Dict[str, Dict[str, Dict[str, float]]],
    reference_label: str,
    current_label: str,
) -> List[Dict[str, Any]]:
    """汇总监督迁移相对基线的 ACC/F1 变化。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in [name for name in CLASSIFIER_ORDER if name in reference_summary and name in current_summary]:
        reference_acc = _metric_mean(reference_summary, classifier_name, "ACC")
        current_acc = _metric_mean(current_summary, classifier_name, "ACC")
        reference_f1 = _metric_mean(reference_summary, classifier_name, "F1")
        current_f1 = _metric_mean(current_summary, classifier_name, "F1")
        rows.append(
            {
                "Classifier": classifier_name,
                f"{reference_label}_ACC": f"{reference_acc:.4f}",
                f"{current_label}_ACC": f"{current_acc:.4f}",
                "Delta_ACC": f"{current_acc - reference_acc:+.4f}",
                f"{reference_label}_F1": f"{reference_f1:.4f}",
                f"{current_label}_F1": f"{current_f1:.4f}",
                "Delta_F1": f"{current_f1 - reference_f1:+.4f}",
            }
        )
    return rows


def _summarize_method_rows(method_rows: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """把单个方法下的逐单元指标汇总成 mean/std。"""
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for classifier_name, metric_rows in method_rows.items():
        summary[classifier_name] = summarize_metric_list(metric_rows)
    return summary

def _summary_table_rows(summary_block: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    """把 mean/std 风格的汇总结果转换成 Markdown 表格行。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in [name for name in CLASSIFIER_ORDER if name in summary_block]:
        rows.append(
            {
                "Classifier": classifier_name,
                "ACC": f"{summary_block[classifier_name]['ACC']['mean']:.4f}+/-{summary_block[classifier_name]['ACC']['std']:.4f}",
                "F1": f"{summary_block[classifier_name]['F1']['mean']:.4f}+/-{summary_block[classifier_name]['F1']['std']:.4f}",
                "BACC": f"{summary_block[classifier_name]['BACC']['mean']:.4f}+/-{summary_block[classifier_name]['BACC']['std']:.4f}",
                "MCC": f"{summary_block[classifier_name]['MCC']['mean']:.4f}+/-{summary_block[classifier_name]['MCC']['std']:.4f}",
            }
        )
    return rows


def _method_comparison_rows(
    source_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    target_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    raw_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
    final_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, Any]]:
    """把 source-only、target-only、raw transfer、final transfer 汇总成同一张对照表。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        if classifier_name not in source_only_summary:
            continue
        rows.append(
            {
                "Classifier": classifier_name,
                "SourceOnly_ACC": f"{_metric_mean(source_only_summary, classifier_name, 'ACC'):.4f}",
                "SourceOnly_F1": f"{_metric_mean(source_only_summary, classifier_name, 'F1'):.4f}",
                "TargetOnly_ACC": f"{_metric_mean(target_only_summary, classifier_name, 'ACC'):.4f}",
                "TargetOnly_F1": f"{_metric_mean(target_only_summary, classifier_name, 'F1'):.4f}",
                "RawTransfer_ACC": f"{_metric_mean(raw_transfer_summary, classifier_name, 'ACC'):.4f}",
                "RawTransfer_F1": f"{_metric_mean(raw_transfer_summary, classifier_name, 'F1'):.4f}",
                "FinalTransfer_ACC": f"{_metric_mean(final_transfer_summary, classifier_name, 'ACC'):.4f}",
                "FinalTransfer_F1": f"{_metric_mean(final_transfer_summary, classifier_name, 'F1'):.4f}",
            }
        )
    return rows


def _gate_summary_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按分类器汇总 gate/保留次数，帮助定位负迁移分布。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row["Classifier"]) == classifier_name]
        if not classifier_rows:
            continue
        gate_count = sum(1 for row in classifier_rows if str(row.get("SelectionMode", "")) == "target_only_gate")
        keep_count = len(classifier_rows) - gate_count
        gate_rate = gate_count / max(len(classifier_rows), 1)
        gated_rows = [row for row in classifier_rows if str(row.get("SelectionMode", "")) == "target_only_gate"]
        reasons = [str(row.get("GateReason", "")) for row in gated_rows if str(row.get("GateReason", ""))]
        reason_counts: Dict[str, int] = {}
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reason = (
            sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
            if reason_counts
            else ""
        )
        rows.append(
            {
                "Classifier": classifier_name,
                "Units": int(len(classifier_rows)),
                "TransferKept": int(keep_count),
                "TargetOnlyGate": int(gate_count),
                "GateRate": f"{gate_rate:.4f}",
                "TopGateReason": top_reason,
            }
        )
    return rows


def _transfer_decision_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按分类器汇总哪些目标单元保留迁移、哪些目标单元回退到 target-only。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row["Classifier"]) == classifier_name]
        if not classifier_rows:
            continue
        transfer_units = sorted(
            str(row["TargetUnit"])
            for row in classifier_rows
            if str(row.get("SelectionMode", "")) == "transfer"
        )
        fallback_rows = [row for row in classifier_rows if str(row.get("SelectionMode", "")) == "target_only_gate"]
        fallback_units = sorted(str(row["TargetUnit"]) for row in fallback_rows)
        reason_counts: Dict[str, int] = {}
        for row in fallback_rows:
            reason = str(row.get("GateReason", "")).strip()
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "Classifier": classifier_name,
                "TransferCount": int(len(transfer_units)),
                "FallbackCount": int(len(fallback_units)),
                "TransferUnits": ",".join(transfer_units) if transfer_units else "-",
                "FallbackUnits": ",".join(fallback_units) if fallback_units else "-",
                "FallbackReasons": ", ".join(
                    f"{reason}:{count}"
                    for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
                )
                if reason_counts
                else "-",
            }
        )
    return rows


def _val_test_gap_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按分类器汇总验证集与测试集之间的 ACC/F1 落差。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row["Classifier"]) == classifier_name]
        if not classifier_rows:
            continue
        val_acc = np.mean([float(row["Val_ACC"]) for row in classifier_rows])
        test_acc = np.mean([float(row["Transfer_ACC"]) for row in classifier_rows])
        val_f1 = np.mean([float(row["Val_F1"]) for row in classifier_rows])
        test_f1 = np.mean([float(row["Transfer_F1"]) for row in classifier_rows])
        gap_acc = np.mean([float(row["ValTestGap_ACC"]) for row in classifier_rows])
        gap_f1 = np.mean([float(row["ValTestGap_F1"]) for row in classifier_rows])
        rows.append(
            {
                "Classifier": classifier_name,
                "Val_ACC": f"{val_acc:.4f}",
                "Test_ACC": f"{test_acc:.4f}",
                "Gap_ACC": f"{gap_acc:+.4f}",
                "Val_F1": f"{val_f1:.4f}",
                "Test_F1": f"{test_f1:.4f}",
                "Gap_F1": f"{gap_f1:+.4f}",
            }
        )
    return rows


def _prefix_choice_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """统计不同分类器最终偏好的前缀 rank/size 与强度比例。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [
            row
            for row in subject_rows
            if str(row["Classifier"]) == classifier_name and str(row.get("SelectionMode", "")) == "transfer"
        ]
        if not classifier_rows:
            rows.append(
                {
                    "Classifier": classifier_name,
                    "TopPrefixRank": "0",
                    "TopPrefixSize": "0",
                    "TopChoiceCount": 0,
                    "MeanDesiredRatio": "0.0000",
                    "MeanActualRatio": "0.0000",
                }
            )
            continue
        choice_counter: Dict[Tuple[str, str], int] = {}
        ratio_values: List[float] = []
        actual_ratio_values: List[float] = []
        for row in classifier_rows:
            key = (str(row.get("PrefixRank", "")), str(row.get("PrefixSize", "")))
            choice_counter[key] = choice_counter.get(key, 0) + 1
            ratio_values.append(float(row.get("DesiredRatio", 0.0)))
            actual_ratio_values.append(float(row.get("ActualRatio", 0.0)))
        top_choice = sorted(choice_counter.items(), key=lambda item: (-int(item[1]), item[0][0], item[0][1]))[0]
        rows.append(
            {
                "Classifier": classifier_name,
                "TopPrefixRank": top_choice[0][0],
                "TopPrefixSize": top_choice[0][1],
                "TopChoiceCount": int(top_choice[1]),
                "MeanDesiredRatio": f"{float(np.mean(ratio_values)):.4f}",
                "MeanActualRatio": f"{float(np.mean(actual_ratio_values)):.4f}",
            }
        )
    return rows


def _source_frequency_rows(subject_rows: Sequence[Dict[str, Any]], top_k: int = 12) -> List[Dict[str, Any]]:
    """统计各分类器最终结果里最常出现的源域被试。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row["Classifier"]) == classifier_name]
        frequency: Dict[str, int] = {}
        for row in classifier_rows:
            raw_sources = str(row.get("TopSources", "")).strip()
            if not raw_sources:
                continue
            for source_id in [item.strip() for item in raw_sources.split(",") if item.strip()]:
                frequency[source_id] = frequency.get(source_id, 0) + 1
        sorted_items = sorted(frequency.items(), key=lambda item: (-int(item[1]), str(item[0])))[:top_k]
        for source_id, count in sorted_items:
            rows.append(
                {
                    "Classifier": classifier_name,
                    "SourceId": str(source_id),
                    "ChosenCount": int(count),
                }
            )
    return rows


def _jda_diagnostic_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """汇总最终结果中的 JDA 诊断信息，便于判断是否出现伪标签抖动或对齐失效。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row.get("Classifier")) == classifier_name]
        jda_rows = [row for row in classifier_rows if int(row.get("JDAIterations", 0)) > 0]
        if not jda_rows:
            continue
        rows.append(
            {
                "Classifier": classifier_name,
                "Units": int(len(jda_rows)),
                "MeanJDAIter": f"{float(np.mean([int(row.get('JDAIterations', 0)) for row in jda_rows])):.2f}",
                "MeanEndMMD": f"{float(np.mean([float(row.get('JDALastMMD', 0.0)) for row in jda_rows])):.4f}",
                "MeanMMDDelta": f"{float(np.mean([float(row.get('JDAMMDDelta', 0.0)) for row in jda_rows])):+.4f}",
                "MeanPseudoChange": f"{float(np.mean([float(row.get('JDAPseudoChange', 0.0)) for row in jda_rows])):.4f}",
                "MeanConfidence": f"{float(np.mean([float(row.get('JDAConfidence', 0.0)) for row in jda_rows])):.4f}",
            }
        )
    return rows


def _load_reference_summary(config: TransferLearningConfig) -> tuple[Path | None, Dict[str, Any] | None]:
    """尝试加载上一轮可直接对比的迁移学习结果。"""
    candidate_paths: List[Path] = []
    if config.reference_summary_path is not None:
        candidate_paths.append(ensure_ai_output_path(config.reference_summary_path))
    else:
        artifact_dir = Path("AI/artifacts")
        if artifact_dir.exists():
            candidate_paths.extend(
                sorted(
                    [path for path in artifact_dir.glob("*.json") if path != config.summary_path],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )

    for path in candidate_paths:
        try:
            json_obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(json_obj, dict) and "transfer_summary" in json_obj:
            return path, json_obj
    return None, None


def _legacy_reference_rows(reference_json: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """把上一轮迁移结果整理成简洁对照表。"""
    if not reference_json:
        return []
    reference_summary = reference_json.get("transfer_summary")
    if not isinstance(reference_summary, dict):
        return []

    rows: List[Dict[str, Any]] = []
    for classifier_name in [name for name in CLASSIFIER_ORDER if name in reference_summary]:
        classifier_block = reference_summary.get(classifier_name)
        if not isinstance(classifier_block, dict):
            continue
        acc_block = classifier_block.get("ACC")
        f1_block = classifier_block.get("F1")
        acc_value = float(acc_block.get("mean", acc_block)) if isinstance(acc_block, dict) else float(acc_block)
        f1_value = float(f1_block.get("mean", f1_block)) if isinstance(f1_block, dict) else float(f1_block)
        rows.append(
            {
                "Classifier": classifier_name,
                "Prev_ACC": f"{acc_value:.4f}",
                "Prev_F1": f"{f1_value:.4f}",
            }
        )
    return rows


def _reference_delta_rows(
    reference_json: Dict[str, Any] | None,
    current_summary: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, Any]]:
    """计算当前迁移结果相对上一轮迁移结果的 ACC/F1 变化。"""
    if not reference_json:
        return []
    reference_summary = reference_json.get("transfer_summary")
    if not isinstance(reference_summary, dict):
        return []
    return _gain_rows(
        reference_summary=reference_summary,
        current_summary=current_summary,
        reference_label="Prev",
        current_label="Current",
    )


def _artifact_file_name(path: Path | str) -> str:
    """提取产物路径中的文件名，便于在报告里快速对照 JSON/MD。"""
    return Path(path).name


def _transfer_report_artifact_rows(config: TransferLearningConfig) -> List[Dict[str, str]]:
    """整理迁移阶段关键产物文件名及用途说明。"""
    return [
        {
            "Artifact": "summary_json",
            "FileName": _artifact_file_name(config.summary_path),
            "Purpose": "迁移阶段结构化摘要，保存最终指标、迁移/回退结果与全部超参数。",
        },
        {
            "Artifact": "report_md",
            "FileName": _artifact_file_name(config.report_path),
            "Purpose": "迁移阶段中文可视化报告，便于查看验证集/测试集表现与错误来源。",
        },
        {
            "Artifact": "mask_pkl",
            "FileName": _artifact_file_name(config.mask_path),
            "Purpose": "阶段一冻结特征掩码；迁移阶段会从这里读取特征子集。",
        },
    ]


def _transfer_basic_param_rows(config: TransferLearningConfig) -> List[Dict[str, str]]:
    """整理迁移阶段的基础实验口径参数及其中文作用。"""
    return [
        {
            "Parameter": "feature_source_mode",
            "Value": str(config.normalized_feature_source_mode()),
            "Meaning": "决定迁移阶段读取阶段一 mask 还是直接使用全部原始特征。",
        },
        {
            "Parameter": "feature_source_label",
            "Value": str(config.normalized_feature_source_label()),
            "Meaning": "给本次特征入口打标签，方便日志、JSON 和报告定位同一实验口径。",
        },
        {
            "Parameter": "evaluation_protocol",
            "Value": str(config.normalized_protocol()),
            "Meaning": "迁移阶段的外层评估协议；通常是按目标被试逐个留出的 LOSO。",
        },
        {
            "Parameter": "cv_splits",
            "Value": str(int(config.cv_splits)),
            "Meaning": "协议为 group_kfold 时的计划折数；LOSO 下主要用于统一配置记录。",
        },
        {
            "Parameter": "max_target_subjects",
            "Value": "all" if config.max_target_subjects is None else str(int(config.max_target_subjects)),
            "Meaning": "本次最多跑多少个目标被试；调试时可裁剪，被设为 all 表示全量。",
        },
        {
            "Parameter": "target_train_ratio/val_ratio/test_ratio",
            "Value": f"{float(config.target_train_ratio):.2f}/{float(config.target_val_ratio):.2f}/{float(config.target_test_ratio):.2f}",
            "Meaning": "目标域 train/val/test 的样本比例；本项目固定为 60/20/20。",
        },
        {
            "Parameter": "transfer_variant",
            "Value": str(config.normalized_transfer_variant()),
            "Meaning": "选择监督迁移还是增强 JDA 迁移分支；会直接影响后续训练流程。",
        },
    ]


def _transfer_source_param_rows(config: TransferLearningConfig) -> List[Dict[str, str]]:
    """整理源域筛选、采样和权重相关参数及其中文作用。"""
    return [
        {
            "Parameter": "max_source_subjects",
            "Value": str(int(config.max_source_subjects)),
            "Meaning": "最终最多保留多少个源被试进入迁移训练，避免源域过多拖慢训练并带入噪声。",
        },
        {
            "Parameter": "source_sample_cap",
            "Value": "full" if config.source_sample_cap is None else str(int(config.source_sample_cap)),
            "Meaning": "每个源被试最多抽多少样本用于训练；越大越稳，越小越省时。",
        },
        {
            "Parameter": "mmd_sample_cap",
            "Value": "full" if config.mmd_sample_cap is None else str(int(config.mmd_sample_cap)),
            "Meaning": "计算 MMD 时每个域最多抽多少样本，控制相似度估计成本。",
        },
        {
            "Parameter": "mmd_prefix_top_k",
            "Value": str(config.normalized_mmd_prefix_label()),
            "Meaning": "累计前缀 MMD 搜索上限；越大越充分，但迁移筛选更耗时。",
        },
        {
            "Parameter": "target_repeat_grid",
            "Value": str(list(config.normalized_target_repeat_grid())),
            "Meaning": "目标域样本在混合训练时允许重复的倍数网格，用来平衡源域/目标域强度。",
        },
        {
            "Parameter": "source_target_ratio_grid",
            "Value": str(list(config.normalized_source_target_ratio_grid())),
            "Meaning": "搜索源域有效权重与目标域有效权重比例的候选网格。",
        },
        {
            "Parameter": "source_positive_ratio_gap_threshold",
            "Value": f"{config.normalized_source_positive_ratio_gap_threshold():.4f}",
            "Meaning": "源被试与目标训练集正类比例允许的最大差距；超过就先过滤掉。",
        },
        {
            "Parameter": "use_inverse_mmd_source_weighting",
            "Value": str(bool(config.use_inverse_mmd_source_weighting)),
            "Meaning": "是否按 inverse-MMD 给更像目标域的源被试更高权重。",
        },
        {
            "Parameter": "use_positive_ratio_source_weighting",
            "Value": str(bool(config.use_positive_ratio_source_weighting)),
            "Meaning": "是否额外按类别比例接近度调节源被试权重，减小类别失配影响。",
        },
        {
            "Parameter": "positive_ratio_weight_strength",
            "Value": f"{float(config.positive_ratio_weight_strength):.4f}",
            "Meaning": "类别比例接近度权重的放大强度；越大越重视分布匹配。",
        },
        {
            "Parameter": "inverse_mmd_weight_epsilon",
            "Value": f"{float(config.inverse_mmd_weight_epsilon):.6f}",
            "Meaning": "inverse-MMD 权重的数值稳定项，避免极小分母导致权重爆炸。",
        },
    ]


def _transfer_gate_param_rows(config: TransferLearningConfig) -> List[Dict[str, str]]:
    """整理 gate 与回退机制参数及其中文作用。"""
    return [
        {
            "Parameter": "gate_repeat_count",
            "Value": str(int(config.normalized_gate_repeat_count())),
            "Meaning": "主验证切分没通过 gate 时，允许重复复核多少次，减少偶然切分误杀。",
        },
        {
            "Parameter": "gate_thresholds",
            "Value": str(_classifier_gate_threshold_text()),
            "Meaning": "各分类器要保留迁移时，验证集相对 target-only 至少需要达到的 ACC/F1 增益阈值。",
        },
    ]


def _transfer_jda_param_rows(config: TransferLearningConfig) -> List[Dict[str, str]]:
    """整理增强 JDA 相关参数及其中文作用；监督迁移时也保留默认值方便审计。"""
    jda_enabled_text = "启用" if config.normalized_transfer_variant() == "enhanced_jda" else "未启用，仅记录默认值"
    return [
        {
            "Parameter": "jda_dim",
            "Value": str(int(config.normalized_jda_dim())),
            "Meaning": f"JDA 投影维度，控制域对齐后的特征空间大小；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_iterations",
            "Value": str(int(config.normalized_jda_iterations())),
            "Meaning": f"JDA 迭代轮数上限，决定伪标签更新和分布对齐可持续多久；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_n_components",
            "Value": str(int(config.normalized_jda_n_components())),
            "Meaning": f"JDA 核子空间保留成分数，用来控制对齐表示容量；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_lambda",
            "Value": f"{float(config.jda_lambda):.4f}",
            "Meaning": f"JDA 分布对齐项权重，越大越强调跨域分布贴近；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_reg",
            "Value": f"{float(config.jda_reg):.6f}",
            "Meaning": f"JDA 数值稳定正则，防止矩阵求解病态；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_pseudo_labeler",
            "Value": str(config.jda_pseudo_labeler),
            "Meaning": f"JDA 生成伪标签时使用的分类器类型；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_pseudo_neighbors",
            "Value": str(int(config.jda_pseudo_neighbors)),
            "Meaning": f"伪标签器的近邻数，仅在近邻式伪标签器下生效；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_pseudo_change_tol",
            "Value": f"{float(config.jda_pseudo_change_tol):.6f}",
            "Meaning": f"伪标签变化率收敛阈值，越小表示要求更稳定才认为收敛；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_mmd_delta_tol",
            "Value": f"{float(config.jda_mmd_delta_tol):.6f}",
            "Meaning": f"MMD 变化收敛阈值，用来判断域对齐收益是否已趋于稳定；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_confidence_delta_tol",
            "Value": f"{float(config.jda_confidence_delta_tol):.6f}",
            "Meaning": f"伪标签平均置信度变化阈值，帮助判断后续迭代是否还在明显变化；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_early_stop_patience",
            "Value": str(int(config.jda_early_stop_patience)),
            "Meaning": f"连续满足稳定条件多少轮后允许提前停止 JDA；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_min_iterations",
            "Value": str(int(config.jda_min_iterations)),
            "Meaning": f"JDA 至少运行多少轮，避免过早停止；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_pseudo_keep_ratio_grid",
            "Value": str(list(config.normalized_jda_pseudo_keep_ratio_grid())),
            "Meaning": f"JDA 中保留高置信伪标签样本比例的搜索网格；当前{jda_enabled_text}。",
        },
        {
            "Parameter": "jda_pseudo_target_repeat_grid",
            "Value": str(list(config.normalized_jda_pseudo_target_repeat_grid())),
            "Meaning": f"JDA 中高置信伪标签样本的重复倍数网格；当前{jda_enabled_text}。",
        },
    ]


def _write_transfer_report(
    config: TransferLearningConfig,
    selected_feature_count: int,
    stage1_lines: Sequence[str],
    split_overview_rows: Sequence[Dict[str, Any]],
    mmd_selection_rows: Sequence[Dict[str, Any]],
    gate_summary_rows: Sequence[Dict[str, Any]],
    transfer_decision_rows: Sequence[Dict[str, Any]],
    val_test_gap_rows: Sequence[Dict[str, Any]],
    prefix_choice_rows: Sequence[Dict[str, Any]],
    source_frequency_rows: Sequence[Dict[str, Any]],
    source_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    target_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    raw_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
    final_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
    method_comparison_rows: Sequence[Dict[str, Any]],
    gain_vs_source_rows: Sequence[Dict[str, Any]],
    gain_vs_target_rows: Sequence[Dict[str, Any]],
    subject_rows: Sequence[Dict[str, Any]],
    hard_rows: Sequence[Dict[str, Any]],
    legacy_reference_path: Path | None,
    legacy_reference_rows: Sequence[Dict[str, Any]],
    reference_delta_rows: Sequence[Dict[str, Any]],
) -> Path:
    """生成监督迁移阶段的 Markdown 报告。"""
    source_only_bar_rows = _summary_bar_rows(source_only_summary)
    target_only_bar_rows = _summary_bar_rows(target_only_summary)
    transfer_bar_rows = _summary_bar_rows(final_transfer_summary)

    intro_lines = [
        "本报告固定使用目标域 `60% train / 20% val / 20% test` 划分。",
        "对比基线包括只用源域训练的 `source-only`，以及只用目标域训练的 `target-only`。",
        f"当前特征入口为 `{config.normalized_feature_source_label()}`，监督迁移阶段先按累计前缀 MMD 做 `top-k` 初筛，再让 `LR/KNN/DT` 各自在验证集上复排并挑选自己的最优源域前缀。",
        "训练时不再直接扫源域绝对缩放，而是约束 `源域有效强度 / 目标域有效强度` 的比例，同时按目标单元动态融合 MMD 相似度与正类比例接近度来调整源被试权重。",
        f"对应 JSON 摘要文件为 `{_artifact_file_name(config.summary_path)}`。",
    ]
    sections: List[Dict[str, Any]] = [
        {
            "title": "运行配置",
            "body_lines": [
                f"- feature_source_label: `{config.normalized_feature_source_label()}`",
                f"- feature_source_mode: `{config.normalized_feature_source_mode()}`",
                f"- mask_path: `{config.mask_path if config.normalized_feature_source_mode() == 'mask' else 'full_feature_mode'}`",
                f"- 选中特征数: `{selected_feature_count}`",
                f"- evaluation_protocol: `{config.normalized_protocol()}`",
                f"- max_target_subjects: `{config.max_target_subjects or 'all'}`",
                f"- max_source_subjects: `{config.max_source_subjects}`",
                f"- source_sample_cap: `{config.source_sample_cap}`",
                f"- mmd_sample_cap: `{config.mmd_sample_cap}`",
                f"- mmd_prefix_top_k: `{config.normalized_mmd_prefix_top_k()}`",
                f"- target_repeat_grid: `{list(config.normalized_target_repeat_grid())}`",
                f"- source_target_ratio_grid: `{list(config.normalized_source_target_ratio_grid())}`",
                f"- 对应 JSON 摘要文件: `{_artifact_file_name(config.summary_path)}`",
                f"- 对应 Markdown 报告文件: `{_artifact_file_name(config.report_path)}`",
                f"- source_weighting_mode: `{'inverse_mmd_ratio_align_mean1' if config.use_inverse_mmd_source_weighting and config.use_positive_ratio_source_weighting else 'inverse_mmd_normalized_mean1' if config.use_inverse_mmd_source_weighting else 'disabled'}`",
                "- 说明：LR/DT 用样本权重实现比例约束，KNN 用加权重采样来近似目标比例。",
            ],
        },
        {
            "title": "产物文件",
            "table": format_metric_table(
                _transfer_report_artifact_rows(config),
                [("Artifact", "产物类型"), ("FileName", "文件名"), ("Purpose", "作用")],
            ),
        },
        {
            "title": "参数注释：基础口径",
            "table": format_metric_table(
                _transfer_basic_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：源域筛选与权重",
            "table": format_metric_table(
                _transfer_source_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：Gate 与回退",
            "table": format_metric_table(
                _transfer_gate_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：JDA",
            "table": format_metric_table(
                _transfer_jda_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "阶段一参考",
            "body_lines": list(stage1_lines),
        },
        {
            "title": "目标域切分概览",
            "table": format_metric_table(
                split_overview_rows,
                [
                    "TargetUnit",
                    "TargetSubjects",
                    "Train",
                    "Val",
                    "Test",
                    "TrainPosRatio",
                    "ValPosRatio",
                    "TestPosRatio",
                    "Top3Sources",
                    "ChosenPrefix",
                    "ChosenMMD",
                ],
            ),
        },
        {
            "title": "MMD Top-K 初筛",
            "table": format_metric_table(
                mmd_selection_rows,
                ["TargetUnit", "BestMMDPrefix", "BestMMD", "BestMMDSources", "TopKPreview", "Top3Sources", "WeightPreview"],
            ),
        },
    ]

    if legacy_reference_rows:
        sections.append(
            {
                "title": "上一轮迁移结果",
                "body_lines": [f"- 参考文件: `{legacy_reference_path}`"] if legacy_reference_path else [],
                "table": format_metric_table(legacy_reference_rows, ["Classifier", "Prev_ACC", "Prev_F1"]),
            }
        )

    if reference_delta_rows:
        sections.append(
            {
                "title": "相对上一轮的变化",
                "table": format_metric_table(
                    reference_delta_rows,
                    ["Classifier", "Prev_ACC", "Current_ACC", "Delta_ACC", "Prev_F1", "Current_F1", "Delta_F1"],
                ),
            }
        )

    if gate_summary_rows:
        sections.append(
            {
                "title": "Gate 统计",
                "table": format_metric_table(
                    gate_summary_rows,
                    ["Classifier", "Units", "TransferKept", "TargetOnlyGate", "GateRate", "TopGateReason"],
                ),
            }
        )

    if transfer_decision_rows:
        sections.append(
            {
                "title": "迁移/回退明细",
                "table": format_metric_table(
                    transfer_decision_rows,
                    ["Classifier", "TransferCount", "FallbackCount", "TransferUnits", "FallbackUnits", "FallbackReasons"],
                ),
            }
        )

    if val_test_gap_rows:
        sections.append(
            {
                "title": "验证到测试落差",
                "table": format_metric_table(
                    val_test_gap_rows,
                    ["Classifier", "Val_ACC", "Test_ACC", "Gap_ACC", "Val_F1", "Test_F1", "Gap_F1"],
                ),
            }
        )

    if prefix_choice_rows:
        sections.append(
            {
                "title": "前缀选择分布",
                "table": format_metric_table(
                    prefix_choice_rows,
                    ["Classifier", "TopPrefixRank", "TopPrefixSize", "TopChoiceCount", "MeanDesiredRatio", "MeanActualRatio"],
                ),
            }
        )

    if source_frequency_rows:
        sections.append(
            {
                "title": "高频源域",
                "table": format_metric_table(
                    source_frequency_rows,
                    ["Classifier", "SourceId", "ChosenCount"],
                ),
            }
        )

    sections.extend(
        [
            {
                "title": "Source-Only 基线",
                "body_lines": format_metric_bars("Source-Only ACC", source_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("Source-Only F1", source_only_bar_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(source_only_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            },
            {
                "title": "Target-Only 基线",
                "body_lines": format_metric_bars("Target-Only ACC", target_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("Target-Only F1", target_only_bar_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(target_only_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            },
            {
                "title": "监督迁移结果",
                "body_lines": format_metric_bars("Transfer ACC", transfer_bar_rows, "ACC")
                + [""]
                + format_metric_bars("Transfer F1", transfer_bar_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(transfer_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            },
            {
                "title": "相对 Source-Only 的变化",
                "table": format_metric_table(
                    gain_vs_source_rows,
                    ["Classifier", "SourceOnly_ACC", "Transfer_ACC", "Delta_ACC", "SourceOnly_F1", "Transfer_F1", "Delta_F1"],
                ),
            },
            {
                "title": "相对 Target-Only 的变化",
                "table": format_metric_table(
                    gain_vs_target_rows,
                    ["Classifier", "TargetOnly_ACC", "Transfer_ACC", "Delta_ACC", "TargetOnly_F1", "Transfer_F1", "Delta_F1"],
                ),
            },
            {
                "title": "单元级结果",
                "table": format_metric_table(
                    subject_rows,
                    [
                        "TargetUnit",
                        "Classifier",
                        "SourceOnly_ACC",
                        "TargetOnly_ACC",
                        "Transfer_ACC",
                        "Transfer_F1",
                        "DeltaVsSource_ACC",
                        "DeltaVsTarget_ACC",
                        "Val_ACC",
                        "Val_F1",
                        "ValTestGap_ACC",
                        "ValTestGap_F1",
                        "PrefixSize",
                        "PrefixRank",
                        "PrefixMMD",
                        "TargetRepeat",
                        "DesiredRatio",
                        "ActualRatio",
                        "SelectionMode",
                        "GateReason",
                        "Threshold",
                        "TopSources",
                    ],
                ),
            },
            {
                "title": "最难目标单元",
                "table": format_metric_table(
                    hard_rows,
                    [
                        "TargetUnit",
                        "Classifier",
                        "Transfer_ACC",
                        "Transfer_F1",
                        "DeltaVsSource_ACC",
                        "DeltaVsTarget_ACC",
                        "PrefixMMD",
                        "Threshold",
                        "TopSources",
                    ],
                ),
            },
        ]
    )

    return write_markdown_report(
        path=config.report_path,
        title="监督迁移学习报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def _normalize_classifier_masks_for_transfer(
    mask_artifact: Dict[str, Any],
    total_feature_count: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], List[Dict[str, Any]]]:
    """把阶段一 mask 结果规范成按分类器索引的统一结构，并兼容旧版共享 mask 产物。"""
    shared_indices = np.asarray(mask_artifact.get("selected_raw_indices", []), dtype=np.int32)
    raw_dict = mask_artifact.get("selected_raw_indices_by_classifier", {})
    count_dict = mask_artifact.get("selected_feature_count_by_classifier", {})
    frozen_summary = mask_artifact.get("frozen_mask_baseline_summary", {})

    selected_raw_indices_by_classifier: Dict[str, np.ndarray] = {}
    selected_feature_count_by_classifier: Dict[str, int] = {}
    stage1_classifier_rows: List[Dict[str, Any]] = []

    for classifier_name in CLASSIFIER_ORDER:
        classifier_indices = np.asarray(raw_dict.get(classifier_name, shared_indices), dtype=np.int32)
        if classifier_indices.size == 0:
            raise ValueError(f"mask 文件中缺少 {classifier_name} 的 selected_raw_indices，无法继续监督迁移。")
        selected_raw_indices_by_classifier[classifier_name] = classifier_indices
        selected_feature_count_by_classifier[classifier_name] = int(
            count_dict.get(classifier_name, len(classifier_indices))
        )
        classifier_metrics = frozen_summary.get(classifier_name, {})
        stage1_classifier_rows.append(
            {
                "Classifier": classifier_name,
                "FeatureCount": int(selected_feature_count_by_classifier[classifier_name]),
                "Stage1_ACC": f"{float(classifier_metrics.get('ACC', 0.0)):.4f}",
                "Stage1_F1": f"{float(classifier_metrics.get('F1', 0.0)):.4f}",
            }
        )

    if not selected_raw_indices_by_classifier:
        raise ValueError("mask 文件中没有可用的分类器 mask。")
    if total_feature_count <= 0:
        raise ValueError("total_feature_count 必须为正整数。")
    return selected_raw_indices_by_classifier, selected_feature_count_by_classifier, stage1_classifier_rows


def _classifier_feature_count_text(selected_feature_count_by_classifier: Dict[str, int]) -> str:
    """把迁移阶段按分类器使用的特征数压缩成便于打印的文本。"""
    return ", ".join(
        f"{classifier_name}={int(selected_feature_count_by_classifier[classifier_name])}"
        for classifier_name in CLASSIFIER_ORDER
    )


def _write_classifier_mask_transfer_report(
    config: TransferLearningConfig,
    selected_feature_count_by_classifier: Dict[str, int],
    stage1_lines: Sequence[str],
    stage1_classifier_rows: Sequence[Dict[str, Any]],
    split_overview_rows: Sequence[Dict[str, Any]],
    mmd_selection_rows: Sequence[Dict[str, Any]],
    gate_summary_rows: Sequence[Dict[str, Any]],
    transfer_decision_rows: Sequence[Dict[str, Any]],
    val_test_gap_rows: Sequence[Dict[str, Any]],
    prefix_choice_rows: Sequence[Dict[str, Any]],
    source_frequency_rows: Sequence[Dict[str, Any]],
    source_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    target_only_summary: Dict[str, Dict[str, Dict[str, float]]],
    raw_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
    final_transfer_summary: Dict[str, Dict[str, Dict[str, float]]],
    method_comparison_rows: Sequence[Dict[str, Any]],
    gain_vs_source_rows: Sequence[Dict[str, Any]],
    gain_vs_target_rows: Sequence[Dict[str, Any]],
    subject_rows: Sequence[Dict[str, Any]],
    hard_rows: Sequence[Dict[str, Any]],
    legacy_reference_path: Path | None,
    legacy_reference_rows: Sequence[Dict[str, Any]],
    reference_delta_rows: Sequence[Dict[str, Any]],
) -> Path:
    """生成分类器专属 mask 版本的监督迁移 Markdown 报告。"""
    transfer_variant = config.normalized_transfer_variant()
    transfer_variant_label = _transfer_variant_label(transfer_variant)
    source_only_bar_rows = _summary_bar_rows(source_only_summary)
    target_only_bar_rows = _summary_bar_rows(target_only_summary)
    raw_transfer_bar_rows = _summary_bar_rows(raw_transfer_summary)
    final_transfer_bar_rows = _summary_bar_rows(final_transfer_summary)
    jda_diag_rows = _jda_diagnostic_rows(subject_rows)

    intro_lines = [
        "本报告固定使用目标域 `60% train / 20% val / 20% test` 划分。",
        "对比基线包括只用源域训练的 `source-only`，以及只用目标域训练的 `target-only`。",
        f"当前特征入口为 `{config.normalized_feature_source_label()}`，迁移阶段改为 `LR/KNN/DT` 各自读取自己的阶段一 mask。",
        "目标域切分先在每个目标被试内部按等量整数样本做分层切分，再拼成统一 train/val/test，避免不同被试对 MMD 对齐贡献失衡。",
        "源域搜索不再只看 top-k，而是把 `前1、前2、前3...` 的累计前缀全部纳入候选，再让分类器在验证集上挑自己的最优前缀。",
        "源域进入 MMD 排序前，会先按目标域训练集的正类比例做一轮过滤，尽量剔除类别分布差太远的源被试。",
        "如果主验证切分没有通过 gate，会再做重复切分复核，减少单次验证切分导致的误杀。",
        "训练时不再直接扫源域绝对缩放，而是约束 `源域有效强度 / 目标域有效强度` 的比例，同时按目标单元动态融合 MMD 相似度与正类比例接近度来调整源被试权重。",
        f"当前迁移训练分支为 `{transfer_variant_label}`。",
        *( [f"JDA 配置: `{_jda_config_text(config)}`"] if transfer_variant == "enhanced_jda" else [] ),
        f"对应 JSON 摘要文件为 `{_artifact_file_name(config.summary_path)}`。",
        "`raw transfer` 表示迁移候选在 gate 之前的原始结果，`final transfer` 表示和 `target-only` 比较并执行保留/回退后的最终结果。",
    ]
    sections: List[Dict[str, Any]] = [
        {
            "title": "运行配置",
            "body_lines": [
                f"- feature_source_label: `{config.normalized_feature_source_label()}`",
                f"- feature_source_mode: `{config.normalized_feature_source_mode()}`",
                f"- mask_path: `{config.mask_path if config.normalized_feature_source_mode() == 'mask' else 'full_feature_mode'}`",
                f"- 分类器专属特征数: `{_classifier_feature_count_text(selected_feature_count_by_classifier)}`",
                f"- evaluation_protocol: `{config.normalized_protocol()}`",
                f"- max_target_subjects: `{config.max_target_subjects or 'all'}`",
                f"- max_source_subjects: `{config.max_source_subjects}`",
                f"- source_sample_cap: `{config.source_sample_cap}`",
                f"- mmd_sample_cap: `{config.mmd_sample_cap}`",
                f"- mmd_prefix_search: `{config.normalized_mmd_prefix_label()}`",
                f"- target_repeat_grid: `{list(config.normalized_target_repeat_grid())}`",
                f"- source_target_ratio_grid: `{list(config.normalized_source_target_ratio_grid())}`",
                f"- transfer_variant: `{transfer_variant}`",
                f"- source_positive_ratio_gap_threshold: `{config.normalized_source_positive_ratio_gap_threshold():.4f}`",
                f"- knn_source_positive_ratio_gap_threshold: `{_classifier_source_filter_gap_threshold('KNN', config):.4f}`",
                f"- gate_repeat_count: `{config.normalized_gate_repeat_count()}`",
                f"- gate_thresholds: `{_classifier_gate_threshold_text()}`",
                *( [f"- jda_config: `{_jda_config_text(config)}`"] if transfer_variant == "enhanced_jda" else [] ),
                f"- 对应 JSON 摘要文件: `{_artifact_file_name(config.summary_path)}`",
                f"- 对应 Markdown 报告文件: `{_artifact_file_name(config.report_path)}`",
                f"- source_weighting_mode: `{'inverse_mmd_ratio_align_mean1' if config.use_inverse_mmd_source_weighting and config.use_positive_ratio_source_weighting else 'inverse_mmd_normalized_mean1' if config.use_inverse_mmd_source_weighting else 'disabled'}`",
                "- 说明：LR/DT 用样本权重实现比例约束，KNN 用加权重采样来近似目标比例。",
            ],
        },
        {
            "title": "产物文件",
            "table": format_metric_table(
                _transfer_report_artifact_rows(config),
                [("Artifact", "产物类型"), ("FileName", "文件名"), ("Purpose", "作用")],
            ),
        },
        {
            "title": "参数注释：基础口径",
            "table": format_metric_table(
                _transfer_basic_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：源域筛选与权重",
            "table": format_metric_table(
                _transfer_source_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：Gate 与回退",
            "table": format_metric_table(
                _transfer_gate_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：JDA",
            "table": format_metric_table(
                _transfer_jda_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "阶段一参考",
            "body_lines": list(stage1_lines),
            "table": format_metric_table(stage1_classifier_rows, ["Classifier", "FeatureCount", "Stage1_ACC", "Stage1_F1"]),
        },
        {
            "title": "目标域切分概览",
            "table": format_metric_table(
                split_overview_rows,
                [
                    "Classifier",
                    "MaskFeatureCount",
                    "TargetUnit",
                    "TargetSubjects",
                    "Train",
                    "Val",
                    "Test",
                    "SourceFilterKept",
                    "SplitMode",
                    "PerSubjectSplit",
                    "TrainPosRatio",
                    "ValPosRatio",
                    "TestPosRatio",
                    "SourceRankPreview",
                    "ChosenPrefix",
                    "ChosenMMD",
                ],
            ),
        },
        {
            "title": "累计前缀 MMD 排序",
            "table": format_metric_table(
                mmd_selection_rows,
                [
                    "Classifier",
                    "MaskFeatureCount",
                    "TargetUnit",
                    "BestMMDPrefix",
                    "BestMMD",
                    "SourceFilterKept",
                    "BestMMDSources",
                    "PrefixRankPreview",
                    "SourceRankPreview",
                    "WeightPreview",
                ],
            ),
        },
    ]

    if legacy_reference_rows:
        sections.append(
            {
                "title": "上一轮迁移结果",
                "body_lines": [f"- 参考文件: `{legacy_reference_path}`"] if legacy_reference_path else [],
                "table": format_metric_table(legacy_reference_rows, ["Classifier", "Prev_ACC", "Prev_F1"]),
            }
        )

    if reference_delta_rows:
        sections.append(
            {
                "title": "相对上一轮的变化",
                "table": format_metric_table(
                    reference_delta_rows,
                    ["Classifier", "Prev_ACC", "Current_ACC", "Delta_ACC", "Prev_F1", "Current_F1", "Delta_F1"],
                ),
            }
        )

    if gate_summary_rows:
        sections.append(
            {
                "title": "Gate 统计",
                "table": format_metric_table(
                    gate_summary_rows,
                    ["Classifier", "Units", "TransferKept", "TargetOnlyGate", "GateRate", "TopGateReason"],
                ),
            }
        )

    if transfer_decision_rows:
        sections.append(
            {
                "title": "迁移/回退明细",
                "table": format_metric_table(
                    transfer_decision_rows,
                    ["Classifier", "TransferCount", "FallbackCount", "TransferUnits", "FallbackUnits", "FallbackReasons"],
                ),
            }
        )

    if val_test_gap_rows:
        sections.append(
            {
                "title": "验证到测试落差",
                "table": format_metric_table(
                    val_test_gap_rows,
                    ["Classifier", "Val_ACC", "Test_ACC", "Gap_ACC", "Val_F1", "Test_F1", "Gap_F1"],
                ),
            }
        )

    if prefix_choice_rows:
        sections.append(
            {
                "title": "前缀选择分布",
                "table": format_metric_table(
                    prefix_choice_rows,
                    ["Classifier", "TopPrefixRank", "TopPrefixSize", "TopChoiceCount", "MeanDesiredRatio", "MeanActualRatio"],
                ),
            }
        )

    if source_frequency_rows:
        sections.append(
            {
                "title": "高频源域",
                "table": format_metric_table(source_frequency_rows, ["Classifier", "SourceId", "ChosenCount"]),
            }
        )

    if jda_diag_rows:
        sections.append(
            {
                "title": "JDA 诊断",
                "table": format_metric_table(
                    jda_diag_rows,
                    ["Classifier", "Units", "MeanJDAIter", "MeanEndMMD", "MeanMMDDelta", "MeanPseudoChange", "MeanConfidence"],
                ),
            }
        )

    if method_comparison_rows:
        sections.append(
            {
                "title": "四路方法对照",
                "body_lines": [
                    "- `source-only`: 只用源域训练。",
                    "- `target-only`: 只用目标域训练。",
                    "- `raw transfer`: 迁移候选未经 gate 回退前的原始测试结果。",
                    "- `final transfer`: 执行 gate 后真正进入最终汇总的结果。",
                ],
                "table": format_metric_table(
                    method_comparison_rows,
                    [
                        "Classifier",
                        "SourceOnly_ACC",
                        "SourceOnly_F1",
                        "TargetOnly_ACC",
                        "TargetOnly_F1",
                        "RawTransfer_ACC",
                        "RawTransfer_F1",
                        "FinalTransfer_ACC",
                        "FinalTransfer_F1",
                    ],
                ),
            }
        )

    sections.extend(
        [
            {
                "title": "Source-Only 基线",
                "body_lines": format_metric_bars("source-only ACC", source_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("source-only F1", source_only_bar_rows, "F1"),
                "table": format_metric_table(source_only_bar_rows, ["Classifier", "ACC", "F1"]),
            },
            {
                "title": "Target-Only 基线",
                "body_lines": format_metric_bars("target-only ACC", target_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("target-only F1", target_only_bar_rows, "F1"),
                "table": format_metric_table(target_only_bar_rows, ["Classifier", "ACC", "F1"]),
            },
            {
                "title": "Raw Transfer 未门控结果",
                "body_lines": format_metric_bars("raw transfer ACC", raw_transfer_bar_rows, "ACC")
                + [""]
                + format_metric_bars("raw transfer F1", raw_transfer_bar_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(raw_transfer_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            },
            {
                "title": "Final Transfer 最终结果",
                "body_lines": format_metric_bars("final transfer ACC", final_transfer_bar_rows, "ACC")
                + [""]
                + format_metric_bars("final transfer F1", final_transfer_bar_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(final_transfer_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            },
            {
                "title": "相对 Source-Only 的增益",
                "table": format_metric_table(
                    gain_vs_source_rows,
                    ["Classifier", "SourceOnly_ACC", "Transfer_ACC", "Delta_ACC", "SourceOnly_F1", "Transfer_F1", "Delta_F1"],
                ),
            },
            {
                "title": "相对 Target-Only 的增益",
                "table": format_metric_table(
                    gain_vs_target_rows,
                    ["Classifier", "TargetOnly_ACC", "Transfer_ACC", "Delta_ACC", "TargetOnly_F1", "Transfer_F1", "Delta_F1"],
                ),
            },
            {
                "title": "目标单元明细",
                "table": format_metric_table(
                    subject_rows,
                    [
                        "TargetUnit",
                        "Classifier",
                        "MaskFeatureCount",
                        "SourceOnly_ACC",
                        "TargetOnly_ACC",
                        "RawTransfer_ACC",
                        "RawTransfer_F1",
                        "Val_ACC",
                        "Val_F1",
                        "Transfer_ACC",
                        "Transfer_F1",
                        "DeltaVsSource_ACC",
                        "DeltaVsTarget_ACC",
                        "RawDeltaVsTarget_ACC",
                        "RawDeltaVsTarget_F1",
                        "PrefixMMD",
                        "TransferVariant",
                        "AlignmentMethod",
                        "JDAIterations",
                        "JDALastMMD",
                        "JDAMMDDelta",
                        "JDAPseudoChange",
                        "JDAConfidence",
                        "SelectionMode",
                        "GateReason",
                        "GateRepeatPassed",
                        "GateRepeatMeanGain_ACC",
                        "GateRepeatMeanGain_F1",
                        "Threshold",
                        "TopSources",
                    ],
                ),
            },
            {
                "title": "最难目标单元",
                "table": format_metric_table(
                    hard_rows,
                    [
                        "TargetUnit",
                        "Classifier",
                        "MaskFeatureCount",
                        "RawTransfer_ACC",
                        "RawTransfer_F1",
                        "Transfer_ACC",
                        "Transfer_F1",
                        "DeltaVsSource_ACC",
                        "DeltaVsTarget_ACC",
                        "PrefixMMD",
                        "TransferVariant",
                        "JDALastMMD",
                        "JDAPseudoChange",
                        "Threshold",
                        "TopSources",
                    ],
                ),
            },
        ]
    )

    return write_markdown_report(
        path=config.report_path,
        title=f"{transfer_variant_label}报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def run_transfer_learning_pipeline(config: TransferLearningConfig | None = None) -> Dict[str, Any]:
    """运行带目标域监督的迁移/微调流程，并输出 JSON 与 Markdown 报告。"""
    if config is None:
        config = TransferLearningConfig()

    config.validate_ratios()
    ensure_artifact_dir()
    config.summary_path = ensure_ai_output_path(config.summary_path)
    config.report_path = ensure_ai_output_path(config.report_path)
    if config.normalized_feature_source_mode() == "mask":
        config.mask_path = ensure_ai_output_path(config.mask_path)
    if config.reference_summary_path is not None:
        config.reference_summary_path = ensure_ai_output_path(config.reference_summary_path)

    ensure_numpy_pickle_compat()
    X_raw, y, groups = load_deap_features()
    layout = dataset_layout_summary(len(y))
    feature_source_mode = config.normalized_feature_source_mode()
    feature_source_label = config.normalized_feature_source_label()
    transfer_variant = config.normalized_transfer_variant()
    if feature_source_mode == "full":
        selected_raw_indices_by_classifier = {
            classifier_name: np.arange(X_raw.shape[1], dtype=np.int32)
            for classifier_name in CLASSIFIER_ORDER
        }
        selected_feature_count_by_classifier = {
            classifier_name: int(X_raw.shape[1])
            for classifier_name in CLASSIFIER_ORDER
        }
        stage1_best_name = "full_feature_no_stage1"
        stage1_best_metrics = {"ACC": 0.0, "F1": 0.0}
        stage1_lines = [
            f"- 当前特征入口: `{feature_source_label}`",
            "- 当前模式直接使用全部原始特征，不依赖阶段一冻结掩码。",
            f"- 原始特征数: `{int(X_raw.shape[1])}`",
            "- 说明：该入口更适合判断迁移收益是否来自域适配，而不是来自特征筛选。",
        ]
        stage1_classifier_rows = [
            {
                "Classifier": classifier_name,
                "FeatureCount": int(X_raw.shape[1]),
                "Stage1_ACC": "-",
                "Stage1_F1": "-",
            }
            for classifier_name in CLASSIFIER_ORDER
        ]
    else:
        mask_artifact = load_pickle(config.mask_path)
        selected_raw_indices_by_classifier, selected_feature_count_by_classifier, stage1_classifier_rows = _normalize_classifier_masks_for_transfer(
            mask_artifact=mask_artifact,
            total_feature_count=int(X_raw.shape[1]),
        )
        stage1_best_name = str(mask_artifact.get("best_classifier_name", "unknown"))
        stage1_best_metrics = mask_artifact.get("best_classifier_metrics", {})
        stage1_lines = [
            f"- 当前特征入口: `{feature_source_label}`",
            f"- 阶段一冻结掩码最佳分类器: `{stage1_best_name}`",
            f"- 阶段一冻结掩码 ACC: `{float(stage1_best_metrics.get('ACC', 0.0)):.4f}`",
            f"- 阶段一冻结掩码 F1: `{float(stage1_best_metrics.get('F1', 0.0)):.4f}`",
            f"- 分类器专属特征数: `{_classifier_feature_count_text(selected_feature_count_by_classifier)}`",
            "- 说明：阶段一指标来自跨被试筛选，只能作为是否值得继续迁移的参考，不是当前 60/20/20 设置下的最终基线。",
        ]
    all_subject_ids = np.unique(groups).astype(np.int32).tolist()
    subject_blocks_by_classifier = {
        classifier_name: _build_subject_feature_blocks(
            X_raw=X_raw,
            y=y,
            groups=groups,
            selected_raw_indices=np.asarray(selected_raw_indices_by_classifier[classifier_name], dtype=np.int32),
        )
        for classifier_name in CLASSIFIER_ORDER
    }
    print(
        f"[Transfer] feature_source={feature_source_label} mode={feature_source_mode} "
        f"transfer_variant={transfer_variant} "
        f"classifier_feature_counts={_classifier_feature_count_text(selected_feature_count_by_classifier)} "
        f"subject_count={layout['subject_count']} samples_per_subject={layout['samples_per_subject']} "
        f"source_positive_ratio_gap_threshold={config.normalized_source_positive_ratio_gap_threshold():.4f} "
        f"knn_source_positive_ratio_gap_threshold={_classifier_source_filter_gap_threshold('KNN', config):.4f} "
        f"gate_repeat_count={config.normalized_gate_repeat_count()} "
        f"gate_thresholds={_classifier_gate_threshold_text()}"
    )
    if transfer_variant == "enhanced_jda":
        print(f"[Transfer] jda_config={_jda_config_text(config)}")

    evaluation_units = _build_evaluation_units(
        y=y,
        groups=groups,
        evaluation_protocol=config.normalized_protocol(),
        cv_splits=config.cv_splits,
        random_state=config.random_state,
        max_units=config.max_target_subjects,
    )

    source_only_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in CLASSIFIER_ORDER}
    target_only_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in CLASSIFIER_ORDER}
    raw_transfer_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in CLASSIFIER_ORDER}
    transfer_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in CLASSIFIER_ORDER}
    subject_rows: List[Dict[str, Any]] = []
    split_overview_rows: List[Dict[str, Any]] = []
    mmd_selection_rows: List[Dict[str, Any]] = []
    resume_pairs = _load_transfer_resume_pairs(config.summary_path)
    if resume_pairs:
        print(
            f"[Transfer][Resume] loaded_pairs={len(resume_pairs)} "
            f"checkpoint={config.summary_path}"
        )

    for unit_offset, evaluation_unit in enumerate(evaluation_units):
        target_subject_ids = [int(subject_id) for subject_id in evaluation_unit["target_subject_ids"]]
        target_unit_label = str(evaluation_unit["unit_label"])
        source_subject_ids = [subject_id for subject_id in all_subject_ids if subject_id not in set(target_subject_ids)]
        print("\n" + "=" * 88)
        print(
            f"[Transfer] target_unit={target_unit_label} target_subject_ids={target_subject_ids} "
            f"classifier_feature_counts={_classifier_feature_count_text(selected_feature_count_by_classifier)}"
        )

        for classifier_offset, classifier_name in enumerate(CLASSIFIER_ORDER):
            classifier_subject_blocks = subject_blocks_by_classifier[classifier_name]
            target_split = _split_target_subjects(
                target_subject_ids=target_subject_ids,
                subject_blocks=classifier_subject_blocks,
                config=config,
                random_state=config.random_state + unit_offset * 101,
            )
            if transfer_variant == "enhanced_jda":
                filtered_source_subject_ids = list(source_subject_ids)
                print(
                    f"[Transfer][SourceFilter] classifier={classifier_name} variant=enhanced_jda "
                    f"mode=paper52_mmd_only kept={len(filtered_source_subject_ids)}/{len(source_subject_ids)} "
                    f"note=论文 5.2 的 JDA 分支不做正类比例过滤"
                )
            else:
                filtered_source_subject_ids, _source_ratio_rows, _kept_source_ratio_rows = _filter_source_subject_ids_by_positive_ratio(
                    classifier_name=classifier_name,
                    source_subject_ids=source_subject_ids,
                    subject_blocks=classifier_subject_blocks,
                    target_train_y=target_split["y_train"],
                    config=config,
                )
            ranked_source_rows = _rank_source_subjects_by_mmd(
                source_subject_ids=filtered_source_subject_ids,
                subject_blocks=classifier_subject_blocks,
                target_train_raw=target_split["X_train"],
                sample_cap=config.mmd_sample_cap,
                random_state=config.random_state + unit_offset * 103 + classifier_offset * 11,
            )
            mmd_prefix_selection = _select_mmd_prefix_locally(
                ranked_source_rows=ranked_source_rows,
                subject_blocks=classifier_subject_blocks,
                target_train_raw=target_split["X_train"],
                target_train_y=target_split["y_train"],
                config=config,
                random_state=config.random_state + unit_offset * 107 + classifier_offset * 13,
            )
            best_mmd_prefix = dict(mmd_prefix_selection["best_mmd_prefix"])
            source_rank_preview = _format_source_rank_preview(ranked_source_rows=ranked_source_rows, limit=6)
            split_overview_rows.append(
                {
                    "Classifier": classifier_name,
                    "MaskFeatureCount": int(selected_feature_count_by_classifier[classifier_name]),
                    "TargetUnit": target_unit_label,
                    "TargetSubjects": ",".join(str(x) for x in target_subject_ids),
                    "Train": int(len(target_split["y_train"])),
                    "Val": int(len(target_split["y_val"])),
                    "Test": int(len(target_split["y_test"])),
                    "SourceFilterKept": f"{len(filtered_source_subject_ids)}/{len(source_subject_ids)}",
                    "SplitMode": str(target_split.get("split_mode", "")),
                    "PerSubjectSplit": str(target_split.get("per_subject_split_text", "")),
                    "TrainPosRatio": f"{float(np.mean(target_split['y_train'])):.4f}",
                    "ValPosRatio": f"{float(np.mean(target_split['y_val'])):.4f}",
                    "TestPosRatio": f"{float(np.mean(target_split['y_test'])):.4f}",
                    "SourceRankPreview": source_rank_preview,
                    "ChosenPrefix": int(best_mmd_prefix["prefix_size"]),
                    "ChosenMMD": f"{float(best_mmd_prefix['prefix_mmd']):.4f}",
                }
            )
            mmd_selection_rows.append(
                {
                    "Classifier": classifier_name,
                    "MaskFeatureCount": int(selected_feature_count_by_classifier[classifier_name]),
                    "TargetUnit": target_unit_label,
                    "BestMMDPrefix": int(best_mmd_prefix["prefix_size"]),
                    "BestMMD": f"{float(best_mmd_prefix['prefix_mmd']):.4f}",
                    "SourceFilterKept": f"{len(filtered_source_subject_ids)}/{len(source_subject_ids)}",
                    "BestMMDSources": ",".join(str(x) for x in best_mmd_prefix["chosen_source_ids"]),
                    "PrefixRankPreview": str(mmd_prefix_selection.get("prefix_rank_preview", "")),
                    "SourceRankPreview": source_rank_preview,
                    "WeightPreview": str(best_mmd_prefix.get("source_weight_preview", "")),
                }
            )
            print(
                f"[Transfer][UnitDetail] target_unit={target_unit_label} classifier={classifier_name} "
                f"mask_feature_count={selected_feature_count_by_classifier[classifier_name]} "
                f"source_filter_kept={len(filtered_source_subject_ids)}/{len(source_subject_ids)} "
                f"train={len(target_split['y_train'])} val={len(target_split['y_val'])} test={len(target_split['y_test'])} "
                f"source_rank_preview={source_rank_preview} best_prefix={best_mmd_prefix['prefix_size']} "
                f"best_mmd={float(best_mmd_prefix['prefix_mmd']):.4f}"
            )
            resume_key = _resume_pair_key(target_unit_label, classifier_name)
            if resume_key in resume_pairs:
                resume_payload = dict(resume_pairs[resume_key])
                source_only_rows[classifier_name].append(
                    {
                        str(metric_name): float(metric_value)
                        for metric_name, metric_value in dict(resume_payload.get("source_only_metrics", {})).items()
                    }
                )
                target_only_rows[classifier_name].append(
                    {
                        str(metric_name): float(metric_value)
                        for metric_name, metric_value in dict(resume_payload.get("target_only_metrics", {})).items()
                    }
                )
                raw_transfer_rows[classifier_name].append(
                    {
                        str(metric_name): float(metric_value)
                        for metric_name, metric_value in dict(resume_payload.get("raw_transfer_metrics", {})).items()
                    }
                )
                transfer_rows[classifier_name].append(
                    {
                        str(metric_name): float(metric_value)
                        for metric_name, metric_value in dict(resume_payload.get("final_transfer_metrics", {})).items()
                    }
                )
                subject_rows.append(dict(resume_payload.get("subject_row", {})))
                print(
                    f"[Transfer][Resume] target_unit={target_unit_label} "
                    f"classifier={classifier_name} source=checkpoint"
                )
                continue
            source_only_result = _run_source_only_baseline(
                classifier_name=classifier_name,
                source_subject_ids=source_subject_ids,
                subject_blocks=classifier_subject_blocks,
                target_split=target_split,
                config=config,
                random_state=config.random_state + unit_offset * 1001 + classifier_offset * 97,
            )
            target_only_result = _run_target_only_baseline(
                classifier_name=classifier_name,
                target_split=target_split,
                config=config,
                random_state=config.random_state + unit_offset * 1001 + classifier_offset * 97 + 7,
            )
            transfer_result = _run_selected_transfer_variant(
                classifier_name=classifier_name,
                mmd_prefix_selection=mmd_prefix_selection,
                subject_blocks=classifier_subject_blocks,
                target_split=target_split,
                config=config,
                random_state=config.random_state + unit_offset * 1001 + classifier_offset * 97 + 13,
                target_only_val_metrics=target_only_result["val_metrics"],
            )
            transfer_result["gate_repeat_summary"] = _run_repeat_gate_check(
                classifier_name=classifier_name,
                target_subject_ids=target_subject_ids,
                subject_blocks=classifier_subject_blocks,
                target_only_result=target_only_result,
                transfer_result=transfer_result,
                config=config,
                random_state=config.random_state + unit_offset * 1001 + classifier_offset * 97 + 29,
            )
            raw_transfer_result = copy.deepcopy(transfer_result)
            transfer_result = _select_transfer_or_target_only(
                classifier_name=classifier_name,
                target_only_result=target_only_result,
                transfer_result=transfer_result,
            )

            source_only_rows[classifier_name].append(source_only_result["test_metrics"])
            target_only_rows[classifier_name].append(target_only_result["test_metrics"])
            raw_transfer_rows[classifier_name].append(raw_transfer_result["test_metrics"])
            transfer_rows[classifier_name].append(transfer_result["test_metrics"])

            subject_rows.append(
                {
                    "TargetUnit": target_unit_label,
                    "Classifier": classifier_name,
                    "MaskFeatureCount": int(selected_feature_count_by_classifier[classifier_name]),
                    "SourceOnly_ACC": f"{source_only_result['test_metrics']['ACC']:.4f}",
                    "TargetOnly_ACC": f"{target_only_result['test_metrics']['ACC']:.4f}",
                    "RawTransfer_ACC": f"{raw_transfer_result['test_metrics']['ACC']:.4f}",
                    "RawTransfer_F1": f"{raw_transfer_result['test_metrics']['F1']:.4f}",
                    "Val_ACC": f"{transfer_result['val_metrics']['ACC']:.4f}",
                    "Val_F1": f"{transfer_result['val_metrics']['F1']:.4f}",
                    "Transfer_ACC": f"{transfer_result['test_metrics']['ACC']:.4f}",
                    "Transfer_F1": f"{transfer_result['test_metrics']['F1']:.4f}",
                    "ValTestGap_ACC": f"{transfer_result['test_metrics']['ACC'] - transfer_result['val_metrics']['ACC']:+.4f}",
                    "ValTestGap_F1": f"{transfer_result['test_metrics']['F1'] - transfer_result['val_metrics']['F1']:+.4f}",
                    "DeltaVsSource_ACC": f"{transfer_result['test_metrics']['ACC'] - source_only_result['test_metrics']['ACC']:+.4f}",
                    "DeltaVsTarget_ACC": f"{transfer_result['test_metrics']['ACC'] - target_only_result['test_metrics']['ACC']:+.4f}",
                    "RawDeltaVsTarget_ACC": f"{raw_transfer_result['test_metrics']['ACC'] - target_only_result['test_metrics']['ACC']:+.4f}",
                    "RawDeltaVsTarget_F1": f"{raw_transfer_result['test_metrics']['F1'] - target_only_result['test_metrics']['F1']:+.4f}",
                    "PrefixSize": int(transfer_result["prefix_size"]),
                    "PrefixRank": int(transfer_result.get("prefix_rank", 0)),
                    "PrefixMMD": f"{float(transfer_result['prefix_mmd']):.4f}",
                    "TargetRepeat": int(transfer_result["target_repeat"]),
                    "PseudoKeepRatio": f"{float(transfer_result.get('pseudo_keep_ratio', 0.0)):.2f}",
                    "PseudoSelected": int(transfer_result.get("pseudo_selected_count", 0)),
                    "PseudoConfMean": f"{float(transfer_result.get('pseudo_selected_conf_mean', 0.0)):.4f}",
                    "DesiredRatio": f"{float(transfer_result.get('desired_source_target_ratio', 0.0)):.2f}",
                    "ActualRatio": f"{float(transfer_result.get('final_balance_info', {}).get('effective_source_target_ratio', 0.0)):.2f}",
                    "Threshold": f"{float(transfer_result['threshold']):.3f}",
                    "TopSources": ",".join(str(x) for x in transfer_result["chosen_source_ids"]),
                    "SourceWeightMode": str(transfer_result["source_weight_info"]["mode"]),
                    "TransferVariant": str(transfer_result.get("transfer_variant", transfer_variant)),
                    "AlignmentMethod": str(transfer_result.get("alignment_method", "none")),
                    "JDAIterations": int(float(transfer_result.get("final_jda_diag", {}).get("iterations", 0.0))),
                    "JDALastMMD": f"{float(transfer_result.get('final_jda_diag', {}).get('end_mmd', 0.0)):.4f}",
                    "JDAMMDDelta": f"{float(transfer_result.get('final_jda_diag', {}).get('delta_mmd', 0.0)):+.4f}",
                    "JDAPseudoChange": f"{float(transfer_result.get('final_jda_diag', {}).get('last_pseudo_change_ratio', 0.0)):.4f}",
                    "JDAConfidence": f"{float(transfer_result.get('final_jda_diag', {}).get('last_confidence_mean', 0.0)):.4f}",
                    "SelectionMode": str(transfer_result.get("selection_mode", "transfer")),
                    "GateReason": str(transfer_result.get("gate_reason", "")),
                    "GateRepeatPassed": "Y" if bool(transfer_result.get("gate_repeat_summary", {}).get("repeat_gate_passed", False)) else "",
                    "GateRepeatMeanGain_ACC": f"{float(transfer_result.get('gate_repeat_summary', {}).get('mean_gain_acc', 0.0)):+.4f}",
                    "GateRepeatMeanGain_F1": f"{float(transfer_result.get('gate_repeat_summary', {}).get('mean_gain_f1', 0.0)):+.4f}",
                    "Params": _format_params(transfer_result["params"]),
                }
            )
            resume_pairs[resume_key] = {
                "source_only_metrics": dict(source_only_result["test_metrics"]),
                "target_only_metrics": dict(target_only_result["test_metrics"]),
                "raw_transfer_metrics": dict(raw_transfer_result["test_metrics"]),
                "final_transfer_metrics": dict(transfer_result["test_metrics"]),
                "subject_row": dict(subject_rows[-1]),
            }
            _save_transfer_resume_pairs(config.summary_path, config=config, resume_pairs=resume_pairs)

    source_only_summary = _summarize_method_rows(source_only_rows)
    target_only_summary = _summarize_method_rows(target_only_rows)
    raw_transfer_summary = _summarize_method_rows(raw_transfer_rows)
    final_transfer_summary = _summarize_method_rows(transfer_rows)
    transfer_summary = final_transfer_summary
    method_comparison_rows = _method_comparison_rows(
        source_only_summary=source_only_summary,
        target_only_summary=target_only_summary,
        raw_transfer_summary=raw_transfer_summary,
        final_transfer_summary=final_transfer_summary,
    )
    gate_summary_rows = _gate_summary_rows(subject_rows)
    transfer_decision_rows = _transfer_decision_rows(subject_rows)
    val_test_gap_rows = _val_test_gap_rows(subject_rows)
    prefix_choice_rows = _prefix_choice_rows(subject_rows)
    source_frequency_rows = _source_frequency_rows(subject_rows)
    gain_vs_source_rows = _gain_rows(
        reference_summary=source_only_summary,
        current_summary=transfer_summary,
        reference_label="SourceOnly",
        current_label="Transfer",
    )
    gain_vs_target_rows = _gain_rows(
        reference_summary=target_only_summary,
        current_summary=transfer_summary,
        reference_label="TargetOnly",
        current_label="Transfer",
    )
    hard_rows = sorted(subject_rows, key=lambda row: (float(row["Transfer_ACC"]), float(row["Transfer_F1"])))[: min(10, len(subject_rows))]

    legacy_reference_path, legacy_reference_json = _load_reference_summary(config)
    legacy_reference_rows = _legacy_reference_rows(legacy_reference_json)
    reference_delta_rows = _reference_delta_rows(legacy_reference_json, transfer_summary)

    report_path = _write_classifier_mask_transfer_report(
        config=config,
        selected_feature_count_by_classifier=selected_feature_count_by_classifier,
        stage1_lines=stage1_lines,
        stage1_classifier_rows=stage1_classifier_rows,
        split_overview_rows=split_overview_rows,
        mmd_selection_rows=mmd_selection_rows,
        gate_summary_rows=gate_summary_rows,
        transfer_decision_rows=transfer_decision_rows,
        val_test_gap_rows=val_test_gap_rows,
        prefix_choice_rows=prefix_choice_rows,
        source_frequency_rows=source_frequency_rows,
        source_only_summary=source_only_summary,
        target_only_summary=target_only_summary,
        raw_transfer_summary=raw_transfer_summary,
        final_transfer_summary=final_transfer_summary,
        method_comparison_rows=method_comparison_rows,
        gain_vs_source_rows=gain_vs_source_rows,
        gain_vs_target_rows=gain_vs_target_rows,
        subject_rows=subject_rows,
        hard_rows=hard_rows,
        legacy_reference_path=legacy_reference_path,
        legacy_reference_rows=legacy_reference_rows,
        reference_delta_rows=reference_delta_rows,
    )

    summary = {
        "config": asdict(config),
        "feature_source_mode": feature_source_mode,
        "feature_source_label": feature_source_label,
        "transfer_variant": transfer_variant,
        "transfer_variant_label": _transfer_variant_label(transfer_variant),
        "selected_feature_count": int(max(selected_feature_count_by_classifier.values())),
        "selected_feature_count_by_classifier": {
            classifier_name: int(selected_feature_count_by_classifier[classifier_name])
            for classifier_name in CLASSIFIER_ORDER
        },
        "stage1_best_classifier_name": stage1_best_name,
        "stage1_best_classifier_metrics": stage1_best_metrics,
        "stage1_classifier_rows": stage1_classifier_rows,
        "source_selection_note": (
            "论文 5.2 的 JDA 分支不做正类比例过滤：先计算每个子源域与目标训练集的 MMD，升序排序后构造累计前缀，并直接选择累计前缀 MMD 最小的源域组合作为最终迁移源域。"
            if transfer_variant == "enhanced_jda"
            else "每个分类器先使用自己的阶段一 mask，再先按目标域正类比例过滤源被试，对前1、前2、前3等累计前缀计算 MMD 排序，并在验证集上复排选择最优前缀。"
        ),
        "source_weighting_note": (
            "论文 5.2 的 JDA 分支先用目标训练块做无标签分布对齐，再从对齐后的目标训练块里筛选高置信伪标签样本参与自训练；最终分类器使用“源域真标签 + 高置信目标伪标签”联合训练。"
            if transfer_variant == "enhanced_jda"
            else "训练时显式约束源域/目标域有效强度比例，并按目标单元动态融合 inverse-MMD 与正类比例接近度；LR/DT 使用样本权重实现，KNN 通过加权重采样近似实现。"
        ),
        "adaptation_note": (
            "当前运行分支为论文 5.2 风格的多源域选择 + JDA 联合分布自适应：先按累计前缀 MMD 直接选局部最优源域组合，再通过 JDA 同时适配边缘分布和条件分布，最后把高置信目标伪标签样本加入自训练。"
            if transfer_variant == "enhanced_jda"
            else "当前运行分支为监督迁移：直接在共享预处理空间中混合源域与目标训练集做监督迁移与 gate。"
        ),
        "jda_config_text": _jda_config_text(config) if transfer_variant == "enhanced_jda" else "",
        "gate_repeat_note": "主验证切分没有通过 gate 时，会对固定迁移候选做重复切分复核；若重复复核的平均 ACC/F1 增益都不差于 target-only，则允许保留迁移。",
        "split_overview_rows": split_overview_rows,
        "mmd_selection_rows": mmd_selection_rows,
        "gate_summary_rows": gate_summary_rows,
        "transfer_decision_rows": transfer_decision_rows,
        "val_test_gap_rows": val_test_gap_rows,
        "prefix_choice_rows": prefix_choice_rows,
        "source_frequency_rows": source_frequency_rows,
        "method_comparison_rows": method_comparison_rows,
        "source_only_summary": source_only_summary,
        "target_only_summary": target_only_summary,
        "raw_transfer_summary": raw_transfer_summary,
        "final_transfer_summary": final_transfer_summary,
        "transfer_summary": transfer_summary,
        "gain_vs_source_rows": gain_vs_source_rows,
        "gain_vs_target_rows": gain_vs_target_rows,
        "subject_rows": subject_rows,
        "hard_rows": hard_rows,
        "legacy_reference_path": None if legacy_reference_path is None else str(legacy_reference_path),
        "legacy_reference_rows": legacy_reference_rows,
        "reference_delta_rows": reference_delta_rows,
        "report_path": str(report_path),
    }

    save_json(to_serializable(summary), config.summary_path)
    print(f"[Transfer] summary_path={config.summary_path}")
    print(f"[Transfer] report_path={report_path}")
    for classifier_name in CLASSIFIER_ORDER:
        print(
            f"[Transfer][Summary] classifier={classifier_name} "
            f"source_only=({_format_summary_metric_snapshot(source_only_summary, classifier_name)}) "
            f"target_only=({_format_summary_metric_snapshot(target_only_summary, classifier_name)}) "
            f"raw_transfer=({_format_summary_metric_snapshot(raw_transfer_summary, classifier_name)}) "
            f"final_transfer=({_format_summary_metric_snapshot(final_transfer_summary, classifier_name)})"
        )
    for row in gate_summary_rows:
        print(
            f"[Transfer][GateSummary] classifier={row['Classifier']} units={row['Units']} "
            f"kept={row['TransferKept']} gated={row['TargetOnlyGate']} gate_rate={row['GateRate']} "
            f"top_reason={row['TopGateReason']}"
        )
    for row in transfer_decision_rows:
        print(
            f"[Transfer][Decision] classifier={row['Classifier']} "
            f"transfer_count={row['TransferCount']} fallback_count={row['FallbackCount']} "
            f"transfer_units={row['TransferUnits']} fallback_units={row['FallbackUnits']} "
            f"fallback_reasons={row['FallbackReasons']}"
        )
    for row in val_test_gap_rows:
        print(
            f"[Transfer][ValTestGap] classifier={row['Classifier']} "
            f"val_ACC={row['Val_ACC']} test_ACC={row['Test_ACC']} gap_ACC={row['Gap_ACC']} "
            f"val_F1={row['Val_F1']} test_F1={row['Test_F1']} gap_F1={row['Gap_F1']}"
        )
    for row in prefix_choice_rows:
        print(
            f"[Transfer][PrefixChoice] classifier={row['Classifier']} "
            f"top_rank={row['TopPrefixRank']} top_size={row['TopPrefixSize']} count={row['TopChoiceCount']} "
            f"mean_desired_ratio={row['MeanDesiredRatio']} mean_actual_ratio={row['MeanActualRatio']}"
        )
    for row in source_frequency_rows[: min(18, len(source_frequency_rows))]:
        print(
            f"[Transfer][SourceFreq] classifier={row['Classifier']} "
            f"source_id={row['SourceId']} chosen_count={row['ChosenCount']}"
        )
    return summary

