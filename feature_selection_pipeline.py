from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from AI.common import (
    DEFAULT_RANDOM_STATE,
    binary_metrics,
    ensure_ai_output_path,
    ensure_artifact_dir,
    load_deap_features,
    make_outer_folds,
    positive_class_scores,
    save_json,
    save_pickle,
    stratified_group_subsample_indices,
    to_serializable,
)
from AI.feature_selection_backends import (
    aggregate_raw_feature_votes,
    normalize_selector_name,
    run_feature_selector,
    selector_display_name,
)
from AI.reporting import format_metric_bars, format_metric_table, write_markdown_report

CLASSIFIER_ORDER: Sequence[str] = ("LR", "KNN", "DT")


@dataclass
class FeatureSelectionConfig:
    """管理特征选择阶段的核心配置。"""

    evaluation_protocol: str = "loso_subject"
    selector_name: str = "miifs"
    n_splits: int = 4
    max_folds: int | None = 2
    random_state: int = DEFAULT_RANDOM_STATE
    variance_threshold: float = 1e-3
    selector_max_features: int = 96
    selector_bins: int = 32
    evaluation_feature_counts: List[int] = field(default_factory=lambda: [48, 64, 80, 96])
    mask_feature_count: int | None = None
    min_accept_acc: float = 0.60
    train_sample_cap_per_subject: int | None = 160
    artifact_path: Path = Path("AI/artifacts/feature_selection_cv_latest.pkl")
    summary_path: Path = Path("AI/artifacts/feature_selection_summary_latest.json")
    mask_path: Path = Path("AI/artifacts/miifs_mask_latest.pkl")
    report_path: Path = Path("AI/artifacts/feature_selection_report_latest.md")

    def normalized_feature_counts(self) -> List[int]:
        """返回合法且去重后的候选特征数列表。"""
        counts = sorted(
            {
                int(count)
                for count in self.evaluation_feature_counts
                if 0 < int(count) <= int(self.selector_max_features)
            }
        )
        if not counts:
            raise ValueError("evaluation_feature_counts 至少要包含一个正整数。")
        return counts

    def normalized_selector_name(self) -> str:
        """规范化特征选择器名称。"""
        return normalize_selector_name(self.selector_name)

    def normalized_evaluation_protocol(self) -> str:
        """规范化评估协议名称。"""
        protocol = str(self.evaluation_protocol).lower().strip()
        if protocol not in {"loso_subject", "group_kfold", "sample_kfold"}:
            raise ValueError(f"不支持的 evaluation_protocol: {self.evaluation_protocol!r}")
        return protocol


class LegacyFoldPreprocessor:
    """封装单折内的方差过滤、对数变换与鲁棒标准化。"""

    def __init__(self, variance_threshold: float = 1e-3) -> None:
        """记录预处理超参数并初始化状态。"""
        self.variance_threshold_value = float(variance_threshold)
        self.variance_threshold: VarianceThreshold | None = None
        self.positive_mask: np.ndarray | None = None
        self.scaler: RobustScaler | None = None
        self._raw_to_processed: Dict[int, int] = {}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在训练折上拟合预处理器并返回处理后的特征。"""
        X = np.asarray(X, dtype=np.float32).copy()
        self.variance_threshold = VarianceThreshold(threshold=self.variance_threshold_value)
        X_vt = self.variance_threshold.fit_transform(X)
        self.positive_mask = np.all(X_vt > 0, axis=0)
        if np.any(self.positive_mask):
            X_vt[:, self.positive_mask] = np.log1p(X_vt[:, self.positive_mask])
        self.scaler = RobustScaler()
        X_scaled = np.asarray(self.scaler.fit_transform(X_vt), dtype=np.float32)

        raw_indices = self.original_feature_indices(np.arange(X_scaled.shape[1], dtype=np.int32))
        self._raw_to_processed = {int(raw_idx): int(pos) for pos, raw_idx in enumerate(raw_indices)}
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """把训练折学到的预处理规则应用到验证折。"""
        if self.variance_threshold is None or self.positive_mask is None or self.scaler is None:
            raise ValueError("预处理器尚未拟合，不能直接调用 transform。")

        X = np.asarray(X, dtype=np.float32).copy()
        X_vt = self.variance_threshold.transform(X)
        if np.any(self.positive_mask):
            X_vt[:, self.positive_mask] = np.log1p(X_vt[:, self.positive_mask])
        return np.asarray(self.scaler.transform(X_vt), dtype=np.float32)

    def original_feature_indices(self, processed_indices: Sequence[int]) -> np.ndarray:
        """把处理后特征索引映射回原始特征索引。"""
        if self.variance_threshold is None:
            raise ValueError("预处理器尚未拟合，不能映射原始特征索引。")
        kept = self.variance_threshold.get_support(indices=True)
        processed_indices = np.asarray(processed_indices, dtype=np.int32)
        return kept[processed_indices]

    def processed_indices_from_raw(self, raw_indices: Sequence[int]) -> np.ndarray:
        """把原始特征索引映射到当前折的处理后特征空间。"""
        if not self._raw_to_processed:
            raise ValueError("预处理器尚未拟合，不能映射处理后特征索引。")
        mapped = [self._raw_to_processed[int(raw_idx)] for raw_idx in raw_indices if int(raw_idx) in self._raw_to_processed]
        return np.asarray(mapped, dtype=np.int32)

    def describe(self) -> Dict[str, int]:
        """输出当前折预处理器的关键统计信息。"""
        if self.variance_threshold is None or self.positive_mask is None:
            raise ValueError("预处理器尚未拟合，不能输出描述信息。")
        return {
            "input_feature_count": int(self.variance_threshold.n_features_in_),
            "post_variance_feature_count": int(np.sum(self.variance_threshold.get_support())),
            "log1p_feature_count": int(np.sum(self.positive_mask)),
        }


def _build_classifiers() -> Dict[str, Any]:
    """构建特征选择阶段固定使用的分类器。"""
    return {
        "LR": LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1),
        "DT": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_STATE,
        ),
    }


def _evaluate_classifier_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """在同一组特征上训练分类器并返回诊断指标。"""
    results: Dict[str, Dict[str, float]] = {}
    for classifier_name, classifier in _build_classifiers().items():
        model = clone(classifier)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        y_prob = positive_class_scores(model, X_eval)
        results[classifier_name] = binary_metrics(y_eval, y_pred, y_prob)
    return results


def _metric_means(metric_rows: Sequence[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """对多折结果求均值，得到分类器级汇总指标。"""
    summary: Dict[str, Dict[str, float]] = {}
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row[classifier_name] for row in metric_rows]
        summary[classifier_name] = {}
        for metric_name in ["ACC", "PRE", "REC", "F1", "AUC", "BACC", "MCC", "SPE"]:
            values = np.asarray([metrics[metric_name] for metrics in classifier_rows], dtype=np.float64)
            summary[classifier_name][metric_name] = float(np.mean(values))
    return summary


def _metric_snapshot(metrics: Dict[str, float]) -> str:
    """把常见分类指标压缩成一行文本，便于高密度调试打印。"""
    return (
        f"ACC={float(metrics['ACC']):.4f} F1={float(metrics['F1']):.4f} "
        f"AUC={float(metrics['AUC']):.4f} BACC={float(metrics['BACC']):.4f} "
        f"MCC={float(metrics['MCC']):.4f} PRE={float(metrics['PRE']):.4f} "
        f"REC={float(metrics['REC']):.4f} SPE={float(metrics['SPE']):.4f}"
    )


def _classifier_rank_key(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
    """用 ACC 和 F1 优先排序分类器结果。"""
    return (
        float(metrics["ACC"]),
        float(metrics["F1"]),
        float(metrics["BACC"]),
        float(metrics["MCC"]),
    )


def _feature_count_rank_key(summary_block: Dict[str, Dict[str, float]], feature_count: int) -> tuple[float, float, float, float, int]:
    """用三类分类器的平均 ACC/F1 为特征数排序。"""
    classifier_names = list(CLASSIFIER_ORDER)
    mean_acc = float(np.mean([summary_block[name]["ACC"] for name in classifier_names]))
    mean_f1 = float(np.mean([summary_block[name]["F1"] for name in classifier_names]))
    best_acc = float(np.max([summary_block[name]["ACC"] for name in classifier_names]))
    best_f1 = float(np.max([summary_block[name]["F1"] for name in classifier_names]))
    return mean_acc, mean_f1, best_acc, best_f1, -int(feature_count)


def _best_classifier_from_summary(summary_block: Dict[str, Dict[str, float]]) -> tuple[str, Dict[str, float]]:
    """返回当前汇总结果里表现最好的分类器。"""
    best_name = max(CLASSIFIER_ORDER, key=lambda name: _classifier_rank_key(summary_block[name]))
    return best_name, summary_block[best_name]


def _classifier_feature_count_rank_key(
    classifier_name: str,
    metrics: Dict[str, float],
    feature_count: int,
) -> tuple[float, float, float, float, float, int]:
    """按单个分类器的 ACC/F1 表现给候选特征数排序。"""
    return (
        min(float(metrics["ACC"]), float(metrics["F1"])),
        float(metrics["ACC"]),
        float(metrics["F1"]),
        float(metrics["BACC"]),
        float(metrics["MCC"]),
        -int(feature_count),
    )


def _select_mask_feature_count(
    config: FeatureSelectionConfig,
    per_count_summary: Dict[int, Dict[str, Dict[str, float]]],
) -> int:
    """确定最终高频掩码使用的特征数。"""
    if config.mask_feature_count is not None:
        return int(config.mask_feature_count)
    return max(sorted(per_count_summary), key=lambda count: _feature_count_rank_key(per_count_summary[count], count))


def _select_classifier_mask_feature_counts(
    config: FeatureSelectionConfig,
    per_count_summary: Dict[int, Dict[str, Dict[str, float]]],
) -> Dict[str, int]:
    """为每个分类器单独选择最适合自己的冻结 mask 特征数。"""
    if config.mask_feature_count is not None:
        forced_count = int(config.mask_feature_count)
        return {classifier_name: forced_count for classifier_name in CLASSIFIER_ORDER}

    chosen_counts: Dict[str, int] = {}
    for classifier_name in CLASSIFIER_ORDER:
        chosen_counts[classifier_name] = max(
            sorted(per_count_summary),
            key=lambda count: _classifier_feature_count_rank_key(
                classifier_name=classifier_name,
                metrics=per_count_summary[count][classifier_name],
                feature_count=count,
            ),
        )
    return chosen_counts


def _feature_count_rows(per_count_summary: Dict[int, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    """把候选特征数结果整理成 Markdown 表格行。"""
    rows: List[Dict[str, Any]] = []
    for feature_count in sorted(per_count_summary):
        summary_block = per_count_summary[int(feature_count)]
        rows.append(
            {
                "FeatureCount": int(feature_count),
                "LR_ACC": f"{summary_block['LR']['ACC']:.4f}",
                "KNN_ACC": f"{summary_block['KNN']['ACC']:.4f}",
                "DT_ACC": f"{summary_block['DT']['ACC']:.4f}",
                "LR_F1": f"{summary_block['LR']['F1']:.4f}",
                "KNN_F1": f"{summary_block['KNN']['F1']:.4f}",
                "DT_F1": f"{summary_block['DT']['F1']:.4f}",
            }
        )
    return rows


def _summary_delta_rows(
    reference_summary: Dict[str, Dict[str, float]],
    current_summary: Dict[str, Dict[str, float]],
    reference_label: str,
    current_label: str,
) -> List[Dict[str, Any]]:
    """把两组分类器结果整理成增量表。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        reference_metrics = reference_summary[classifier_name]
        current_metrics = current_summary[classifier_name]
        rows.append(
            {
                "Classifier": classifier_name,
                f"{reference_label}_ACC": f"{reference_metrics['ACC']:.4f}",
                f"{current_label}_ACC": f"{current_metrics['ACC']:.4f}",
                "Delta_ACC": f"{current_metrics['ACC'] - reference_metrics['ACC']:+.4f}",
                f"{reference_label}_F1": f"{reference_metrics['F1']:.4f}",
                f"{current_label}_F1": f"{current_metrics['F1']:.4f}",
                "Delta_F1": f"{current_metrics['F1'] - reference_metrics['F1']:+.4f}",
            }
        )
    return rows


def _subsample_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_cap_per_subject: int | None,
    random_state: int,
) -> np.ndarray:
    """按被试做轻量采样，降低调试阶段的 MIIFS 计算量。"""
    if sample_cap_per_subject is None or sample_cap_per_subject <= 0:
        return train_idx
    local_keep = stratified_group_subsample_indices(
        y=y[train_idx],
        groups=groups[train_idx],
        max_per_group=int(sample_cap_per_subject),
        random_state=int(random_state),
    )
    return np.asarray(train_idx[local_keep], dtype=np.int32)


def _build_loso_subject_splits(
    config: FeatureSelectionConfig,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, Any]]:
    """按被试严格留一生成特征选择折信息。"""
    split_rows: List[Dict[str, Any]] = []
    unique_subject_ids = np.unique(groups).astype(np.int32)

    for fold_id, subject_id in enumerate(unique_subject_ids.tolist(), start=1):
        val_idx = np.flatnonzero(groups == subject_id).astype(np.int32)
        train_idx = np.flatnonzero(groups != subject_id).astype(np.int32)
        split_rows.append(
            {
                "fold_id": int(fold_id),
                "train_idx": _subsample_train_indices(
                    train_idx=train_idx,
                    y=y,
                    groups=groups,
                    sample_cap_per_subject=config.train_sample_cap_per_subject,
                    random_state=config.random_state + fold_id * 31,
                ),
                "val_idx": val_idx,
                "train_subject_ids": np.unique(groups[train_idx]).astype(np.int32).tolist(),
                "val_subject_ids": [int(subject_id)],
                "raw_train_size": int(len(train_idx)),
            }
        )
    return split_rows


def _build_feature_selection_splits(
    config: FeatureSelectionConfig,
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, Any]]:
    """根据配置构造特征选择阶段的训练/验证切分。"""
    protocol = config.normalized_evaluation_protocol()
    split_rows: List[Dict[str, Any]] = []

    if protocol == "loso_subject":
        return _build_loso_subject_splits(config=config, y=y, groups=groups)

    if protocol == "group_kfold":
        n_splits = min(int(config.n_splits), int(len(np.unique(groups))))
        for fold_id, train_idx, val_idx in make_outer_folds(
            y=y,
            groups=groups,
            n_splits=n_splits,
            random_state=config.random_state,
        ):
            split_rows.append(
                {
                    "fold_id": int(fold_id),
                    "train_idx": _subsample_train_indices(
                        train_idx=train_idx,
                        y=y,
                        groups=groups,
                        sample_cap_per_subject=config.train_sample_cap_per_subject,
                        random_state=config.random_state + fold_id * 31,
                    ),
                    "val_idx": np.asarray(val_idx, dtype=np.int32),
                    "train_subject_ids": np.unique(groups[train_idx]).astype(np.int32).tolist(),
                    "val_subject_ids": np.unique(groups[val_idx]).astype(np.int32).tolist(),
                    "raw_train_size": int(len(train_idx)),
                }
            )
        return split_rows

    class_counts = np.bincount(np.asarray(y, dtype=np.int32))
    nonzero_counts = class_counts[class_counts > 0]
    min_class_count = int(np.min(nonzero_counts)) if nonzero_counts.size else 2
    n_splits = max(2, min(int(config.n_splits), max(min_class_count, 2)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y), start=1):
        train_idx = np.asarray(train_idx, dtype=np.int32)
        val_idx = np.asarray(val_idx, dtype=np.int32)
        split_rows.append(
            {
                "fold_id": int(fold_id),
                "train_idx": _subsample_train_indices(
                    train_idx=train_idx,
                    y=y,
                    groups=groups,
                    sample_cap_per_subject=config.train_sample_cap_per_subject,
                    random_state=config.random_state + fold_id * 31,
                ),
                "val_idx": val_idx,
                "train_subject_ids": np.unique(groups[train_idx]).astype(np.int32).tolist(),
                "val_subject_ids": np.unique(groups[val_idx]).astype(np.int32).tolist(),
                "raw_train_size": int(len(train_idx)),
            }
        )
    return split_rows


def _evaluate_frozen_mask(
    X_raw: np.ndarray,
    y: np.ndarray,
    fold_rows: Sequence[Dict[str, Any]],
    selected_raw_indices: np.ndarray,
    variance_threshold: float,
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """在同一批折上回放最终冻结掩码，得到真正可复用的基线结果。"""
    metric_rows: List[Dict[str, Dict[str, float]]] = []
    debug_rows: List[Dict[str, Any]] = []
    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        train_idx = np.asarray(fold_row["train_idx"], dtype=np.int32)
        val_idx = np.asarray(fold_row["val_idx"], dtype=np.int32)

        preprocessor = LegacyFoldPreprocessor(variance_threshold=variance_threshold)
        X_train = preprocessor.fit_transform(X_raw[train_idx])
        X_val = preprocessor.transform(X_raw[val_idx])
        local_indices = preprocessor.processed_indices_from_raw(selected_raw_indices)
        if local_indices.size == 0:
            raise ValueError(f"Fold {fold_id} 在冻结掩码回放时没有可用特征。")

        metrics = _evaluate_classifier_set(
            X_train=X_train[:, local_indices],
            y_train=y[train_idx],
            X_eval=X_val[:, local_indices],
            y_eval=y[val_idx],
        )
        metric_rows.append(metrics)
        debug_rows.append(
            {
                "Fold": fold_id,
                "UsableFeatureCount": int(local_indices.size),
                "LR_ACC": f"{metrics['LR']['ACC']:.4f}",
                "LR_F1": f"{metrics['LR']['F1']:.4f}",
                "KNN_ACC": f"{metrics['KNN']['ACC']:.4f}",
                "KNN_F1": f"{metrics['KNN']['F1']:.4f}",
                "DT_ACC": f"{metrics['DT']['ACC']:.4f}",
                "DT_F1": f"{metrics['DT']['F1']:.4f}",
            }
        )
        print(
            f"[FeatureSelection][Replay][Fold {fold_id}] usable_features={local_indices.size} "
            f"LR_ACC={metrics['LR']['ACC']:.4f} LR_F1={metrics['LR']['F1']:.4f} "
            f"KNN_ACC={metrics['KNN']['ACC']:.4f} KNN_F1={metrics['KNN']['F1']:.4f} "
            f"DT_ACC={metrics['DT']['ACC']:.4f} DT_F1={metrics['DT']['F1']:.4f}"
        )
    return _metric_means(metric_rows), debug_rows


def _mean_single_classifier_metrics(metric_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """对同一分类器在多折上的指标求均值。"""
    summary: Dict[str, float] = {}
    for metric_name in ["ACC", "PRE", "REC", "F1", "AUC", "BACC", "MCC", "SPE"]:
        values = np.asarray([metrics[metric_name] for metrics in metric_rows], dtype=np.float64)
        summary[metric_name] = float(np.mean(values))
    return summary


def _evaluate_classifier_specific_masks(
    X_raw: np.ndarray,
    y: np.ndarray,
    fold_rows: Sequence[Dict[str, Any]],
    selected_raw_indices_by_classifier: Dict[str, np.ndarray],
    variance_threshold: float,
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """让每个分类器都只用自己的冻结 mask 回放，得到真正对应迁移阶段的基线。"""
    metric_rows_by_classifier: Dict[str, List[Dict[str, float]]] = {name: [] for name in CLASSIFIER_ORDER}
    debug_rows: List[Dict[str, Any]] = []
    classifiers = _build_classifiers()

    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        train_idx = np.asarray(fold_row["train_idx"], dtype=np.int32)
        val_idx = np.asarray(fold_row["val_idx"], dtype=np.int32)

        preprocessor = LegacyFoldPreprocessor(variance_threshold=variance_threshold)
        X_train = preprocessor.fit_transform(X_raw[train_idx])
        X_val = preprocessor.transform(X_raw[val_idx])
        debug_row: Dict[str, Any] = {"Fold": fold_id}
        log_parts: List[str] = []

        for classifier_name in CLASSIFIER_ORDER:
            selected_raw_indices = np.asarray(selected_raw_indices_by_classifier[classifier_name], dtype=np.int32)
            local_indices = preprocessor.processed_indices_from_raw(selected_raw_indices)
            if local_indices.size == 0:
                raise ValueError(f"Fold {fold_id} 的 {classifier_name} 专属 mask 在回放时没有可用特征。")

            model = clone(classifiers[classifier_name])
            model.fit(X_train[:, local_indices], y[train_idx])
            y_pred = model.predict(X_val[:, local_indices])
            y_prob = positive_class_scores(model, X_val[:, local_indices])
            metrics = binary_metrics(y[val_idx], y_pred, y_prob)
            metric_rows_by_classifier[classifier_name].append(metrics)

            debug_row[f"{classifier_name}_UsableFeatureCount"] = int(local_indices.size)
            debug_row[f"{classifier_name}_ACC"] = f"{metrics['ACC']:.4f}"
            debug_row[f"{classifier_name}_F1"] = f"{metrics['F1']:.4f}"
            log_parts.append(
                f"{classifier_name}[count={local_indices.size}] ACC={metrics['ACC']:.4f} F1={metrics['F1']:.4f}"
            )

        debug_rows.append(debug_row)
        print(f"[FeatureSelection][Replay][Fold {fold_id}] " + " ".join(log_parts))

    summary = {
        classifier_name: _mean_single_classifier_metrics(metric_rows_by_classifier[classifier_name])
        for classifier_name in CLASSIFIER_ORDER
    }
    return summary, debug_rows


def _classifier_mask_rows(
    selected_feature_count_by_classifier: Dict[str, int],
    selected_summary_by_classifier: Dict[str, Dict[str, float]],
    frozen_mask_summary: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """整理每个分类器的专属 mask 大小，以及筛选期/回放期指标。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        selected_metrics = selected_summary_by_classifier[classifier_name]
        replay_metrics = frozen_mask_summary[classifier_name]
        rows.append(
            {
                "Classifier": classifier_name,
                "FeatureCount": int(selected_feature_count_by_classifier[classifier_name]),
                "CV_ACC": f"{selected_metrics['ACC']:.4f}",
                "CV_F1": f"{selected_metrics['F1']:.4f}",
                "Replay_ACC": f"{replay_metrics['ACC']:.4f}",
                "Replay_F1": f"{replay_metrics['F1']:.4f}",
            }
        )
    return rows


def _classifier_top_feature_rows(
    frequency_counts_by_count: Dict[int, np.ndarray],
    selected_feature_count_by_classifier: Dict[str, int],
    selected_raw_indices_by_classifier: Dict[str, np.ndarray],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """整理每个分类器专属 mask 里最常出现的原始特征。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        feature_count = int(selected_feature_count_by_classifier[classifier_name])
        frequency_counts = frequency_counts_by_count[feature_count]
        selected_raw_indices = np.asarray(selected_raw_indices_by_classifier[classifier_name], dtype=np.int32)
        for raw_idx in selected_raw_indices[: min(top_n, len(selected_raw_indices))]:
            rows.append(
                {
                    "Classifier": classifier_name,
                    "FeatureCount": feature_count,
                    "RawIndex": int(raw_idx),
                    "VoteCount": int(frequency_counts[int(raw_idx)]),
                }
            )
    return rows


def _classifier_mask_count_text(selected_feature_count_by_classifier: Dict[str, int]) -> str:
    """把分类器专属 mask 特征数压缩成便于打印的文本。"""
    return ", ".join(
        f"{classifier_name}={int(selected_feature_count_by_classifier[classifier_name])}"
        for classifier_name in CLASSIFIER_ORDER
    )


def _top_feature_rows(frequency_counts: np.ndarray, selected_raw_indices: np.ndarray, top_n: int = 20) -> List[Dict[str, Any]]:
    """整理最终高频原始特征列表。"""
    rows: List[Dict[str, Any]] = []
    for raw_idx in selected_raw_indices[:top_n]:
        rows.append(
            {
                "RawIndex": int(raw_idx),
                "VoteCount": int(frequency_counts[int(raw_idx)]),
            }
        )
    return rows


def _subject_overlap_summary(
    train_subject_ids: Sequence[int],
    val_subject_ids: Sequence[int],
) -> tuple[List[int], str]:
    """统计单折训练/验证被试是否重叠，用于定位被试级数据泄露风险。"""
    train_subject_set = {int(subject_id) for subject_id in train_subject_ids}
    val_subject_set = {int(subject_id) for subject_id in val_subject_ids}
    overlap_subject_ids = sorted(train_subject_set & val_subject_set)
    overlap_text = ",".join(str(subject_id) for subject_id in overlap_subject_ids) if overlap_subject_ids else "-"
    return overlap_subject_ids, overlap_text


def _artifact_file_name(path: Path | str) -> str:
    """提取产物路径里的文件名，便于在报告中快速定位对应文件。"""
    return Path(path).name


def _feature_report_artifact_rows(config: FeatureSelectionConfig) -> List[Dict[str, str]]:
    """整理阶段一相关产物文件名及用途，方便用户对照 JSON/MD/PKL。"""
    return [
        {
            "Artifact": "summary_json",
            "FileName": _artifact_file_name(config.summary_path),
            "Purpose": "阶段一结构化摘要，保存本次特征选择的全部关键结果与超参数。",
        },
        {
            "Artifact": "report_md",
            "FileName": _artifact_file_name(config.report_path),
            "Purpose": "阶段一中文可视化报告，便于直接查看 ACC/F1 与折诊断。",
        },
        {
            "Artifact": "artifact_pkl",
            "FileName": _artifact_file_name(config.artifact_path),
            "Purpose": "阶段一完整中间产物，含折内结果与可复用信息。",
        },
        {
            "Artifact": "mask_pkl",
            "FileName": _artifact_file_name(config.mask_path),
            "Purpose": "后续迁移学习直接读取的冻结特征掩码文件。",
        },
    ]


def _feature_report_param_rows(config: FeatureSelectionConfig) -> List[Dict[str, str]]:
    """给阶段一关键超参数补充中文作用说明。"""
    return [
        {
            "Parameter": "evaluation_protocol",
            "Value": str(config.normalized_evaluation_protocol()),
            "Meaning": "外层评估口径；当前通常表示按被试 LOSO 或分组 KFold 来验证跨被试泛化。",
        },
        {
            "Parameter": "n_splits",
            "Value": str(int(config.n_splits)),
            "Meaning": "计划运行的外层折数，用来控制阶段一评估覆盖范围。",
        },
        {
            "Parameter": "max_folds",
            "Value": "all" if config.max_folds is None else str(int(config.max_folds)),
            "Meaning": "本次实际最多跑多少折；用于快速调试时缩短阶段一运行时间。",
        },
        {
            "Parameter": "train_sample_cap_per_subject",
            "Value": "full_subject" if config.train_sample_cap_per_subject is None else str(int(config.train_sample_cap_per_subject)),
            "Meaning": "每个训练被试最多抽多少样本进入特征选择；越小越省时，越大越稳定。",
        },
        {
            "Parameter": "variance_threshold",
            "Value": f"{float(config.variance_threshold):.6f}",
            "Meaning": "方差过滤阈值；先移除近似常量特征，减少噪声和无效维度。",
        },
        {
            "Parameter": "selector_bins",
            "Value": str(int(config.selector_bins)),
            "Meaning": "MIIFS/MRMR 计算离散互信息时使用的分箱数；越大越细，但估计更敏感。",
        },
        {
            "Parameter": "selector_max_features",
            "Value": str(int(config.selector_max_features)),
            "Meaning": "贪心特征选择允许探索到的最大原始特征数上限。",
        },
        {
            "Parameter": "evaluation_feature_counts",
            "Value": str(config.normalized_feature_counts()),
            "Meaning": "阶段一会逐个比较的候选特征数网格，用来决定不同特征数下的效果。",
        },
        {
            "Parameter": "mask_feature_count",
            "Value": "auto" if config.mask_feature_count is None else str(int(config.mask_feature_count)),
            "Meaning": "最终冻结给迁移学习使用的特征数；必须是候选特征数网格里的一个值。",
        },
        {
            "Parameter": "min_accept_acc",
            "Value": f"{float(config.min_accept_acc):.4f}",
            "Meaning": "阶段一冻结掩码回放 ACC 的最低参考线，低于它通常说明口径不够稳。",
        },
    ]


def _write_feature_selection_report(
    config: FeatureSelectionConfig,
    selector_label: str,
    actual_fold_count: int,
    selected_feature_count: int,
    pre_feature_baseline_summary: Dict[str, Dict[str, float]],
    selected_summary: Dict[str, Dict[str, float]],
    frozen_mask_summary: Dict[str, Dict[str, float]],
    fold_overview_rows: Sequence[Dict[str, Any]],
    feature_count_rows: Sequence[Dict[str, Any]],
    replay_debug_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    top_feature_rows: Sequence[Dict[str, Any]],
) -> Path:
    """生成特征选择阶段的可视化 Markdown 报告。"""
    pre_rows = [{"Classifier": name, **metrics} for name, metrics in pre_feature_baseline_summary.items()]
    selected_rows = [{"Classifier": name, **metrics} for name, metrics in selected_summary.items()]
    frozen_rows = [{"Classifier": name, **metrics} for name, metrics in frozen_mask_summary.items()]
    protocol_name = config.normalized_evaluation_protocol()
    subject_leakage_detected = any(int(row.get("OverlapSubjectCount", 0)) > 0 for row in fold_overview_rows)
    print(f"[FeatureSelection] frozen_mask_feature_count={int(selected_feature_count)}")
    subject_leakage_status = "FAIL" if subject_leakage_detected else "PASS"

    intro_lines = [
        "本报告只关注 ACC 与 F1，同时保留少量诊断指标帮助定位问题。",
        f"当前特征选择器为 `{selector_label}`，评估协议为 `{protocol_name}`，最终冻结掩码特征数为 `{selected_feature_count}`。",
        f"对应 JSON 摘要文件为 `{_artifact_file_name(config.summary_path)}`。",
        "跨被试分析里真正建议拿来衔接迁移学习的是“冻结掩码回放基线”，而不是折内最优结果。",
    ]
    sections = [
        {
            "title": "运行配置",
            "body_lines": [
                f"- 评估协议: `{config.normalized_evaluation_protocol()}`",
                f"- 计划折数: `{config.n_splits}`",
                f"- 实际折数: `{actual_fold_count}`",
                f"- 训练采样上限: `{config.train_sample_cap_per_subject or 'full_subject'}`",
                f"- 被试重叠泄露检查: `{subject_leakage_status}`",
                f"- selector_max_features: `{config.selector_max_features}`",
                f"- 候选特征数: `{config.normalized_feature_counts()}`",
                f"- 对应 JSON 摘要文件: `{_artifact_file_name(config.summary_path)}`",
                f"- 对应 Markdown 报告文件: `{_artifact_file_name(config.report_path)}`",
                f"- 输出掩码文件: `{config.mask_path}`",
            ],
        },
        {
            "title": "产物文件",
            "table": format_metric_table(
                _feature_report_artifact_rows(config),
                [("Artifact", "产物类型"), ("FileName", "文件名"), ("Purpose", "作用")],
            ),
        },
        {
            "title": "参数注释",
            "table": format_metric_table(
                _feature_report_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "折概览",
            "table": format_metric_table(
                fold_overview_rows,
                [
                    "Fold",
                    "RawTrain",
                    "UsedTrain",
                    "Val",
                    "TrainSubjects",
                    "ValSubjects",
                    "OverlapSubjectCount",
                    "OverlapSubjectIds",
                    "LeakageCheck",
                    "BestClassifier",
                    "BestACC",
                    "BestF1",
                ],
            ),
        },
        {
            "title": "候选特征数对比",
            "table": format_metric_table(
                feature_count_rows,
                ["FeatureCount", "LR_ACC", "KNN_ACC", "DT_ACC", "LR_F1", "KNN_F1", "DT_F1"],
            ),
        },
        {
            "title": "特征选择前基线",
            "body_lines": format_metric_bars("未筛选特征 ACC", pre_rows, "ACC")
            + [""]
            + format_metric_bars("未筛选特征 F1", pre_rows, "F1"),
            "table": format_metric_table(pre_rows, ["Classifier", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "折内最优特征结果",
            "body_lines": format_metric_bars("折内候选最优 ACC", selected_rows, "ACC")
            + [""]
            + format_metric_bars("折内候选最优 F1", selected_rows, "F1"),
            "table": format_metric_table(selected_rows, ["Classifier", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "冻结掩码回放基线",
            "body_lines": [
                "这一节才是后续迁移学习最应该对齐的 baseline：",
                "先固定最终选出的原始特征，再只用源域训练、直接打到目标域，不使用任何目标标签。",
            ]
            + format_metric_bars("冻结掩码 ACC", frozen_rows, "ACC")
            + [""]
            + format_metric_bars("冻结掩码 F1", frozen_rows, "F1"),
            "table": format_metric_table(frozen_rows, ["Classifier", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "回放与未筛选对比",
            "table": format_metric_table(
                delta_rows,
                ["Classifier", "Pre_ACC", "Replay_ACC", "Delta_ACC", "Pre_F1", "Replay_F1", "Delta_F1"],
            ),
        },
        {
            "title": "回放折诊断",
            "table": format_metric_table(
                replay_debug_rows,
                ["Fold", "UsableFeatureCount", "LR_ACC", "LR_F1", "KNN_ACC", "KNN_F1", "DT_ACC", "DT_F1"],
            ),
        },
        {
            "title": "高频特征 Top20",
            "table": format_metric_table(top_feature_rows, ["RawIndex", "VoteCount"]),
        },
    ]
    return write_markdown_report(
        path=config.report_path,
        title="特征选择阶段报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def _write_classifier_mask_feature_selection_report(
    config: FeatureSelectionConfig,
    selector_label: str,
    actual_fold_count: int,
    selected_feature_count_by_classifier: Dict[str, int],
    pre_feature_baseline_summary: Dict[str, Dict[str, float]],
    selected_summary_by_classifier: Dict[str, Dict[str, float]],
    frozen_mask_summary: Dict[str, Dict[str, float]],
    fold_overview_rows: Sequence[Dict[str, Any]],
    feature_count_rows: Sequence[Dict[str, Any]],
    classifier_mask_rows: Sequence[Dict[str, Any]],
    replay_debug_rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    top_feature_rows: Sequence[Dict[str, Any]],
) -> Path:
    """生成分类器专属 mask 版本的特征选择报告。"""
    pre_rows = [{"Classifier": name, **metrics} for name, metrics in pre_feature_baseline_summary.items()]
    selected_rows = [
        {
            "Classifier": name,
            "FeatureCount": int(selected_feature_count_by_classifier[name]),
            **selected_summary_by_classifier[name],
        }
        for name in CLASSIFIER_ORDER
    ]
    frozen_rows = [{"Classifier": name, **metrics} for name, metrics in frozen_mask_summary.items()]
    protocol_name = config.normalized_evaluation_protocol()
    subject_leakage_detected = any(int(row.get("OverlapSubjectCount", 0)) > 0 for row in fold_overview_rows)
    subject_leakage_status = "FAIL" if subject_leakage_detected else "PASS"

    intro_lines = [
        "本报告只关注 ACC 和 F1，同时保留少量诊断指标帮助定位问题。",
        f"当前特征选择器为 `{selector_label}`，评估协议为 `{protocol_name}`，最终改为按分类器独立生成 mask。",
        f"分类器专属 mask 特征数：`{_classifier_mask_count_text(selected_feature_count_by_classifier)}`。",
        f"对应 JSON 摘要文件为 `{_artifact_file_name(config.summary_path)}`。",
        "跨被试分析里真正建议拿来衔接迁移学习的是“每个分类器各自的冻结 mask 回放基线”，而不是折内最优结果。",
    ]
    sections = [
        {
            "title": "运行配置",
            "body_lines": [
                f"- 评估协议: `{config.normalized_evaluation_protocol()}`",
                f"- 计划折数: `{config.n_splits}`",
                f"- 实际折数: `{actual_fold_count}`",
                f"- 训练采样上限: `{config.train_sample_cap_per_subject or 'full_subject'}`",
                f"- 被试重叠泄露检查: `{subject_leakage_status}`",
                f"- selector_max_features: `{config.selector_max_features}`",
                f"- 候选特征数: `{config.normalized_feature_counts()}`",
                f"- 分类器专属 mask 特征数: `{_classifier_mask_count_text(selected_feature_count_by_classifier)}`",
                f"- 对应 JSON 摘要文件: `{_artifact_file_name(config.summary_path)}`",
                f"- 对应 Markdown 报告文件: `{_artifact_file_name(config.report_path)}`",
                f"- 输出掩码文件: `{config.mask_path}`",
            ],
        },
        {
            "title": "产物文件",
            "table": format_metric_table(
                _feature_report_artifact_rows(config),
                [("Artifact", "产物类型"), ("FileName", "文件名"), ("Purpose", "作用")],
            ),
        },
        {
            "title": "参数注释",
            "table": format_metric_table(
                _feature_report_param_rows(config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "折概览",
            "table": format_metric_table(
                fold_overview_rows,
                [
                    "Fold",
                    "RawTrain",
                    "UsedTrain",
                    "Val",
                    "TrainSubjects",
                    "ValSubjects",
                    "OverlapSubjectCount",
                    "OverlapSubjectIds",
                    "LeakageCheck",
                    "BestClassifier",
                    "BestACC",
                    "BestF1",
                ],
            ),
        },
        {
            "title": "分类器专属 Mask",
            "table": format_metric_table(
                classifier_mask_rows,
                ["Classifier", "FeatureCount", "CV_ACC", "CV_F1", "Replay_ACC", "Replay_F1"],
            ),
        },
        {
            "title": "候选特征数对比",
            "table": format_metric_table(
                feature_count_rows,
                ["FeatureCount", "LR_ACC", "KNN_ACC", "DT_ACC", "LR_F1", "KNN_F1", "DT_F1"],
            ),
        },
        {
            "title": "特征选择前基线",
            "body_lines": format_metric_bars("未筛选特征 ACC", pre_rows, "ACC")
            + [""]
            + format_metric_bars("未筛选特征 F1", pre_rows, "F1"),
            "table": format_metric_table(pre_rows, ["Classifier", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "折内最优特征结果",
            "body_lines": format_metric_bars("折内候选最优 ACC", selected_rows, "ACC")
            + [""]
            + format_metric_bars("折内候选最优 F1", selected_rows, "F1"),
            "table": format_metric_table(selected_rows, ["Classifier", "FeatureCount", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "冻结掩码回放基线",
            "body_lines": [
                "这一节才是后续迁移学习最应该对齐的 baseline。",
                "每个分类器都使用自己的冻结 mask 做回放，不再强行共享一套特征。",
            ]
            + format_metric_bars("冻结掩码 ACC", frozen_rows, "ACC")
            + [""]
            + format_metric_bars("冻结掩码 F1", frozen_rows, "F1"),
            "table": format_metric_table(frozen_rows, ["Classifier", "ACC", "F1", "BACC", "MCC", "REC", "SPE"]),
        },
        {
            "title": "回放与未筛选对比",
            "table": format_metric_table(
                delta_rows,
                ["Classifier", "Pre_ACC", "Replay_ACC", "Delta_ACC", "Pre_F1", "Replay_F1", "Delta_F1"],
            ),
        },
        {
            "title": "回放折诊断",
            "table": format_metric_table(
                replay_debug_rows,
                [
                    "Fold",
                    "LR_UsableFeatureCount",
                    "LR_ACC",
                    "LR_F1",
                    "KNN_UsableFeatureCount",
                    "KNN_ACC",
                    "KNN_F1",
                    "DT_UsableFeatureCount",
                    "DT_ACC",
                    "DT_F1",
                ],
            ),
        },
        {
            "title": "高频特征 Top10",
            "table": format_metric_table(top_feature_rows, ["Classifier", "FeatureCount", "RawIndex", "VoteCount"]),
        },
    ]
    return write_markdown_report(
        path=config.report_path,
        title="特征选择阶段报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def run_feature_selection_pipeline(config: FeatureSelectionConfig | None = None) -> Dict[str, Any]:
    """运行 MIIFS/MRMR 特征选择，并输出可复用掩码与诊断报告。"""
    if config is None:
        config = FeatureSelectionConfig()

    ensure_artifact_dir()
    config.artifact_path = ensure_ai_output_path(config.artifact_path)
    config.summary_path = ensure_ai_output_path(config.summary_path)
    config.mask_path = ensure_ai_output_path(config.mask_path)
    config.report_path = ensure_ai_output_path(config.report_path)

    feature_counts = config.normalized_feature_counts()
    if config.mask_feature_count is not None and int(config.mask_feature_count) not in feature_counts:
        raise ValueError("mask_feature_count 必须包含在 evaluation_feature_counts 中。")

    X_raw, y, groups = load_deap_features()
    selector_name = config.normalized_selector_name()
    selector_label = selector_display_name(selector_name)

    frequency_counts_by_count: Dict[int, np.ndarray] = {
        int(count): np.zeros(X_raw.shape[1], dtype=np.int32)
        for count in feature_counts
    }
    pre_metric_rows: List[Dict[str, Dict[str, float]]] = []
    metric_rows_by_count: Dict[int, List[Dict[str, Dict[str, float]]]] = {int(count): [] for count in feature_counts}
    fold_rows: List[Dict[str, Any]] = []
    fold_overview_rows: List[Dict[str, Any]] = []

    split_rows = _build_feature_selection_splits(config=config, X_raw=X_raw, y=y, groups=groups)
    print(
        f"[FeatureSelection] selector={selector_label} protocol={config.normalized_evaluation_protocol()} "
        f"planned_folds={len(split_rows)} feature_counts={feature_counts}"
    )

    for split_row in split_rows:
        fold_id = int(split_row["fold_id"])
        if config.max_folds is not None and fold_id > int(config.max_folds):
            break

        train_idx = np.asarray(split_row["train_idx"], dtype=np.int32)
        val_idx = np.asarray(split_row["val_idx"], dtype=np.int32)
        raw_train_size = int(split_row["raw_train_size"])
        fold_start = time.perf_counter()

        print("\n" + "=" * 88)
        print(
            f"[FeatureSelection][Fold {fold_id}] raw_train={raw_train_size} used_train={len(train_idx)} "
            f"val={len(val_idx)} train_subjects={split_row['train_subject_ids']} val_subjects={split_row['val_subject_ids']}"
        )
        print(
            f"[FeatureSelection][Fold {fold_id}] "
            f"train_pos_ratio={float(np.mean(y[train_idx])):.4f} val_pos_ratio={float(np.mean(y[val_idx])):.4f}"
        )
        overlap_subject_ids, overlap_subject_text = _subject_overlap_summary(
            train_subject_ids=split_row["train_subject_ids"],
            val_subject_ids=split_row["val_subject_ids"],
        )
        leakage_check = "FAIL" if overlap_subject_ids else "PASS"
        print(
            f"[FeatureSelection][Fold {fold_id}] leakage_check={leakage_check} "
            f"overlap_subject_count={len(overlap_subject_ids)} overlap_subject_ids={overlap_subject_text}"
        )

        preprocessor = LegacyFoldPreprocessor(variance_threshold=config.variance_threshold)
        X_train = preprocessor.fit_transform(X_raw[train_idx])
        X_val = preprocessor.transform(X_raw[val_idx])
        prep_info = preprocessor.describe()
        print(f"[FeatureSelection][Fold {fold_id}] preprocessor={prep_info}")

        pre_metrics = _evaluate_classifier_set(X_train, y[train_idx], X_val, y[val_idx])
        pre_metric_rows.append(pre_metrics)
        print(
            f"[FeatureSelection][Fold {fold_id}] pre_baseline "
            f"LR_ACC={pre_metrics['LR']['ACC']:.4f} KNN_ACC={pre_metrics['KNN']['ACC']:.4f} "
            f"DT_ACC={pre_metrics['DT']['ACC']:.4f} "
            f"LR_F1={pre_metrics['LR']['F1']:.4f} KNN_F1={pre_metrics['KNN']['F1']:.4f} "
            f"DT_F1={pre_metrics['DT']['F1']:.4f}"
        )

        ranking_processed, selection_scores, selector_debug = run_feature_selector(
            selector_name=selector_name,
            X_train=X_train,
            y_train=y[train_idx],
            selector_max_features=config.selector_max_features,
            selector_bins=config.selector_bins,
        )
        ranking_raw = preprocessor.original_feature_indices(ranking_processed)
        print(
            f"[FeatureSelection][Fold {fold_id}] selector_backend={selector_debug['selector_name']} "
            f"selector_framework={selector_debug.get('framework_name', '-')} "
            f"score_label={selector_debug['score_label']} top10_raw={ranking_raw[:10].tolist()}"
        )
        print(f"[FeatureSelection][Fold {fold_id}] backend_note={selector_debug.get('backend_note', '-')}")
        print(
            f"[FeatureSelection][Fold {fold_id}] top_scores="
            f"{[round(float(score), 6) for score in selection_scores[: min(10, len(selection_scores))]]}"
        )

        fold_feature_results: Dict[int, Dict[str, Dict[str, float]]] = {}
        for feature_count in feature_counts:
            raw_subset = np.asarray(ranking_raw[:feature_count], dtype=np.int32)
            frequency_counts_by_count[int(feature_count)][raw_subset] += 1
            local_indices = np.asarray(ranking_processed[:feature_count], dtype=np.int32)
            metrics = _evaluate_classifier_set(
                X_train=X_train[:, local_indices],
                y_train=y[train_idx],
                X_eval=X_val[:, local_indices],
                y_eval=y[val_idx],
            )
            metric_rows_by_count[int(feature_count)].append(metrics)
            fold_feature_results[int(feature_count)] = metrics
            print(
                f"[FeatureSelection][Fold {fold_id}] n_features={feature_count} "
                f"LR_ACC={metrics['LR']['ACC']:.4f} LR_F1={metrics['LR']['F1']:.4f} "
                f"KNN_ACC={metrics['KNN']['ACC']:.4f} KNN_F1={metrics['KNN']['F1']:.4f} "
                f"DT_ACC={metrics['DT']['ACC']:.4f} DT_F1={metrics['DT']['F1']:.4f}"
            )

        selected_count_for_fold = _select_mask_feature_count(config, fold_feature_results)
        selected_summary_for_fold = fold_feature_results[selected_count_for_fold]
        best_classifier_name, best_classifier_metrics = _best_classifier_from_summary(selected_summary_for_fold)
        fold_duration_sec = float(time.perf_counter() - fold_start)
        print(
            f"[FeatureSelection][Fold {fold_id}] selected_count={selected_count_for_fold} "
            f"best={best_classifier_name} ACC={best_classifier_metrics['ACC']:.4f} "
            f"F1={best_classifier_metrics['F1']:.4f} fold_duration_sec={fold_duration_sec:.2f}"
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "ranking_raw": np.asarray(ranking_raw, dtype=np.int32),
                "ranking_processed": np.asarray(ranking_processed, dtype=np.int32),
                "selector_scores": np.asarray(selection_scores, dtype=np.float32),
                "metrics_by_feature_count": fold_feature_results,
                "pre_feature_baseline_metrics": pre_metrics,
                "selected_feature_count": int(selected_count_for_fold),
                "selected_summary": selected_summary_for_fold,
            }
        )
        fold_overview_rows.append(
            {
                "Fold": fold_id,
                "RawTrain": raw_train_size,
                "UsedTrain": len(train_idx),
                "Val": len(val_idx),
                "TrainSubjects": len(split_row["train_subject_ids"]),
                "ValSubjects": ",".join(str(x) for x in split_row["val_subject_ids"]),
                "OverlapSubjectCount": len(overlap_subject_ids),
                "OverlapSubjectIds": overlap_subject_text,
                "LeakageCheck": leakage_check,
                "BestClassifier": best_classifier_name,
                "BestACC": f"{best_classifier_metrics['ACC']:.4f}",
                "BestF1": f"{best_classifier_metrics['F1']:.4f}",
            }
        )

    actual_fold_count = len(fold_rows)
    if actual_fold_count == 0:
        raise ValueError("没有实际完成任何折，请检查 max_folds 与 n_splits 配置。")

    pre_feature_baseline_summary = _metric_means(pre_metric_rows)
    per_count_summary = {int(count): _metric_means(rows) for count, rows in metric_rows_by_count.items() if rows}
    selected_feature_count = _select_mask_feature_count(config, per_count_summary)
    selected_feature_count_by_classifier = _select_classifier_mask_feature_counts(config, per_count_summary)
    selected_summary = {
        classifier_name: per_count_summary[int(selected_feature_count_by_classifier[classifier_name])][classifier_name]
        for classifier_name in CLASSIFIER_ORDER
    }
    selected_raw_indices_by_classifier: Dict[str, np.ndarray] = {}
    aggregation_debug_rows_by_classifier: Dict[str, List[Dict[str, Any]]] = {}
    for classifier_name in CLASSIFIER_ORDER:
        classifier_selected_raw_indices, classifier_debug_rows = aggregate_raw_feature_votes(
            fold_rows=fold_rows,
            selected_feature_count=int(selected_feature_count_by_classifier[classifier_name]),
        )
        selected_raw_indices_by_classifier[classifier_name] = np.asarray(classifier_selected_raw_indices, dtype=np.int32)
        aggregation_debug_rows_by_classifier[classifier_name] = classifier_debug_rows

    frozen_mask_summary, replay_debug_rows = _evaluate_classifier_specific_masks(
        X_raw=X_raw,
        y=y,
        fold_rows=fold_rows,
        selected_raw_indices_by_classifier=selected_raw_indices_by_classifier,
        variance_threshold=config.variance_threshold,
    )

    frozen_best_name, frozen_best_metrics = _best_classifier_from_summary(frozen_mask_summary)
    gate_passed = float(frozen_best_metrics["ACC"]) >= float(config.min_accept_acc)
    delta_rows = _summary_delta_rows(
        reference_summary=pre_feature_baseline_summary,
        current_summary=frozen_mask_summary,
        reference_label="Pre",
        current_label="Replay",
    )
    feature_count_rows = _feature_count_rows(per_count_summary)
    classifier_mask_rows = _classifier_mask_rows(
        selected_feature_count_by_classifier=selected_feature_count_by_classifier,
        selected_summary_by_classifier=selected_summary,
        frozen_mask_summary=frozen_mask_summary,
    )
    top_feature_rows = _classifier_top_feature_rows(
        frequency_counts_by_count=frequency_counts_by_count,
        selected_feature_count_by_classifier=selected_feature_count_by_classifier,
        selected_raw_indices_by_classifier=selected_raw_indices_by_classifier,
    )
    subject_leakage_detected = any(int(row.get("OverlapSubjectCount", 0)) > 0 for row in fold_overview_rows)

    print("\n[FeatureSelection] 汇总结果")
    print(
        f"[FeatureSelection] frozen_mask_best={frozen_best_name} "
        f"metrics=({_metric_snapshot(frozen_best_metrics)}) gate_passed={gate_passed} "
        f"classifier_mask_counts={_classifier_mask_count_text(selected_feature_count_by_classifier)}"
    )
    print(f"[FeatureSelection] subject_leakage_detected={subject_leakage_detected}")
    for classifier_name in CLASSIFIER_ORDER:
        print(
            f"[FeatureSelection][ReplaySummary] classifier={classifier_name} "
            f"metrics=({_metric_snapshot(frozen_mask_summary[classifier_name])}) "
            f"selected_count={selected_feature_count_by_classifier[classifier_name]}"
        )
    for row in delta_rows:
        print(
            f"[FeatureSelection] delta classifier={row['Classifier']} "
            f"ACC={row['Delta_ACC']} F1={row['Delta_F1']} "
            f"BACC={row.get('Delta_BACC', 'n/a')} MCC={row.get('Delta_MCC', 'n/a')}"
        )

    report_path = _write_classifier_mask_feature_selection_report(
        config=config,
        selector_label=selector_label,
        actual_fold_count=actual_fold_count,
        selected_feature_count_by_classifier=selected_feature_count_by_classifier,
        pre_feature_baseline_summary=pre_feature_baseline_summary,
        selected_summary_by_classifier=selected_summary,
        frozen_mask_summary=frozen_mask_summary,
        fold_overview_rows=fold_overview_rows,
        feature_count_rows=feature_count_rows,
        classifier_mask_rows=classifier_mask_rows,
        replay_debug_rows=replay_debug_rows,
        delta_rows=delta_rows,
        top_feature_rows=top_feature_rows,
    )

    feature_raw_mask_by_classifier: Dict[str, np.ndarray] = {}
    for classifier_name in CLASSIFIER_ORDER:
        classifier_mask = np.zeros(X_raw.shape[1], dtype=bool)
        classifier_mask[np.asarray(selected_raw_indices_by_classifier[classifier_name], dtype=np.int32)] = True
        feature_raw_mask_by_classifier[classifier_name] = classifier_mask

    shared_selected_raw_indices = np.asarray(selected_raw_indices_by_classifier[frozen_best_name], dtype=np.int32)
    shared_feature_raw_mask = np.asarray(feature_raw_mask_by_classifier[frozen_best_name], dtype=bool)

    artifact = {
        "selector_name": selector_name,
        "classifier_mask_mode": "per_classifier",
        "config": asdict(config),
        "actual_fold_count": actual_fold_count,
        "selected_feature_count": int(selected_feature_count_by_classifier[frozen_best_name]),
        "selected_feature_count_by_classifier": {
            classifier_name: int(selected_feature_count_by_classifier[classifier_name])
            for classifier_name in CLASSIFIER_ORDER
        },
        "selected_raw_indices": shared_selected_raw_indices,
        "selected_raw_indices_by_classifier": selected_raw_indices_by_classifier,
        "feature_raw_mask": shared_feature_raw_mask,
        "feature_raw_mask_by_classifier": feature_raw_mask_by_classifier,
        "frequency_counts_by_feature_count": frequency_counts_by_count,
        "aggregation_debug_rows": aggregation_debug_rows_by_classifier[frozen_best_name],
        "aggregation_debug_rows_by_classifier": aggregation_debug_rows_by_classifier,
        "fold_overview_rows": fold_overview_rows,
        "per_feature_count_summary": per_count_summary,
        "pre_feature_baseline_summary": pre_feature_baseline_summary,
        "feature_selection_cv_summary": selected_summary,
        "classifier_mask_rows": classifier_mask_rows,
        "frozen_mask_baseline_summary": frozen_mask_summary,
        "source_only_transfer_baseline_summary": frozen_mask_summary,
        "subject_leakage_detected": subject_leakage_detected,
        "best_classifier_name": frozen_best_name,
        "best_classifier_metrics": frozen_best_metrics,
        "gate_passed": gate_passed,
        "report_path": str(report_path),
    }
    mask_artifact = {
        "selector_name": selector_name,
        "classifier_mask_mode": "per_classifier",
        "config": asdict(config),
        "selected_feature_count": int(selected_feature_count_by_classifier[frozen_best_name]),
        "selected_feature_count_by_classifier": {
            classifier_name: int(selected_feature_count_by_classifier[classifier_name])
            for classifier_name in CLASSIFIER_ORDER
        },
        "selected_raw_indices": shared_selected_raw_indices,
        "selected_raw_indices_by_classifier": selected_raw_indices_by_classifier,
        "feature_raw_mask": shared_feature_raw_mask,
        "feature_raw_mask_by_classifier": feature_raw_mask_by_classifier,
        "feature_selection_cv_summary": selected_summary,
        "classifier_mask_rows": classifier_mask_rows,
        "frozen_mask_baseline_summary": frozen_mask_summary,
        "source_only_transfer_baseline_summary": frozen_mask_summary,
        "subject_leakage_detected": subject_leakage_detected,
        "best_classifier_name": frozen_best_name,
        "best_classifier_metrics": frozen_best_metrics,
        "gate_passed": gate_passed,
    }

    save_pickle(artifact, config.artifact_path)
    save_pickle(mask_artifact, config.mask_path)
    save_json(to_serializable(artifact), config.summary_path)
    print(f"[FeatureSelection] artifact_path={config.artifact_path}")
    print(f"[FeatureSelection] mask_path={config.mask_path}")
    print(f"[FeatureSelection] summary_path={config.summary_path}")
    print(f"[FeatureSelection] report_path={report_path}")
    return artifact
