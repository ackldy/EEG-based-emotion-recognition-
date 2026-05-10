from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from wwt.runtime_safe import apply_safe_runtime_env

apply_safe_runtime_env()

import numpy as np
import numpy.core
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "AI"
DATA_PATH = ROOT_DIR / "wwt" / "deap_features_1184.pkl"
ARTIFACT_DIR = AI_DIR / "artifacts"
FEATURE_ARTIFACT_PATH = ARTIFACT_DIR / "feature_selection_cv_best.pkl"
FEATURE_SUMMARY_PATH = ARTIFACT_DIR / "feature_selection_summary_best.json"
TRANSFER_SUMMARY_PATH = ARTIFACT_DIR / "transfer_learning_summary_best.json"
DEFAULT_SUBJECT_COUNT = 32
DEFAULT_RANDOM_STATE = 42


def ensure_numpy_pickle_compat() -> None:
    """兼容旧版 pickle 中使用的 `numpy._core` 模块别名。"""
    sys.modules.setdefault("numpy._core", numpy.core)


def ensure_artifact_dir() -> Path:
    """确保所有实验产物统一写入 `AI/artifacts` 目录。"""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def ensure_ai_output_path(path: Path) -> Path:
    """将输出路径约束到 `AI` 目录下，避免实验文件散落到仓库其他位置。"""
    path = Path(path)

    if path.is_absolute():
        try:
            path.relative_to(AI_DIR)
        except ValueError as exc:
            raise ValueError(f"输出路径必须位于 AI 目录下，当前路径为: {path}") from exc
        return path

    if path.parts and path.parts[0].lower() == "ai":
        return path
    return Path("") / path


def save_pickle(obj: Any, path: Path) -> None:
    """将 Python 对象序列化为 pickle 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_pickle(path: Path) -> Any:
    """从磁盘读取一个 pickle 产物。"""
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def save_json(obj: Any, path: Path) -> None:
    """将 JSON 兼容对象写入磁盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, ensure_ascii=False, allow_nan=False)


def to_serializable(obj: Any) -> Any:
    """递归把 numpy、Path 等对象转换成可安全写入 JSON 的普通类型。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        scalar = obj.item()
        if isinstance(scalar, float):
            return scalar if math.isfinite(scalar) else None
        return scalar
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    return obj


def dataset_layout_summary(n_samples: int, subject_count: int = DEFAULT_SUBJECT_COUNT) -> Dict[str, int]:
    """根据样本总数推断被试布局，便于检查迁移学习是否按真实被试分组。"""
    if n_samples % subject_count != 0:
        raise ValueError(
            f"样本数 {n_samples} 不能被被试数 {subject_count} 整除，"
            "当前代码默认每个被试的窗口数相同。"
        )

    samples_per_subject = n_samples // subject_count
    trials_per_subject = samples_per_subject // 20 if samples_per_subject % 20 == 0 else -1
    return {
        "n_samples": int(n_samples),
        "subject_count": int(subject_count),
        "samples_per_subject": int(samples_per_subject),
        "windows_per_trial": 20,
        "trials_per_subject": int(trials_per_subject),
    }


def load_deap_features(data_path: Path = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 DEAP 特征文件，并返回特征矩阵、二分类标签和被试编号。"""
    #潜台词： DEAP 官方数据文件是 Python 2 时代用 pickle 保存的，且内含 numpy 数组。latin1 编码是为了防止 Python 3 读取时因字符串编码差异而报错。
    # raw_obj 是一个字典，内含 "signals"（特征）、"labels1"（效价评分）和 "labels2"（唤醒度评分）。
    ensure_numpy_pickle_compat()
    print(f"Loading feature file: {data_path}")
    with data_path.open("rb") as file_obj:
        raw_obj = pickle.load(file_obj, encoding="latin1")
    #从raw_obj文件里的键为signal的信息读出来并且将它转化为numpy数组
    X = np.asarray(raw_obj["signals"], dtype=np.float32)
    #把 1–9 分的情绪愉悦度打分，变成了计算机能直接分类的 0（不开心）和 1（开心）两种标签。
    y = (np.asarray(raw_obj["labels1"]) >= 5).astype(np.int32)
    groups = subject_ids_from_samples(X.shape[0], subject_count=DEFAULT_SUBJECT_COUNT)
    layout = dataset_layout_summary(X.shape[0], subject_count=DEFAULT_SUBJECT_COUNT)

    print(f"Loaded X shape={X.shape}, y shape={y.shape}")
    print(
        "[DatasetLayout] "
        f"subject_count={layout['subject_count']} "
        f"samples_per_subject={layout['samples_per_subject']} "
        f"trials_per_subject={layout['trials_per_subject']} "
        f"windows_per_trial={layout['windows_per_trial']}"
    )
    print(f"Class balance: {class_distribution(y)}")
    return X, y, groups


def subject_ids_from_samples(n_samples: int, subject_count: int = DEFAULT_SUBJECT_COUNT) -> np.ndarray:
    """在“每个被试贡献相同窗口数”的前提下，为每个样本生成被试编号。"""
    if n_samples % subject_count != 0:
        raise ValueError(
            f"Sample count {n_samples} is not divisible by subject count {subject_count}. "
            "The current pipeline assumes a fixed number of windows per subject."
        )
    samples_per_subject = n_samples // subject_count
    return np.repeat(np.arange(subject_count, dtype=np.int32), samples_per_subject)


def class_distribution(y: np.ndarray) -> Dict[int, int]:
    """统计每个类别包含多少样本。"""
    values, counts = np.unique(y, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def make_outer_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """为按被试分组的外层交叉验证生成折叠。"""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[Tuple[int, np.ndarray, np.ndarray]] = []
    dummy_X = np.zeros((len(y), 1), dtype=np.float32)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(dummy_X, y, groups), start=1):
        folds.append((fold_id, train_idx, test_idx))
    return folds


def make_group_validation_split(
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """尽量在保持被试分组的前提下构造一次训练/验证划分。"""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 4:
        n_splits = min(4, len(unique_groups))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        dummy_X = np.zeros((len(y), 1), dtype=np.float32)
        train_idx, val_idx = next(splitter.split(dummy_X, y, groups))
        return train_idx, val_idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    dummy_X = np.zeros((len(y), 1), dtype=np.float32)
    train_idx, val_idx = next(splitter.split(dummy_X, y))
    return train_idx, val_idx


def stratified_group_subsample_indices(
    y: np.ndarray,
    groups: np.ndarray,
    max_per_group: int,
    random_state: int,
) -> np.ndarray:
    """在每个被试组内做近似分层采样，用于加快互信息估计。"""
    if max_per_group <= 0:
        return np.arange(len(y), dtype=np.int32)

    rng = np.random.default_rng(random_state)
    chosen: List[int] = []

    for group_id in np.unique(groups):
        group_idx = np.flatnonzero(groups == group_id)
        if len(group_idx) <= max_per_group:
            chosen.extend(group_idx.tolist())
            continue

        group_y = y[group_idx]
        group_choice: List[int] = []

        for class_id in np.unique(group_y):
            cls_idx = group_idx[group_y == class_id]
            ratio = len(cls_idx) / len(group_idx)
            take = int(round(max_per_group * ratio))
            take = min(len(cls_idx), max(1, take))
            group_choice.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

        if len(group_choice) > max_per_group:
            group_choice = rng.choice(np.array(group_choice, dtype=np.int32), size=max_per_group, replace=False).tolist()

        if len(group_choice) < max_per_group:
            remainder = max_per_group - len(group_choice)
            remaining_pool = np.setdiff1d(group_idx, np.array(group_choice, dtype=np.int32), assume_unique=False)
            if len(remaining_pool) > 0:
                extra = rng.choice(remaining_pool, size=min(remainder, len(remaining_pool)), replace=False)
                group_choice.extend(extra.tolist())

        chosen.extend(group_choice)

    chosen = np.array(sorted(chosen), dtype=np.int32)
    return chosen


@dataclass
class FoldPreprocessor:
    """封装单折内的方差过滤、对数变换和鲁棒标准化。"""

    variance_threshold_value: float = 1e-3
    variance_threshold: VarianceThreshold | None = None
    positive_mask: np.ndarray | None = None
    scaler: RobustScaler | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在当前训练折上拟合预处理器，并返回处理后的特征。"""
        self.variance_threshold = VarianceThreshold(threshold=self.variance_threshold_value)
        X_vt = self.variance_threshold.fit_transform(X)
        self.positive_mask = np.all(X_vt > 0, axis=0)
        X_log = X_vt.copy()
        if np.any(self.positive_mask):
            X_log[:, self.positive_mask] = np.log1p(X_log[:, self.positive_mask])

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_log)
        return np.asarray(X_scaled, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """把已经拟合好的预处理规则应用到其他数据划分。"""
        if self.variance_threshold is None or self.positive_mask is None or self.scaler is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        X_vt = self.variance_threshold.transform(X)
        X_log = X_vt.copy()
        if np.any(self.positive_mask):
            X_log[:, self.positive_mask] = np.log1p(X_log[:, self.positive_mask])
        X_scaled = self.scaler.transform(X_log)
        return np.asarray(X_scaled, dtype=np.float32)

    def original_feature_indices(self, processed_indices: Sequence[int]) -> np.ndarray:
        """把预处理后特征空间的索引映射回原始特征索引。"""
        if self.variance_threshold is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        kept = self.variance_threshold.get_support(indices=True)
        processed_indices = np.asarray(processed_indices, dtype=np.int32)
        return kept[processed_indices]

    def describe(self) -> Dict[str, Any]:
        """汇总当前折预处理器的状态，便于日志打印和结果落盘。"""
        if self.variance_threshold is None or self.positive_mask is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        kept_count = int(np.sum(self.variance_threshold.get_support()))
        return {
            "input_feature_count": int(self.variance_threshold.n_features_in_),
            "post_variance_feature_count": kept_count,
            "log1p_feature_count": int(np.sum(self.positive_mask)),
        }


def get_classifier_search_space() -> Dict[str, List[Dict[str, Any]]]:
    """给 LR、KNN、DT 提供一组轻量但实用的候选超参数。"""
    return {
        "KNN": [
            {"n_neighbors": 3, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 5, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 7, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 9, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 11, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 13, "weights": "distance", "metric": "cosine", "n_jobs": -1},
            {"n_neighbors": 5, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 7, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 9, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 11, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
            {"n_neighbors": 13, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        ],
        "LR": [
            {"C": 0.125, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 0.25, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 0.5, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 2.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 4.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
            {"C": 8.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
        ],
        "DT": [
            {"max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
            {"max_depth": None, "min_samples_leaf": 4, "class_weight": "balanced"},
            {"max_depth": 8, "min_samples_leaf": 2, "class_weight": "balanced"},
            {"max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced"},
            {"max_depth": 12, "min_samples_leaf": 4, "class_weight": "balanced"},
            {"max_depth": 14, "min_samples_leaf": 4, "class_weight": "balanced"},
            {"max_depth": 18, "min_samples_leaf": 6, "class_weight": "balanced"},
            {"max_depth": 20, "min_samples_leaf": 8, "class_weight": "balanced"},
            {"max_depth": 24, "min_samples_leaf": 8, "class_weight": "balanced"},
        ],
    }


def build_classifier(name: str, params: Dict[str, Any], random_state: int) -> Any:
    """根据分类器名称和参数字典实例化模型。"""
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "LR":
        return LogisticRegression(random_state=random_state, **params)
    if name == "DT":
        return DecisionTreeClassifier(random_state=random_state, **params)
    raise ValueError(f"Unsupported classifier: {name}")


def positive_class_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """从支持的分类器中提取正类概率或近似概率分数。"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    raise ValueError("Model does not expose probability-like scores.")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """计算二分类任务常用指标，并补充更利于定位问题的诊断指标。"""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_prob = None if y_prob is None else np.asarray(y_prob, dtype=np.float32)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    mcc = 0.0
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "ACC": float(acc),
        "PRE": float(pre),
        "REC": float(rec),
        "F1": float(f1),
        "AUC": float(auc),
        "BACC": float(bacc),
        "MCC": float(mcc),
        "SPE": float(spe),
    }



def summarize_metric_list(metric_list: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """对一组指标字典求均值和标准差。"""
    metric_list = list(metric_list)
    if not metric_list:
        return {}

    summary: Dict[str, Dict[str, float]] = {}
    keys = sorted(metric_list[0].keys())
    for key in keys:
        values = np.asarray([metric[key] for metric in metric_list], dtype=np.float64)
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, Any]]:
    """按被试分别计算指标，便于观察哪些被试最难迁移。"""
    rows: List[Dict[str, Any]] = []
    for subject_id in np.unique(groups):
        subject_mask = groups == subject_id
        subject_metrics = binary_metrics(y_true[subject_mask], y_pred[subject_mask], y_prob[subject_mask])
        rows.append(
            {
                "subject_id": int(subject_id),
                "n_samples": int(np.sum(subject_mask)),
                "metrics": subject_metrics,
            }
        )
    return rows


def merge_subject_metric_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """把逐被试指标汇总成整体均值/标准差。"""
    metrics_only = [row["metrics"] for row in rows]
    return summarize_metric_list(metrics_only)

