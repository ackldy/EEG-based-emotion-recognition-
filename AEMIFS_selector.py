from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y

try:
    import cupy as cp

    HAS_CUPY = True
except ModuleNotFoundError:
    cp = np  # type: ignore[assignment]
    HAS_CUPY = False


def _discretize_gpu(data: cp.ndarray, bins: int) -> cp.ndarray:
    """把连续特征离散化到 `[0, bins-1]`，供信息量公式共用。"""
    if cp.issubdtype(data.dtype, cp.integer):
        return data.astype(cp.int32, copy=False)

    if data.ndim == 1:
        min_val = data.min()
        max_val = data.max()
        if float(max_val - min_val) <= 0.0:
            return cp.zeros(data.shape, dtype=cp.int32)
        data_norm = (data - min_val) / (max_val - min_val + 1e-9)
        data_int = cp.floor(data_norm * (bins - 1e-9)).astype(cp.int32)
        return cp.clip(data_int, 0, bins - 1)

    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    range_val = max_val - min_val
    range_val = cp.where(range_val == 0, 1e-9, range_val)
    data_norm = (data - min_val) / (range_val + 1e-9)
    data_int = cp.floor(data_norm * (bins - 1e-9)).astype(cp.int32)
    return cp.clip(data_int, 0, bins - 1)


def _entropy_gpu_from_int(x: cp.ndarray) -> cp.ndarray:
    """计算离散变量的一维信息熵。"""
    counts = cp.bincount(x)
    p = counts[counts > 0] / x.size
    return -cp.sum(p * cp.log(p))


def _joint_entropy_gpu_from_int(x: cp.ndarray, y: cp.ndarray, bins: int) -> cp.ndarray:
    """计算两个离散变量的联合熵。"""
    xy = x * bins + y
    counts = cp.bincount(xy)
    p = counts[counts > 0] / x.size
    return -cp.sum(p * cp.log(p))


def _joint_entropy_3d_gpu_from_int(x: cp.ndarray, y: cp.ndarray, z: cp.ndarray, bins: int) -> cp.ndarray:
    """计算三个离散变量的联合熵。"""
    xyz = (x * bins + y) * bins + z
    counts = cp.bincount(xyz)
    p = counts[counts > 0] / x.size
    return -cp.sum(p * cp.log(p))


def _preview_feature_scores(score_items: Sequence[tuple[int, float]], top_k: int = 5) -> str:
    """把候选特征分数压缩成便于打印的预览文本。"""
    ordered_items = sorted(score_items, key=lambda item: (-float(item[1]), int(item[0])))
    preview_items = ordered_items[: min(top_k, len(ordered_items))]
    if not preview_items:
        return "-"
    return " | ".join(f"f{feature_idx}:{score_value:.6f}" for feature_idx, score_value in preview_items)


class _UnifiedGreedyInfoSelectorBase(BaseEstimator, SelectorMixin):
    """统一的贪心信息量特征选择骨架，算法差异只放在评分公式里。"""

    def __init__(
        self,
        n_features: int | str = "auto",
        bins: int = 32,
        verbose: int = 1,
        progress_every: int = 10,
    ) -> None:
        """记录统一贪心框架的公共超参数。"""
        self.n_features = n_features
        self.bins = bins
        self.verbose = verbose
        self.progress_every = progress_every
        self._support_mask: np.ndarray | None = None
        self.ranking_: List[int] = []
        self.selection_scores_: List[float] = []

    def _selector_label(self) -> str:
        """返回当前选择器在日志中的显示名称。"""
        raise NotImplementedError

    def _score_label(self) -> str:
        """返回当前选择器输出分数的显示名称。"""
        raise NotImplementedError

    def _score_candidate(
        self,
        feature_idx: int,
        selected: Sequence[int],
        stats: Dict[str, Any],
    ) -> float:
        """计算候选特征在当前已选集合下的贪心分数。"""
        raise NotImplementedError

    def _get_support_mask(self) -> np.ndarray:
        """返回布尔掩码形式的已选特征集合。"""
        if self._support_mask is None:
            raise ValueError("请先运行 fit() 再读取特征掩码。")
        return self._support_mask

    def _print(self, message: str) -> None:
        """按统一前缀打印调试日志。"""
        if self.verbose:
            print(f"[{self._selector_label()}] {message}")

    def _build_statistics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """预计算统一贪心框架需要的单特征统计量与缓存。"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
        n_samples, n_features_all = X_scaled.shape
        self._print(
            f"数据集 samples={n_samples} features={n_features_all} bins={self.bins} "
            f"device={'cupy' if HAS_CUPY else 'numpy'}"
        )

        fit_start = time.perf_counter()
        X_gpu = cp.asarray(X_scaled)
        X_binned = _discretize_gpu(X_gpu, self.bins)
        _, y_mapped = np.unique(y, return_inverse=True)
        y_gpu = cp.asarray(y_mapped, dtype=cp.int32)
        hy = float(_entropy_gpu_from_int(y_gpu))
        self._print(
            f"开始预计算单特征统计量 hy={hy:.6f} "
            f"pair_cache_est_mb={(n_features_all * n_features_all * 4) / (1024 ** 2):.2f}"
        )

        feature_entropy = np.zeros(n_features_all, dtype=np.float32)
        feature_y_joint_entropy = np.zeros(n_features_all, dtype=np.float32)
        log_stride = max(int(self.progress_every) * 4, 64)
        for feature_idx in range(n_features_all):
            x_feature = X_binned[:, feature_idx]
            feature_entropy[feature_idx] = float(_entropy_gpu_from_int(x_feature))
            feature_y_joint_entropy[feature_idx] = float(_joint_entropy_gpu_from_int(x_feature, y_gpu, self.bins))
            should_print = (
                self.verbose
                and (
                    feature_idx < 5
                    or (feature_idx + 1) % log_stride == 0
                    or (feature_idx + 1) == n_features_all
                )
            )
            if should_print:
                relevance_value = feature_entropy[feature_idx] + hy - feature_y_joint_entropy[feature_idx]
                self._print(
                    f"precompute feature={feature_idx + 1}/{n_features_all} "
                    f"entropy={feature_entropy[feature_idx]:.6f} "
                    f"relevance={relevance_value:.6f} "
                    f"elapsed={time.perf_counter() - fit_start:.2f}s"
                )

        relevance = feature_entropy + hy - feature_y_joint_entropy
        self._print(f"top10_relevance={_preview_feature_scores(list(enumerate(relevance.tolist())), top_k=10)}")
        return {
            "X_binned": X_binned,
            "y_gpu": y_gpu,
            "hy": hy,
            "feature_entropy": feature_entropy,
            "feature_y_joint_entropy": feature_y_joint_entropy,
            "relevance": relevance.astype(np.float32, copy=False),
            "pair_joint_entropy_cache": np.full((n_features_all, n_features_all), np.nan, dtype=np.float32),
            "miifs_pair_term_cache": None,
        }

    def _pair_joint_entropy(self, feature_idx: int, selected_idx: int, stats: Dict[str, Any]) -> float:
        """按需缓存候选特征与已选特征的联合熵。"""
        cache = stats["pair_joint_entropy_cache"]
        cached_value = float(cache[feature_idx, selected_idx])
        if not np.isnan(cached_value):
            return cached_value

        joint_entropy = float(
            _joint_entropy_gpu_from_int(
                stats["X_binned"][:, feature_idx],
                stats["X_binned"][:, selected_idx],
                self.bins,
            )
        )
        cache[feature_idx, selected_idx] = joint_entropy
        cache[selected_idx, feature_idx] = joint_entropy
        return joint_entropy

    def _pair_mutual_information(self, feature_idx: int, selected_idx: int, stats: Dict[str, Any]) -> float:
        """计算候选特征与已选特征之间的互信息。"""
        feature_entropy = stats["feature_entropy"]
        joint_entropy = self._pair_joint_entropy(feature_idx, selected_idx, stats)
        return float(feature_entropy[feature_idx] + feature_entropy[selected_idx] - joint_entropy)

    def _miifs_pair_term(self, feature_idx: int, selected_idx: int, stats: Dict[str, Any]) -> float:
        """计算 MIIFS 在一对特征上的方向性增益项。"""
        cache = stats["miifs_pair_term_cache"]
        if cache is None:
            feature_count = int(stats["feature_entropy"].shape[0])
            cache = np.full((feature_count, feature_count), np.nan, dtype=np.float32)
            stats["miifs_pair_term_cache"] = cache

        cached_value = float(cache[feature_idx, selected_idx])
        if not np.isnan(cached_value):
            return cached_value

        feature_entropy = stats["feature_entropy"]
        feature_y_joint_entropy = stats["feature_y_joint_entropy"]
        pair_joint_entropy = self._pair_joint_entropy(feature_idx, selected_idx, stats)
        pair_y_joint_entropy = float(
            _joint_entropy_3d_gpu_from_int(
                stats["X_binned"][:, feature_idx],
                stats["y_gpu"],
                stats["X_binned"][:, selected_idx],
                self.bins,
            )
        )
        pair_term = (
            -float(feature_entropy[feature_idx])
            - 2.0 * float(feature_entropy[selected_idx])
            + 2.0 * pair_joint_entropy
            + float(feature_y_joint_entropy[feature_idx])
            + 2.0 * float(feature_y_joint_entropy[selected_idx])
            - float(stats["hy"])
            - 2.0 * pair_y_joint_entropy
        )
        cache[feature_idx, selected_idx] = pair_term
        return pair_term

    def _should_stop_auto(self) -> bool:
        """在自动选特征模式下，根据分数曲线趋稳程度提前停止。"""
        if self.n_features != "auto" or len(self.selection_scores_) <= 10:
            return False
        smoothed_gradient = signal.savgol_filter(self.selection_scores_[1:], 9, 2, 1)
        return bool(np.abs(np.mean(smoothed_gradient[-5:])) < 1e-6)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_UnifiedGreedyInfoSelectorBase":
        """执行统一贪心框架，并由子类评分公式决定每一步挑谁。"""
        X, y = check_X_y(X, y)
        self._support_mask = None
        self.ranking_ = []
        self.selection_scores_ = []

        fit_start = time.perf_counter()
        stats = self._build_statistics(X, y)
        relevance = stats["relevance"]
        n_features_all = int(relevance.shape[0])

        selected: List[int] = []
        remaining = list(range(n_features_all))
        first_feature = int(np.argmax(relevance))
        selected.append(first_feature)
        remaining.remove(first_feature)
        self.selection_scores_.append(float(relevance[first_feature]))
        self._print(
            f"首个特征={first_feature} {self._score_label()}={self.selection_scores_[-1]:.6f} "
            f"fit_elapsed={time.perf_counter() - fit_start:.2f}s"
        )

        target_feature_count = np.inf if self.n_features == "auto" else int(self.n_features)
        while len(selected) < target_feature_count and remaining:
            round_start = time.perf_counter()
            best_score = -1e18
            best_feature = -1
            second_score = -1e18
            candidate_scores: List[tuple[int, float]] = []

            for feature_idx in remaining:
                candidate_score = float(self._score_candidate(feature_idx, selected, stats))
                candidate_scores.append((int(feature_idx), candidate_score))
                if candidate_score > best_score:
                    second_score = best_score
                    best_score = candidate_score
                    best_feature = int(feature_idx)
                elif candidate_score > second_score:
                    second_score = candidate_score

            selected.append(best_feature)
            remaining.remove(best_feature)
            self.selection_scores_.append(float(best_score))

            should_print = (
                self.verbose
                and (
                    len(selected) <= 5
                    or (np.isfinite(target_feature_count) and len(selected) == int(target_feature_count))
                    or len(selected) % max(int(self.progress_every), 1) == 0
                )
            )
            if should_print:
                score_margin = float(best_score - second_score) if second_score > -1e17 else float("nan")
                self._print(
                    f"已选特征数={len(selected)} last_feature={best_feature} "
                    f"{self._score_label()}={best_score:.6f} "
                    f"next_gap={score_margin:.6f} "
                    f"top_candidates={_preview_feature_scores(candidate_scores, top_k=3)} "
                    f"round_elapsed={time.perf_counter() - round_start:.2f}s"
                )

            if self._should_stop_auto():
                self._print("达到自动收敛条件，提前停止。")
                break

        support_mask = np.zeros(n_features_all, dtype=bool)
        support_mask[selected] = True
        self._support_mask = support_mask
        self.ranking_ = selected
        self._print(f"fit 完成 total_elapsed={time.perf_counter() - fit_start:.2f}s")
        return self


class FastMIIFSGPU(_UnifiedGreedyInfoSelectorBase):
    """统一贪心框架下的AEMIFS 版本，只保留AEMIFS 的评分公式。"""

    def _selector_label(self) -> str:
        """返回 AEMIFS 日志标签。"""
        return "FastAEMIFS"

    def _score_label(self) -> str:
        """返回 AEMIFS分数字段名称。"""
        return "AEMIFSScore"

    def _score_candidate(
        self,
        feature_idx: int,
        selected: Sequence[int],
        stats: Dict[str, Any],
    ) -> float:
        """按 AEMIFS公式计算候选特征分数。"""
        pair_term_sum = 0.0
        for selected_idx in selected:
            pair_term_sum += self._miifs_pair_term(feature_idx, int(selected_idx), stats)
        return (float(stats["relevance"][feature_idx]) + pair_term_sum) / max(len(selected), 1)


class FastMRMRGPU(_UnifiedGreedyInfoSelectorBase):
    """统一贪心框架下的 mRMR 版本，只保留 mRMR-D 的评分公式。"""

    def _selector_label(self) -> str:
        """返回 mRMR 日志标签。"""
        return "FastMRMR"

    def _score_label(self) -> str:
        """返回 mRMR 分数字段名称。"""
        return "MRMRScore"

    def _score_candidate(
        self,
        feature_idx: int,
        selected: Sequence[int],
        stats: Dict[str, Any],
    ) -> float:
        """按 mRMR-D 公式计算候选特征分数。"""
        redundancy_sum = 0.0
        for selected_idx in selected:
            redundancy_sum += self._pair_mutual_information(feature_idx, int(selected_idx), stats)
        redundancy_mean = redundancy_sum / max(len(selected), 1)
        return float(stats["relevance"][feature_idx]) - redundancy_mean
