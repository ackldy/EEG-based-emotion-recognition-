from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from AI.miifs_selector import FastMIIFSGPU, FastMRMRGPU


def normalize_selector_name(selector_name: str) -> str:
    """规范化特征选择器名称，只接受当前项目支持的后端。"""
    normalized_name = str(selector_name).lower().strip()
    if normalized_name not in {"miifs", "mrmr"}:
        raise ValueError(f"不支持的 selector_name: {selector_name!r}")
    return normalized_name


def selector_display_name(selector_name: str) -> str:
    """返回写报告时更直观的特征选择器展示名称。"""
    normalized_name = normalize_selector_name(selector_name)
    return "MIIFS" if normalized_name == "miifs" else "MRMR"


def _build_selector(selector_name: str, selector_max_features: int, selector_bins: int) -> tuple[Any, Dict[str, str]]:
    """构建统一贪心框架下的选择器实例与说明信息。"""
    normalized_name = normalize_selector_name(selector_name)
    selector_registry: Dict[str, tuple[Any, str, str]] = {
        "miifs": (
            FastMIIFSGPU,
            "MIIFSScore",
            "统一贪心框架；评分公式=MIIFS 原始方向性增益项。",
        ),
        "mrmr": (
            FastMRMRGPU,
            "MRMRScore",
            "统一贪心框架；评分公式=mRMR-D（relevance - mean redundancy）。",
        ),
    }
    selector_class, score_label, backend_note = selector_registry[normalized_name]
    selector = selector_class(
        n_features=selector_max_features,
        bins=selector_bins,
        verbose=1,
        progress_every=25,
    )
    debug_info = {
        "selector_name": normalized_name,
        "score_label": score_label,
        "framework_name": "UnifiedGreedyInfoSelectorGPU",
        "backend_note": backend_note,
    }
    return selector, debug_info


def run_feature_selector(
    selector_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    selector_max_features: int,
    selector_bins: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """在当前训练折上运行统一贪心框架，并返回逐步排名与真实分数。"""
    selector, debug_info = _build_selector(
        selector_name=selector_name,
        selector_max_features=selector_max_features,
        selector_bins=selector_bins,
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    selector.fit(X_train, y_train)
    ranking_processed = np.asarray(selector.ranking_, dtype=np.int32)
    selection_scores = np.asarray(selector.selection_scores_, dtype=np.float32)
    return ranking_processed, selection_scores, debug_info


def aggregate_raw_feature_votes(
    fold_rows: Sequence[Dict[str, Any]],
    selected_feature_count: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """把多折的原始特征排名聚合成一个稳定的全局原始特征列表。"""
    if selected_feature_count <= 0:
        raise ValueError("selected_feature_count 必须为正整数。")

    vote_count: Dict[int, int] = {}
    rank_sum: Dict[int, int] = {}
    first_seen: Dict[int, int] = {}
    best_rank: Dict[int, int] = {}
    sequence_id = 0

    for fold_row in sorted(fold_rows, key=lambda row: int(row["fold_id"])):
        ranking_raw = np.asarray(fold_row["ranking_raw"], dtype=np.int32)[:selected_feature_count]
        for local_rank, raw_idx in enumerate(ranking_raw.tolist()):
            raw_idx = int(raw_idx)
            vote_count[raw_idx] = vote_count.get(raw_idx, 0) + 1
            rank_sum[raw_idx] = rank_sum.get(raw_idx, 0) + int(local_rank)
            best_rank[raw_idx] = min(best_rank.get(raw_idx, int(local_rank)), int(local_rank))
            if raw_idx not in first_seen:
                first_seen[raw_idx] = sequence_id
            sequence_id += 1

    ordered_raw_indices = sorted(
        vote_count.keys(),
        key=lambda raw_idx: (
            -vote_count[raw_idx],
            first_seen[raw_idx],
            rank_sum[raw_idx],
            best_rank[raw_idx],
            raw_idx,
        ),
    )
    chosen_raw_indices = np.asarray(ordered_raw_indices[:selected_feature_count], dtype=np.int32)
    debug_rows = [
        {
            "RawIndex": int(raw_idx),
            "VoteCount": int(vote_count[raw_idx]),
            "BestRank": int(best_rank[raw_idx]),
            "RankSum": int(rank_sum[raw_idx]),
            "FirstSeen": int(first_seen[raw_idx]),
        }
        for raw_idx in chosen_raw_indices[: min(20, len(chosen_raw_indices))]
    ]
    return chosen_raw_indices, debug_rows
