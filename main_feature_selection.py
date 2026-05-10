from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 临时修补 wwt.runtime_safe 导入问题 ---
import types
wwt_runtime_safe = types.ModuleType("wwt.runtime_safe")
def apply_safe_runtime_env():
    pass
wwt_runtime_safe.apply_safe_runtime_env = apply_safe_runtime_env
sys.modules["wwt.runtime_safe"] = wwt_runtime_safe
# -----------------------------------------

from AI.feature_selection_pipeline import FeatureSelectionConfig, run_feature_selection_pipeline


def build_parser() -> argparse.ArgumentParser:
    """构建特征选择主程序的命令行参数。"""
    parser = argparse.ArgumentParser(description="运行跨被试特征选择，并生成可复用掩码与 Markdown 报告。")
    parser.add_argument(
        "--evaluation-protocol",
        type=str,
        choices=["loso_subject", "group_kfold", "sample_kfold"],
        default="loso_subject",
        help="评估协议，默认使用严格留一被试分析。",
    )
    parser.add_argument("--selector-name", type=str, choices=["miifs", "mrmr"], default="miifs")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=2,
        help="调试阶段默认只跑前 2 折；传 0 表示按 n_splits 跑完整实验。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-threshold", type=float, default=1e-3)
    parser.add_argument("--selector-max-features", type=int, default=96)
    parser.add_argument("--selector-bins", type=int, default=32)
    parser.add_argument(
        "--feature-grid",
        type=int,
        nargs="+",
        default=[48, 64, 80, 96],
        help="按候选特征数评估 ACC/F1，默认使用较轻量的调试网格。",
    )
    parser.add_argument(
        "--mask-feature-count",
        type=int,
        default=0,
        help="传正数时强制使用该特征数作为最终掩码。",
    )
    parser.add_argument(
        "--train-sample-cap-per-subject",
        type=int,
        default=160,
        help="每个训练被试最多保留多少样本，默认先做小样本调试。",
    )
    parser.add_argument("--min-accept-acc", type=float, default=0.60)
    parser.add_argument("--artifact-path", type=Path, default=Path("AI/artifacts/feature_selection_cv_latest.pkl"))
    parser.add_argument("--summary-path", type=Path, default=Path("AI/artifacts/feature_selection_summary_latest.json"))
    parser.add_argument("--mask-path", type=Path, default=Path("AI/artifacts/miifs_mask_latest.pkl"))
    parser.add_argument("--report-path", type=Path, default=Path("AI/artifacts/feature_selection_report_latest.md"))
    return parser


def main() -> None:
    """解析命令行参数并启动特征选择流程。"""
    args = build_parser().parse_args()
    max_folds = None if args.max_folds is None or args.max_folds <= 0 else args.max_folds
    train_sample_cap_per_subject = (
        None
        if args.train_sample_cap_per_subject is None or args.train_sample_cap_per_subject <= 0
        else args.train_sample_cap_per_subject
    )
    config = FeatureSelectionConfig(
        evaluation_protocol=args.evaluation_protocol,
        selector_name=args.selector_name,
        n_splits=args.n_splits,
        max_folds=max_folds,
        random_state=args.seed,
        variance_threshold=args.variance_threshold,
        selector_max_features=args.selector_max_features,
        selector_bins=args.selector_bins,
        evaluation_feature_counts=args.feature_grid,
        mask_feature_count=None if args.mask_feature_count <= 0 else args.mask_feature_count,
        min_accept_acc=args.min_accept_acc,
        train_sample_cap_per_subject=train_sample_cap_per_subject,
        artifact_path=args.artifact_path,
        summary_path=args.summary_path,
        mask_path=args.mask_path,
        report_path=args.report_path,
    )
    run_feature_selection_pipeline(config)


if __name__ == "__main__":
    main()