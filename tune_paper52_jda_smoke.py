from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.common import save_json, to_serializable
from AI.reporting import format_metric_table, write_markdown_report
from AI.transfer_learning_pipeline import TransferLearningConfig, run_transfer_learning_pipeline


DEFAULT_CANDIDATES: Sequence[Dict[str, Any]] = (
    {
        "tag": "baseline_1nn_default",
        "jda_dim": 48,
        "jda_iterations": 6,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "1nn",
        "jda_pseudo_neighbors": 1,
        "jda_pseudo_change_tol": 1e-3,
        "jda_mmd_delta_tol": 1e-3,
        "jda_confidence_delta_tol": 5e-3,
        "jda_early_stop_patience": 2,
        "jda_min_iterations": 2,
    },
    {
        "tag": "knn3_default",
        "jda_dim": 48,
        "jda_iterations": 6,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "knn",
        "jda_pseudo_neighbors": 3,
        "jda_pseudo_change_tol": 1e-3,
        "jda_mmd_delta_tol": 1e-3,
        "jda_confidence_delta_tol": 5e-3,
        "jda_early_stop_patience": 2,
        "jda_min_iterations": 2,
    },
    {
        "tag": "knn5_default",
        "jda_dim": 48,
        "jda_iterations": 6,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "knn",
        "jda_pseudo_neighbors": 5,
        "jda_pseudo_change_tol": 1e-3,
        "jda_mmd_delta_tol": 1e-3,
        "jda_confidence_delta_tol": 5e-3,
        "jda_early_stop_patience": 2,
        "jda_min_iterations": 2,
    },
    {
        "tag": "lr_default",
        "jda_dim": 48,
        "jda_iterations": 6,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "lr",
        "jda_pseudo_neighbors": 3,
        "jda_pseudo_change_tol": 1e-3,
        "jda_mmd_delta_tol": 1e-3,
        "jda_confidence_delta_tol": 5e-3,
        "jda_early_stop_patience": 2,
        "jda_min_iterations": 2,
    },
    {
        "tag": "svm_default",
        "jda_dim": 48,
        "jda_iterations": 6,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "svm",
        "jda_pseudo_neighbors": 3,
        "jda_pseudo_change_tol": 1e-3,
        "jda_mmd_delta_tol": 1e-3,
        "jda_confidence_delta_tol": 5e-3,
        "jda_early_stop_patience": 2,
        "jda_min_iterations": 2,
    },
    {
        "tag": "knn3_relaxed_stop",
        "jda_dim": 48,
        "jda_iterations": 8,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "knn",
        "jda_pseudo_neighbors": 3,
        "jda_pseudo_change_tol": 3e-3,
        "jda_mmd_delta_tol": 3e-3,
        "jda_confidence_delta_tol": 1e-2,
        "jda_early_stop_patience": 3,
        "jda_min_iterations": 3,
    },
    {
        "tag": "lr_relaxed_stop",
        "jda_dim": 48,
        "jda_iterations": 8,
        "jda_n_components": 64,
        "jda_lambda": 1.0,
        "jda_pseudo_labeler": "lr",
        "jda_pseudo_neighbors": 3,
        "jda_pseudo_change_tol": 3e-3,
        "jda_mmd_delta_tol": 3e-3,
        "jda_confidence_delta_tol": 1e-2,
        "jda_early_stop_patience": 3,
        "jda_min_iterations": 3,
    },
)


def build_parser() -> argparse.ArgumentParser:
    """构建论文式 JDA smoke 调参脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="针对论文式 JDA 的伪标签器和收敛判据做小规模 smoke 调参。")
    parser.add_argument("--mask-path", type=Path, default=Path("AI/artifacts/feature_mask_miifs_loso_k200.pkl"))
    parser.add_argument("--feature-source-mode", type=str, choices=["mask", "full"], default="mask")
    parser.add_argument("--feature-source-label", type=str, default="MIIFS_LOSO_K200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-target-subjects", type=int, default=3, help="搜索阶段使用的目标单元数量。")
    parser.add_argument("--confirm-target-subjects", type=int, default=5, help="确认阶段使用的目标单元数量。")
    parser.add_argument("--max-source-subjects", type=int, default=6)
    parser.add_argument("--source-sample-cap", type=int, default=96)
    parser.add_argument("--mmd-sample-cap", type=int, default=160)
    parser.add_argument("--gate-repeat-count", type=int, default=3)
    parser.add_argument("--jda-reg", type=float, default=1e-6)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("AI/artifacts/paper52_jda_tuning_v2"),
        help="本轮论文式 JDA 调参实验的输出目录。",
    )
    parser.add_argument(
        "--reference-summary-path",
        type=Path,
        default=Path("AI/artifacts/transfer_learning_summary_miifs_loso_k200_supervised_paper52cmp_smoke.json"),
        help="单次实验报告里用于对比的参考结果路径。",
    )
    return parser


def _candidate_key(summary: Dict[str, Any]) -> tuple:
    """定义论文式 JDA 候选参数的排序规则。"""
    # 中文作用: 优先挑真正保留迁移的配置，再看 raw transfer 相对 target-only 的提升，最后参考最终 ACC/F1。
    gate_rows = summary.get("gate_summary_rows", [])
    keep_total = int(sum(int(row.get("TransferKept", 0)) for row in gate_rows))
    classifiers = ["LR", "KNN", "DT"]
    raw_acc_gains: List[float] = []
    raw_f1_gains: List[float] = []
    final_acc_values: List[float] = []
    final_f1_values: List[float] = []
    for classifier_name in classifiers:
        raw_row = summary["raw_transfer_summary"][classifier_name]
        target_row = summary["target_only_summary"][classifier_name]
        final_row = summary["transfer_summary"][classifier_name]
        raw_acc_gains.append(float(raw_row["ACC"]["mean"]) - float(target_row["ACC"]["mean"]))
        raw_f1_gains.append(float(raw_row["F1"]["mean"]) - float(target_row["F1"]["mean"]))
        final_acc_values.append(float(final_row["ACC"]["mean"]))
        final_f1_values.append(float(final_row["F1"]["mean"]))
    mean_raw_acc_gain = float(sum(raw_acc_gains) / len(raw_acc_gains))
    mean_raw_f1_gain = float(sum(raw_f1_gains) / len(raw_f1_gains))
    mean_final_acc = float(sum(final_acc_values) / len(final_acc_values))
    mean_final_f1 = float(sum(final_f1_values) / len(final_f1_values))
    return (
        keep_total,
        min(mean_raw_acc_gain, mean_raw_f1_gain),
        mean_raw_acc_gain,
        mean_raw_f1_gain,
        min(mean_final_acc, mean_final_f1),
        mean_final_acc,
        mean_final_f1,
    )


def _collect_candidate_row(tag: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """把单次实验 summary 压缩成调参汇总表的一行。"""
    gate_rows = summary.get("gate_summary_rows", [])
    classifiers = ["LR", "KNN", "DT"]
    raw_acc_gains: List[float] = []
    raw_f1_gains: List[float] = []
    final_acc_values: List[float] = []
    final_f1_values: List[float] = []
    for classifier_name in classifiers:
        raw_row = summary["raw_transfer_summary"][classifier_name]
        target_row = summary["target_only_summary"][classifier_name]
        final_row = summary["transfer_summary"][classifier_name]
        raw_acc_gains.append(float(raw_row["ACC"]["mean"]) - float(target_row["ACC"]["mean"]))
        raw_f1_gains.append(float(raw_row["F1"]["mean"]) - float(target_row["F1"]["mean"]))
        final_acc_values.append(float(final_row["ACC"]["mean"]))
        final_f1_values.append(float(final_row["F1"]["mean"]))
    keep_total = int(sum(int(row.get("TransferKept", 0)) for row in gate_rows))
    gate_total = int(sum(int(row.get("TargetOnlyGate", 0)) for row in gate_rows))
    config = summary["config"]
    return {
        "Tag": tag,
        "PseudoLabeler": str(config["jda_pseudo_labeler"]),
        "PseudoNeighbors": int(config["jda_pseudo_neighbors"]),
        "PseudoTol": float(config["jda_pseudo_change_tol"]),
        "MMDTol": float(config["jda_mmd_delta_tol"]),
        "ConfTol": float(config["jda_confidence_delta_tol"]),
        "Patience": int(config["jda_early_stop_patience"]),
        "MinIter": int(config["jda_min_iterations"]),
        "LR_Final": f"{float(summary['transfer_summary']['LR']['ACC']['mean']):.4f}/{float(summary['transfer_summary']['LR']['F1']['mean']):.4f}",
        "KNN_Final": f"{float(summary['transfer_summary']['KNN']['ACC']['mean']):.4f}/{float(summary['transfer_summary']['KNN']['F1']['mean']):.4f}",
        "DT_Final": f"{float(summary['transfer_summary']['DT']['ACC']['mean']):.4f}/{float(summary['transfer_summary']['DT']['F1']['mean']):.4f}",
        "MeanFinal_ACC": f"{float(sum(final_acc_values) / len(final_acc_values)):.4f}",
        "MeanFinal_F1": f"{float(sum(final_f1_values) / len(final_f1_values)):.4f}",
        "MeanRawGain_ACC": f"{float(sum(raw_acc_gains) / len(raw_acc_gains)):+.4f}",
        "MeanRawGain_F1": f"{float(sum(raw_f1_gains) / len(raw_f1_gains)):+.4f}",
        "TransferKeptTotal": int(keep_total),
        "TargetOnlyGateTotal": int(gate_total),
        "SummaryPath": str(summary["config"]["summary_path"]),
        "ReportPath": str(summary["report_path"]),
    }


def _build_base_config(args: argparse.Namespace, max_target_subjects: int, summary_path: Path, report_path: Path) -> TransferLearningConfig:
    """按统一 smoke 设置构造论文式 JDA 基础配置。"""
    # 中文作用: 固定数据口径，只让伪标签器和收敛判据变化，避免把别的因素混进调参结果。
    reference_summary_path = args.reference_summary_path
    if reference_summary_path is not None and not Path(reference_summary_path).exists():
        reference_summary_path = None
    return TransferLearningConfig(
        mask_path=args.mask_path,
        feature_source_mode=args.feature_source_mode,
        feature_source_label=args.feature_source_label,
        summary_path=summary_path,
        report_path=report_path,
        random_state=int(args.seed),
        evaluation_protocol="loso",
        cv_splits=5,
        max_target_subjects=int(max_target_subjects),
        target_train_ratio=0.60,
        target_val_ratio=0.20,
        target_test_ratio=0.20,
        max_source_subjects=int(args.max_source_subjects),
        source_sample_cap=int(args.source_sample_cap),
        mmd_sample_cap=int(args.mmd_sample_cap),
        mmd_prefix_top_k=0,
        transfer_variant="enhanced_jda",
        target_repeat_grid=(1, 2, 4),
        source_target_ratio_grid=(0.25, 0.5, 0.75, 1.0),
        jda_dim=48,
        jda_iterations=6,
        jda_n_components=64,
        jda_lambda=1.0,
        jda_reg=float(args.jda_reg),
        jda_pseudo_labeler="1nn",
        jda_pseudo_neighbors=1,
        jda_pseudo_change_tol=1e-3,
        jda_mmd_delta_tol=1e-3,
        jda_confidence_delta_tol=5e-3,
        jda_early_stop_patience=2,
        jda_min_iterations=2,
        reference_summary_path=reference_summary_path,
        source_positive_ratio_gap_threshold=0.08,
        gate_repeat_count=int(args.gate_repeat_count),
    )


def _run_single_candidate(
    base_config: TransferLearningConfig,
    candidate: Dict[str, Any],
    max_target_subjects: int,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """运行单个论文式 JDA 候选配置，并返回其 summary。"""
    tag = str(candidate["tag"])
    summary_path = artifacts_dir / f"transfer_learning_summary_{tag}.json"
    report_path = artifacts_dir / f"transfer_learning_report_{tag}.md"
    config = copy.deepcopy(base_config)
    config.max_target_subjects = int(max_target_subjects)
    config.summary_path = summary_path
    config.report_path = report_path
    config.jda_dim = int(candidate["jda_dim"])
    config.jda_iterations = int(candidate["jda_iterations"])
    config.jda_n_components = int(candidate["jda_n_components"])
    config.jda_lambda = float(candidate["jda_lambda"])
    config.jda_pseudo_labeler = str(candidate["jda_pseudo_labeler"])
    config.jda_pseudo_neighbors = int(candidate["jda_pseudo_neighbors"])
    config.jda_pseudo_change_tol = float(candidate["jda_pseudo_change_tol"])
    config.jda_mmd_delta_tol = float(candidate["jda_mmd_delta_tol"])
    config.jda_confidence_delta_tol = float(candidate["jda_confidence_delta_tol"])
    config.jda_early_stop_patience = int(candidate["jda_early_stop_patience"])
    config.jda_min_iterations = int(candidate["jda_min_iterations"])
    print(
        f"[Paper52TuneV2] start tag={tag} target_units={max_target_subjects} "
        f"pseudo={config.jda_pseudo_labeler}(k={config.jda_pseudo_neighbors}) "
        f"pc_tol={config.jda_pseudo_change_tol:.4f} mmd_tol={config.jda_mmd_delta_tol:.4f} "
        f"conf_tol={config.jda_confidence_delta_tol:.4f} patience={config.jda_early_stop_patience} "
        f"min_iter={config.jda_min_iterations}"
    )
    summary = run_transfer_learning_pipeline(config)
    print(
        f"[Paper52TuneV2] done tag={tag} "
        f"LR={summary['transfer_summary']['LR']['ACC']['mean']:.4f}/{summary['transfer_summary']['LR']['F1']['mean']:.4f} "
        f"KNN={summary['transfer_summary']['KNN']['ACC']['mean']:.4f}/{summary['transfer_summary']['KNN']['F1']['mean']:.4f} "
        f"DT={summary['transfer_summary']['DT']['ACC']['mean']:.4f}/{summary['transfer_summary']['DT']['F1']['mean']:.4f}"
    )
    return summary


def _write_tuning_report(
    report_path: Path,
    search_rows: Sequence[Dict[str, Any]],
    best_search_row: Dict[str, Any],
    confirm_row: Dict[str, Any],
    confirm_summary: Dict[str, Any],
) -> Path:
    """生成论文式 JDA 第二轮调参 Markdown 报告。"""
    intro_lines = [
        "本报告聚焦论文 5.2 风格 JDA 的两处关键部件：伪标签分类器和收敛判据。",
        "本轮固定结构参数为 `dim=48, T=6/8, n_components=64, lambda=1.0`，不再重复搜索 MMD 选源结构。",
        "排序优先级为：保留迁移数量、raw transfer 相对 target-only 的增益、最终 ACC/F1。",
    ]
    sections = [
        {
            "title": "搜索结果",
            "body_lines": [
                f"- 搜索阶段最优候选: `{best_search_row['Tag']}`",
                f"- 最优候选伪标签器: `{best_search_row['PseudoLabeler']}`",
                f"- 最优候选收敛配置: `pc_tol={float(best_search_row['PseudoTol']):.4f}, mmd_tol={float(best_search_row['MMDTol']):.4f}, conf_tol={float(best_search_row['ConfTol']):.4f}, patience={best_search_row['Patience']}, min_iter={best_search_row['MinIter']}`",
            ],
            "table": format_metric_table(
                search_rows,
                [
                    ("Tag", "候选标签"),
                    ("PseudoLabeler", "伪标签器"),
                    ("PseudoNeighbors", "K值"),
                    ("PseudoTol", "伪标签阈值"),
                    ("MMDTol", "MMD阈值"),
                    ("ConfTol", "置信度阈值"),
                    ("Patience", "耐心轮数"),
                    ("MinIter", "最少迭代"),
                    ("LR_Final", "LR 最终ACC/F1"),
                    ("KNN_Final", "KNN 最终ACC/F1"),
                    ("DT_Final", "DT 最终ACC/F1"),
                    ("MeanFinal_ACC", "平均最终ACC"),
                    ("MeanFinal_F1", "平均最终F1"),
                    ("MeanRawGain_ACC", "平均Raw-Target ACC增益"),
                    ("MeanRawGain_F1", "平均Raw-Target F1增益"),
                    ("TransferKeptTotal", "保留迁移总数"),
                    ("TargetOnlyGateTotal", "回退总数"),
                ],
            ),
        },
        {
            "title": "确认结果",
            "body_lines": [
                f"- 确认阶段候选: `{confirm_row['Tag']}`",
                f"- 伪标签器: `{confirm_row['PseudoLabeler']}`，K=`{confirm_row['PseudoNeighbors']}`",
                f"- 收敛配置: `pc_tol={float(confirm_row['PseudoTol']):.4f}, mmd_tol={float(confirm_row['MMDTol']):.4f}, conf_tol={float(confirm_row['ConfTol']):.4f}, patience={confirm_row['Patience']}, min_iter={confirm_row['MinIter']}`",
                f"- 确认阶段报告: `{confirm_row['ReportPath']}`",
            ],
            "table": format_metric_table(
                [
                    {
                        "Classifier": classifier_name,
                        "Final_ACC": f"{float(confirm_summary['transfer_summary'][classifier_name]['ACC']['mean']):.4f}",
                        "Final_F1": f"{float(confirm_summary['transfer_summary'][classifier_name]['F1']['mean']):.4f}",
                        "Raw_ACC": f"{float(confirm_summary['raw_transfer_summary'][classifier_name]['ACC']['mean']):.4f}",
                        "Raw_F1": f"{float(confirm_summary['raw_transfer_summary'][classifier_name]['F1']['mean']):.4f}",
                        "Target_ACC": f"{float(confirm_summary['target_only_summary'][classifier_name]['ACC']['mean']):.4f}",
                        "Target_F1": f"{float(confirm_summary['target_only_summary'][classifier_name]['F1']['mean']):.4f}",
                    }
                    for classifier_name in ["LR", "KNN", "DT"]
                ],
                [
                    ("Classifier", "分类器"),
                    ("Final_ACC", "最终ACC"),
                    ("Final_F1", "最终F1"),
                    ("Raw_ACC", "Raw ACC"),
                    ("Raw_F1", "Raw F1"),
                    ("Target_ACC", "Target-only ACC"),
                    ("Target_F1", "Target-only F1"),
                ],
            ),
        },
    ]
    return write_markdown_report(
        path=report_path,
        title="论文式 JDA 伪标签器与收敛判据调参报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def main() -> None:
    """执行论文式 JDA 的第二轮小规模调参，并输出 JSON/Markdown 汇总。"""
    args = build_parser().parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_search_config = _build_base_config(
        args=args,
        max_target_subjects=int(args.search_target_subjects),
        summary_path=artifacts_dir / "_placeholder_search.json",
        report_path=artifacts_dir / "_placeholder_search.md",
    )

    search_summaries: List[Dict[str, Any]] = []
    search_rows: List[Dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        summary = _run_single_candidate(
            base_config=base_search_config,
            candidate=candidate,
            max_target_subjects=int(args.search_target_subjects),
            artifacts_dir=artifacts_dir,
        )
        search_summaries.append({"tag": str(candidate["tag"]), "summary": summary, "candidate": dict(candidate)})
        search_rows.append(_collect_candidate_row(str(candidate["tag"]), summary))

    best_search_summary = max(search_summaries, key=lambda row: _candidate_key(row["summary"]))
    best_search_row = next(row for row in search_rows if row["Tag"] == best_search_summary["tag"])

    base_confirm_config = _build_base_config(
        args=args,
        max_target_subjects=int(args.confirm_target_subjects),
        summary_path=artifacts_dir / "_placeholder_confirm.json",
        report_path=artifacts_dir / "_placeholder_confirm.md",
    )
    confirm_candidate = dict(best_search_summary["candidate"])
    confirm_candidate["tag"] = f"{best_search_summary['tag']}_confirm"
    confirm_summary = _run_single_candidate(
        base_config=base_confirm_config,
        candidate=confirm_candidate,
        max_target_subjects=int(args.confirm_target_subjects),
        artifacts_dir=artifacts_dir,
    )
    confirm_row = _collect_candidate_row(str(confirm_candidate["tag"]), confirm_summary)

    aggregate_summary = {
        "search_target_subjects": int(args.search_target_subjects),
        "confirm_target_subjects": int(args.confirm_target_subjects),
        "search_rows": search_rows,
        "best_search_row": best_search_row,
        "confirm_row": confirm_row,
        "confirm_summary_path": str(confirm_summary["config"]["summary_path"]),
        "confirm_report_path": str(confirm_summary["report_path"]),
    }
    summary_path = artifacts_dir / "transfer_learning_summary_paper52_jda_tuning_v2_smoke.json"
    report_path = artifacts_dir / "transfer_learning_report_paper52_jda_tuning_v2_smoke.md"
    save_json(to_serializable(aggregate_summary), summary_path)
    _write_tuning_report(
        report_path=report_path,
        search_rows=search_rows,
        best_search_row=best_search_row,
        confirm_row=confirm_row,
        confirm_summary=confirm_summary,
    )
    print(f"[Paper52TuneV2] aggregate_summary_path={summary_path}")
    print(f"[Paper52TuneV2] aggregate_report_path={report_path}")


if __name__ == "__main__":
    main()
