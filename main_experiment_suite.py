from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.common import ensure_ai_output_path, ensure_artifact_dir, save_json, to_serializable
from AI.feature_selection_pipeline import FeatureSelectionConfig, run_feature_selection_pipeline
from AI.reporting import format_metric_bars, format_metric_table, write_markdown_report
from AI.transfer_learning_pipeline import CLASSIFIER_ORDER, TransferLearningConfig, run_transfer_learning_pipeline


@dataclass(frozen=True)
class ExperimentSpec:
    """描述一组固定口径的“特征选择 + 迁移学习”联合实验。"""

    selector_name: str
    selector_label: str
    protocol_key: str
    protocol_label: str
    feature_protocol: str
    transfer_protocol: str
    n_splits: int
    selector_max_features: int
    evaluation_feature_counts: tuple[int, ...]
    train_sample_cap_per_subject: int | None

    @property
    def key(self) -> str:
        """返回写入产物文件名时使用的稳定实验键。"""
        return f"{self.selector_name}_{self.protocol_key}"

    @property
    def display_name(self) -> str:
        """返回报告中更适合直接展示的实验名称。"""
        return f"{self.selector_label} / {self.protocol_label}"


def build_parser() -> argparse.ArgumentParser:
    """构建四组联合实验总控脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="统一运行 MIIFS/mRMR 在 10 折和 LOSO 下的特征选择与迁移学习实验。")
    parser.add_argument("--mode", type=str, choices=["smoke", "full"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-threshold", type=float, default=1e-3)
    parser.add_argument("--selector-bins", type=int, default=32)
    parser.add_argument("--min-accept-acc", type=float, default=0.60)
    parser.add_argument("--groupkfold-selector-max-features", type=int, default=192)
    parser.add_argument("--groupkfold-feature-counts", type=int, nargs="+", default=[128, 160, 192])
    parser.add_argument("--groupkfold-train-sample-cap-per-subject", type=int, default=64)
    parser.add_argument("--loso-selector-max-features", type=int, default=96)
    parser.add_argument("--loso-feature-counts", type=int, nargs="+", default=[48, 64, 80, 96])
    parser.add_argument("--loso-train-sample-cap-per-subject", type=int, default=160)
    parser.add_argument("--target-train-ratio", type=float, default=0.60)
    parser.add_argument("--target-val-ratio", type=float, default=0.20)
    parser.add_argument("--target-test-ratio", type=float, default=0.20)
    parser.add_argument("--max-source-subjects", type=int, default=12)
    parser.add_argument("--source-sample-cap", type=int, default=160)
    parser.add_argument("--mmd-sample-cap", type=int, default=200)
    parser.add_argument(
        "--mmd-prefix-top-k",
        type=int,
        default=0,
        help="迁移学习里累计前缀的搜索上限；小于等于 0 表示搜索前1、前2、前3...全部累计前缀。",
    )
    parser.add_argument("--target-repeat-grid", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--source-target-ratio-grid", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--source-positive-ratio-gap-threshold", type=float, default=0.08)
    parser.add_argument("--gate-repeat-count", type=int, default=3)
    parser.add_argument(
        "--only-experiment",
        type=str,
        nargs="*",
        default=None,
        help="只运行指定实验键，例如 miifs_groupkfold10 / miifs_loso / mrmr_groupkfold10 / mrmr_loso。",
    )
    parser.add_argument(
        "--suite-summary-path",
        type=Path,
        default=Path("AI/artifacts/experiment_suite_summary_full.json"),
    )
    parser.add_argument(
        "--suite-report-path",
        type=Path,
        default=Path("AI/artifacts/experiment_suite_report_full.md"),
    )
    return parser


def _mode_suffix(mode: str) -> str:
    """返回用于文件名的运行模式后缀。"""
    return "smoke" if str(mode).lower().strip() == "smoke" else "full"


def _timestamp_text() -> str:
    """返回当前本地时间字符串，便于写入报告和日志。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mmd_prefix_search_text(limit: int) -> str:
    """把累计前缀搜索上限转成更易读的文本。"""
    # 中文作用: 统一把命令行里的 0/负数解释成“搜索全部累计前缀”。
    return "all_prefixes" if int(limit) <= 0 else str(int(limit))


def _artifact_paths(spec: ExperimentSpec, mode: str) -> Dict[str, Path]:
    """为单组实验生成全部位于 AI/artifacts 下的输出路径。"""
    suffix = _mode_suffix(mode)
    return {
        "feature_artifact_path": Path(f"AI/artifacts/feature_selection_cv_{spec.key}_{suffix}.pkl"),
        "feature_summary_path": Path(f"AI/artifacts/feature_selection_summary_{spec.key}_{suffix}.json"),
        "feature_mask_path": Path(f"AI/artifacts/feature_mask_{spec.key}_{suffix}.pkl"),
        "feature_report_path": Path(f"AI/artifacts/feature_selection_report_{spec.key}_{suffix}.md"),
        "transfer_summary_path": Path(f"AI/artifacts/transfer_learning_summary_{spec.key}_{suffix}.json"),
        "transfer_report_path": Path(f"AI/artifacts/transfer_learning_report_{spec.key}_{suffix}.md"),
    }


def _build_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    """按照用户要求构造四组实验规格。"""
    group_counts = tuple(int(value) for value in args.groupkfold_feature_counts)
    loso_counts = tuple(int(value) for value in args.loso_feature_counts)
    return [
        ExperimentSpec(
            selector_name="miifs",
            selector_label="MIIFS",
            protocol_key="groupkfold10",
            protocol_label="10 折分组交叉验证",
            feature_protocol="group_kfold",
            transfer_protocol="group_kfold",
            n_splits=10,
            selector_max_features=int(args.groupkfold_selector_max_features),
            evaluation_feature_counts=group_counts,
            train_sample_cap_per_subject=None
            if int(args.groupkfold_train_sample_cap_per_subject) <= 0
            else int(args.groupkfold_train_sample_cap_per_subject),
        ),
        ExperimentSpec(
            selector_name="miifs",
            selector_label="MIIFS",
            protocol_key="loso",
            protocol_label="留一被试验证",
            feature_protocol="loso_subject",
            transfer_protocol="loso",
            n_splits=32,
            selector_max_features=int(args.loso_selector_max_features),
            evaluation_feature_counts=loso_counts,
            train_sample_cap_per_subject=None
            if int(args.loso_train_sample_cap_per_subject) <= 0
            else int(args.loso_train_sample_cap_per_subject),
        ),
        ExperimentSpec(
            selector_name="mrmr",
            selector_label="MRMR",
            protocol_key="groupkfold10",
            protocol_label="10 折分组交叉验证",
            feature_protocol="group_kfold",
            transfer_protocol="group_kfold",
            n_splits=10,
            selector_max_features=int(args.groupkfold_selector_max_features),
            evaluation_feature_counts=group_counts,
            train_sample_cap_per_subject=None
            if int(args.groupkfold_train_sample_cap_per_subject) <= 0
            else int(args.groupkfold_train_sample_cap_per_subject),
        ),
        ExperimentSpec(
            selector_name="mrmr",
            selector_label="MRMR",
            protocol_key="loso",
            protocol_label="留一被试验证",
            feature_protocol="loso_subject",
            transfer_protocol="loso",
            n_splits=32,
            selector_max_features=int(args.loso_selector_max_features),
            evaluation_feature_counts=loso_counts,
            train_sample_cap_per_subject=None
            if int(args.loso_train_sample_cap_per_subject) <= 0
            else int(args.loso_train_sample_cap_per_subject),
        ),
    ]


def _runtime_overrides(spec: ExperimentSpec, args: argparse.Namespace) -> Dict[str, Any]:
    """根据 full/smoke 模式返回运行时覆盖参数。"""
    if _mode_suffix(args.mode) != "smoke":
        return {
            "feature_max_folds": None,
            "feature_selector_max_features": int(spec.selector_max_features),
            "feature_counts": list(spec.evaluation_feature_counts),
            "feature_train_sample_cap_per_subject": None if spec.selector_name == "miifs" else spec.train_sample_cap_per_subject,
            "transfer_max_target_subjects": None,
            "transfer_max_source_subjects": int(args.max_source_subjects),
            "transfer_source_sample_cap": int(args.source_sample_cap),
            "transfer_mmd_sample_cap": int(args.mmd_sample_cap),
            "transfer_mmd_prefix_top_k": 0 if int(args.mmd_prefix_top_k) <= 0 else int(args.mmd_prefix_top_k),
            "transfer_target_repeat_grid": tuple(int(value) for value in args.target_repeat_grid),
            "transfer_source_target_ratio_grid": tuple(float(value) for value in args.source_target_ratio_grid),
            "transfer_source_positive_ratio_gap_threshold": float(args.source_positive_ratio_gap_threshold),
            "transfer_gate_repeat_count": int(args.gate_repeat_count),
        }

    if spec.protocol_key == "groupkfold10":
        return {
            "feature_max_folds": 2,
            "feature_selector_max_features": min(int(spec.selector_max_features), 96),
            "feature_counts": [64, 96],
            "feature_train_sample_cap_per_subject": 32,
            "transfer_max_target_subjects": 2,
            "transfer_max_source_subjects": 4,
            "transfer_source_sample_cap": 64,
            "transfer_mmd_sample_cap": 64,
            "transfer_mmd_prefix_top_k": 0 if int(args.mmd_prefix_top_k) <= 0 else int(args.mmd_prefix_top_k),
            "transfer_target_repeat_grid": (1, 2),
            "transfer_source_target_ratio_grid": tuple(float(value) for value in args.source_target_ratio_grid),
            "transfer_source_positive_ratio_gap_threshold": float(args.source_positive_ratio_gap_threshold),
            "transfer_gate_repeat_count": int(args.gate_repeat_count),
        }

    return {
        "feature_max_folds": 3,
        "feature_selector_max_features": min(int(spec.selector_max_features), 48),
        "feature_counts": [32, 48],
        "feature_train_sample_cap_per_subject": 64,
        "transfer_max_target_subjects": 3,
        "transfer_max_source_subjects": 4,
        "transfer_source_sample_cap": 64,
        "transfer_mmd_sample_cap": 64,
        "transfer_mmd_prefix_top_k": 0 if int(args.mmd_prefix_top_k) <= 0 else int(args.mmd_prefix_top_k),
        "transfer_target_repeat_grid": (1, 2),
        "transfer_source_target_ratio_grid": tuple(float(value) for value in args.source_target_ratio_grid),
        "transfer_source_positive_ratio_gap_threshold": float(args.source_positive_ratio_gap_threshold),
        "transfer_gate_repeat_count": int(args.gate_repeat_count),
    }


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    """安全读取已有 JSON 结果，便于和本次结果做前后对比。"""
    path = ensure_ai_output_path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Suite][Warning] 读取旧结果失败 path={path} error={exc}")
        return None


def _metric_mean(summary_block: Dict[str, Any], classifier_name: str, metric_name: str) -> float:
    """兼容扁平指标与 mean/std 指标结构，统一抽取均值。"""
    classifier_block = summary_block.get(classifier_name, {})
    metric_value = classifier_block.get(metric_name, 0.0)
    if isinstance(metric_value, dict):
        return float(metric_value.get("mean", 0.0))
    return float(metric_value)


def _best_classifier_from_summary(summary_block: Dict[str, Any]) -> tuple[str, Dict[str, float]]:
    """按 ACC/F1 优先级返回当前汇总结果里最稳健的分类器。"""
    best_name = max(
        CLASSIFIER_ORDER,
        key=lambda name: (
            min(_metric_mean(summary_block, name, "ACC"), _metric_mean(summary_block, name, "F1")),
            _metric_mean(summary_block, name, "ACC"),
            _metric_mean(summary_block, name, "F1"),
            _metric_mean(summary_block, name, "BACC"),
        ),
    )
    return best_name, {
        "ACC": _metric_mean(summary_block, best_name, "ACC"),
        "F1": _metric_mean(summary_block, best_name, "F1"),
        "BACC": _metric_mean(summary_block, best_name, "BACC"),
        "MCC": _metric_mean(summary_block, best_name, "MCC"),
    }


def _summary_rows(summary_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把汇总结果转换成条形图和表格共用的行结构。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        rows.append(
            {
                "Classifier": classifier_name,
                "ACC": _metric_mean(summary_block, classifier_name, "ACC"),
                "F1": _metric_mean(summary_block, classifier_name, "F1"),
                "BACC": _metric_mean(summary_block, classifier_name, "BACC"),
                "MCC": _metric_mean(summary_block, classifier_name, "MCC"),
            }
        )
    return rows


def _summary_table_rows(summary_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把汇总结果格式化成 Markdown 表格需要的字符串。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        rows.append(
            {
                "Classifier": classifier_name,
                "ACC": f"{_metric_mean(summary_block, classifier_name, 'ACC'):.4f}",
                "F1": f"{_metric_mean(summary_block, classifier_name, 'F1'):.4f}",
                "BACC": f"{_metric_mean(summary_block, classifier_name, 'BACC'):.4f}",
                "MCC": f"{_metric_mean(summary_block, classifier_name, 'MCC'):.4f}",
            }
        )
    return rows


def _selected_feature_count_text(selected_feature_count_by_classifier: Dict[str, Any] | None, fallback_count: int) -> str:
    """把分类器专属特征数压缩成便于总报告展示的文本。"""
    if isinstance(selected_feature_count_by_classifier, dict) and selected_feature_count_by_classifier:
        return ", ".join(
            f"{classifier_name}={int(selected_feature_count_by_classifier.get(classifier_name, 0))}"
            for classifier_name in CLASSIFIER_ORDER
        )
    return str(int(fallback_count))


def _delta_rows(previous_summary: Dict[str, Any] | None, current_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """计算当前结果相对上一次同名结果在 ACC/F1 上的变化。"""
    if not previous_summary:
        return []
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        previous_acc = _metric_mean(previous_summary, classifier_name, "ACC")
        current_acc = _metric_mean(current_summary, classifier_name, "ACC")
        previous_f1 = _metric_mean(previous_summary, classifier_name, "F1")
        current_f1 = _metric_mean(current_summary, classifier_name, "F1")
        rows.append(
            {
                "Classifier": classifier_name,
                "Prev_ACC": f"{previous_acc:.4f}",
                "Current_ACC": f"{current_acc:.4f}",
                "Delta_ACC": f"{current_acc - previous_acc:+.4f}",
                "Prev_F1": f"{previous_f1:.4f}",
                "Current_F1": f"{current_f1:.4f}",
                "Delta_F1": f"{current_f1 - previous_f1:+.4f}",
            }
        )
    return rows


def _transfer_decision_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """汇总每个分类器保留迁移和回退到 target-only 的目标单元。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        classifier_rows = [row for row in subject_rows if str(row.get("Classifier")) == classifier_name]
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


def _feature_summary_from_json(json_obj: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """从旧的特征选择 JSON 中抽取可直接对比的冻结掩码汇总。"""
    if not json_obj:
        return None
    for key in ("frozen_mask_baseline_summary", "source_only_transfer_baseline_summary", "feature_selection_cv_summary"):
        summary_block = json_obj.get(key)
        if isinstance(summary_block, dict):
            return summary_block
    return None


def _transfer_summary_from_json(json_obj: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """从旧的迁移学习 JSON 中抽取可直接对比的迁移汇总。"""
    if not json_obj:
        return None
    summary_block = json_obj.get("transfer_summary")
    return summary_block if isinstance(summary_block, dict) else None


def _print_delta_rows(prefix: str, label: str, delta_rows: Sequence[Dict[str, Any]]) -> None:
    """在控制台打印相对上次结果的 ACC/F1 变化。"""
    if not delta_rows:
        print(f"[Suite][{prefix}] {label} 无可对比的旧结果，本次作为新的基线。")
        return
    for row in delta_rows:
        print(
            f"[Suite][{prefix}] {label} classifier={row['Classifier']} "
            f"ACC={row['Prev_ACC']}->{row['Current_ACC']} ({row['Delta_ACC']}) "
            f"F1={row['Prev_F1']}->{row['Current_F1']} ({row['Delta_F1']})"
        )


def _load_completed_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> Dict[str, Any] | None:
    """从已经落盘的单组产物中恢复实验摘要，便于断点续跑后重建总报告。"""
    paths = _artifact_paths(spec, args.mode)
    feature_json = _load_json_if_exists(paths["feature_summary_path"])
    transfer_json = _load_json_if_exists(paths["transfer_summary_path"])
    if not feature_json or not transfer_json:
        return None

    feature_summary = _feature_summary_from_json(feature_json)
    transfer_summary = _transfer_summary_from_json(transfer_json)
    if not feature_summary or not transfer_summary:
        return None

    feature_best_classifier, feature_best_metrics = _best_classifier_from_summary(feature_summary)
    transfer_best_classifier, transfer_best_metrics = _best_classifier_from_summary(transfer_summary)
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "selector_name": spec.selector_name,
        "feature_protocol": spec.feature_protocol,
        "transfer_protocol": spec.transfer_protocol,
        "selected_feature_count": int(feature_json.get("selected_feature_count", 0)),
        "selected_feature_count_by_classifier": feature_json.get("selected_feature_count_by_classifier", {}),
        "feature_best_classifier": feature_best_classifier,
        "feature_best_metrics": feature_best_metrics,
        "feature_summary": feature_summary,
        "feature_delta_rows": [],
        "feature_summary_path": str(ensure_ai_output_path(paths["feature_summary_path"])),
        "feature_report_path": str(ensure_ai_output_path(paths["feature_report_path"])),
        "transfer_best_classifier": transfer_best_classifier,
        "transfer_best_metrics": transfer_best_metrics,
        "transfer_summary": transfer_summary,
        "transfer_delta_rows": [],
        "transfer_decision_rows": list(transfer_json.get("transfer_decision_rows", [])),
        "transfer_summary_path": str(ensure_ai_output_path(paths["transfer_summary_path"])),
        "transfer_report_path": str(ensure_ai_output_path(paths["transfer_report_path"])),
    }


def _write_suite_report(report_path: Path, args: argparse.Namespace, experiments: Sequence[Dict[str, Any]]) -> Path:
    """生成四组联合实验的总 Markdown 报告。"""
    intro_lines = [
        f"生成时间：`{_timestamp_text()}`",
        "本报告统一汇总四组联合实验：MIIFS/mRMR 各自的 10 折分组交叉验证与留一被试验证，并在同口径下继续运行监督迁移学习。",
        "最终只比较 `ACC` 与 `F1`，同时明确列出哪些目标单元真正保留迁移、哪些目标单元被回退到 `target-only`。",
    ]
    sections: List[Dict[str, Any]] = [
        {
            "title": "统一配置",
            "body_lines": [
                f"- run_mode: `{args.mode}`",
                f"- variance_threshold: `{args.variance_threshold}`",
                f"- selector_bins: `{args.selector_bins}`",
                f"- groupkfold_feature_counts: `{[int(value) for value in args.groupkfold_feature_counts]}`",
                f"- loso_feature_counts: `{[int(value) for value in args.loso_feature_counts]}`",
                f"- target_split: `{args.target_train_ratio:.2f}/{args.target_val_ratio:.2f}/{args.target_test_ratio:.2f}`",
                f"- max_source_subjects: `{args.max_source_subjects}`",
                f"- source_sample_cap: `{args.source_sample_cap}`",
                f"- mmd_sample_cap: `{args.mmd_sample_cap}`",
                f"- mmd_prefix_search: `{_mmd_prefix_search_text(args.mmd_prefix_top_k)}`",
                f"- target_repeat_grid: `{[int(value) for value in args.target_repeat_grid]}`",
                f"- source_target_ratio_grid: `{[float(value) for value in args.source_target_ratio_grid]}`",
                f"- source_positive_ratio_gap_threshold: `{float(args.source_positive_ratio_gap_threshold):.4f}`",
                f"- gate_repeat_count: `{int(args.gate_repeat_count)}`",
                f"- classifiers: `{list(CLASSIFIER_ORDER)}`",
            ],
        }
    ]

    for experiment in experiments:
        feature_rows = _summary_rows(experiment["feature_summary"])
        transfer_rows = _summary_rows(experiment["transfer_summary"])
        body_lines: List[str] = [
            f"- selector: `{experiment['selector_name']}`",
            f"- feature_protocol: `{experiment['feature_protocol']}`",
            f"- transfer_protocol: `{experiment['transfer_protocol']}`",
            f"- classifier_mask_feature_count: `{_selected_feature_count_text(experiment.get('selected_feature_count_by_classifier'), experiment['selected_feature_count'])}`",
            f"- 阶段一最佳分类器: `{experiment['feature_best_classifier']}`，ACC=`{experiment['feature_best_metrics']['ACC']:.4f}`，F1=`{experiment['feature_best_metrics']['F1']:.4f}`",
            f"- 迁移阶段最佳分类器: `{experiment['transfer_best_classifier']}`，ACC=`{experiment['transfer_best_metrics']['ACC']:.4f}`，F1=`{experiment['transfer_best_metrics']['F1']:.4f}`",
            f"- 特征选择摘要: `{experiment['feature_summary_path']}`",
            f"- 特征选择报告: `{experiment['feature_report_path']}`",
            f"- 迁移学习摘要: `{experiment['transfer_summary_path']}`",
            f"- 迁移学习报告: `{experiment['transfer_report_path']}`",
            "",
        ]
        body_lines.extend(format_metric_bars("阶段一冻结掩码 ACC", feature_rows, "ACC"))
        body_lines.append("")
        body_lines.extend(format_metric_bars("阶段一冻结掩码 F1", feature_rows, "F1"))
        body_lines.append("")
        body_lines.extend(format_metric_table(_summary_table_rows(experiment["feature_summary"]), ["Classifier", "ACC", "F1", "BACC", "MCC"]))

        if experiment["feature_delta_rows"]:
            body_lines.extend(["", "### 与上次阶段一结果对比", ""])
            body_lines.extend(
                format_metric_table(
                    experiment["feature_delta_rows"],
                    ["Classifier", "Prev_ACC", "Current_ACC", "Delta_ACC", "Prev_F1", "Current_F1", "Delta_F1"],
                )
            )

        body_lines.extend(["", "### 监督迁移结果", ""])
        body_lines.extend(format_metric_bars("迁移后 ACC", transfer_rows, "ACC"))
        body_lines.append("")
        body_lines.extend(format_metric_bars("迁移后 F1", transfer_rows, "F1"))
        body_lines.append("")
        body_lines.extend(format_metric_table(_summary_table_rows(experiment["transfer_summary"]), ["Classifier", "ACC", "F1", "BACC", "MCC"]))

        if experiment["transfer_delta_rows"]:
            body_lines.extend(["", "### 与上次迁移结果对比", ""])
            body_lines.extend(
                format_metric_table(
                    experiment["transfer_delta_rows"],
                    ["Classifier", "Prev_ACC", "Current_ACC", "Delta_ACC", "Prev_F1", "Current_F1", "Delta_F1"],
                )
            )

        body_lines.extend(["", "### 迁移/回退明细", ""])
        body_lines.extend(
            format_metric_table(
                experiment["transfer_decision_rows"],
                ["Classifier", "TransferCount", "FallbackCount", "TransferUnits", "FallbackUnits", "FallbackReasons"],
            )
        )
        sections.append({"title": experiment["display_name"], "body_lines": body_lines})

    return write_markdown_report(
        path=report_path,
        title="四组联合实验总报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def _run_single_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> Dict[str, Any]:
    """执行单组“特征选择 + 迁移学习”实验，并返回汇总信息。"""
    runtime_override = _runtime_overrides(spec, args)
    paths = _artifact_paths(spec, args.mode)
    previous_feature_json = _load_json_if_exists(paths["feature_summary_path"])
    previous_transfer_json = _load_json_if_exists(paths["transfer_summary_path"])

    feature_config = FeatureSelectionConfig(
        evaluation_protocol=spec.feature_protocol,
        selector_name=spec.selector_name,
        n_splits=int(spec.n_splits),
        max_folds=runtime_override["feature_max_folds"],
        random_state=int(args.seed),
        variance_threshold=float(args.variance_threshold),
        selector_max_features=int(runtime_override["feature_selector_max_features"]),
        selector_bins=int(args.selector_bins),
        evaluation_feature_counts=[int(value) for value in runtime_override["feature_counts"]],
        mask_feature_count=None,
        min_accept_acc=float(args.min_accept_acc),
        train_sample_cap_per_subject=runtime_override["feature_train_sample_cap_per_subject"],
        artifact_path=paths["feature_artifact_path"],
        summary_path=paths["feature_summary_path"],
        mask_path=paths["feature_mask_path"],
        report_path=paths["feature_report_path"],
    )
    print("\n" + "=" * 100)
    print(
        f"[Suite][FeatureSelection] start experiment={spec.display_name} mode={args.mode} "
        f"selector_max_features={feature_config.selector_max_features} feature_counts={feature_config.evaluation_feature_counts} "
        f"max_folds={feature_config.max_folds if feature_config.max_folds is not None else 'all'} "
        f"train_sample_cap_per_subject={feature_config.train_sample_cap_per_subject if feature_config.train_sample_cap_per_subject is not None else 'full'}"
    )
    feature_result = run_feature_selection_pipeline(feature_config)
    feature_summary = feature_result["frozen_mask_baseline_summary"]
    feature_best_classifier, feature_best_metrics = _best_classifier_from_summary(feature_summary)
    feature_delta_rows = _delta_rows(_feature_summary_from_json(previous_feature_json), feature_summary)
    print(
        f"[Suite][FeatureSelection] finish experiment={spec.display_name} "
        f"selected_feature_count={feature_result['selected_feature_count']} "
        f"best={feature_best_classifier} ACC={feature_best_metrics['ACC']:.4f} F1={feature_best_metrics['F1']:.4f}"
    )
    _print_delta_rows("FeatureDelta", spec.display_name, feature_delta_rows)

    transfer_config = TransferLearningConfig(
        mask_path=paths["feature_mask_path"],
        feature_source_mode="mask",
        feature_source_label=spec.display_name,
        summary_path=paths["transfer_summary_path"],
        report_path=paths["transfer_report_path"],
        random_state=int(args.seed),
        evaluation_protocol=spec.transfer_protocol,
        cv_splits=int(spec.n_splits),
        max_target_subjects=runtime_override["transfer_max_target_subjects"],
        target_train_ratio=float(args.target_train_ratio),
        target_val_ratio=float(args.target_val_ratio),
        target_test_ratio=float(args.target_test_ratio),
        max_source_subjects=int(runtime_override["transfer_max_source_subjects"]),
        source_sample_cap=int(runtime_override["transfer_source_sample_cap"]),
        mmd_sample_cap=int(runtime_override["transfer_mmd_sample_cap"]),
        mmd_prefix_top_k=int(runtime_override["transfer_mmd_prefix_top_k"]),
        target_repeat_grid=tuple(int(value) for value in runtime_override["transfer_target_repeat_grid"]),
        source_target_ratio_grid=tuple(float(value) for value in runtime_override["transfer_source_target_ratio_grid"]),
        source_positive_ratio_gap_threshold=float(runtime_override["transfer_source_positive_ratio_gap_threshold"]),
        gate_repeat_count=int(runtime_override["transfer_gate_repeat_count"]),
        reference_summary_path=None,
    )
    print(
        f"[Suite][Transfer] start experiment={spec.display_name} mode={args.mode} "
        f"protocol={transfer_config.evaluation_protocol} "
        f"max_target_subjects={transfer_config.max_target_subjects if transfer_config.max_target_subjects is not None else 'all'} "
        f"max_source_subjects={transfer_config.max_source_subjects} "
        f"source_sample_cap={transfer_config.source_sample_cap} mmd_sample_cap={transfer_config.mmd_sample_cap} "
        f"target_repeat_grid={list(transfer_config.target_repeat_grid)} "
        f"source_target_ratio_grid={list(transfer_config.source_target_ratio_grid)} "
        f"source_positive_ratio_gap_threshold={transfer_config.source_positive_ratio_gap_threshold:.4f} "
        f"gate_repeat_count={transfer_config.gate_repeat_count}"
    )
    transfer_result = run_transfer_learning_pipeline(transfer_config)
    transfer_summary = transfer_result["transfer_summary"]
    transfer_best_classifier, transfer_best_metrics = _best_classifier_from_summary(transfer_summary)
    transfer_delta_rows = _delta_rows(_transfer_summary_from_json(previous_transfer_json), transfer_summary)
    transfer_decision_rows = _transfer_decision_rows(transfer_result["subject_rows"])
    print(
        f"[Suite][Transfer] finish experiment={spec.display_name} "
        f"best={transfer_best_classifier} ACC={transfer_best_metrics['ACC']:.4f} F1={transfer_best_metrics['F1']:.4f}"
    )
    _print_delta_rows("TransferDelta", spec.display_name, transfer_delta_rows)
    for row in transfer_decision_rows:
        print(
            f"[Suite][TransferDecision] experiment={spec.display_name} classifier={row['Classifier']} "
            f"transfer_count={row['TransferCount']} fallback_count={row['FallbackCount']} "
            f"fallback_reasons={row['FallbackReasons']}"
        )

    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "selector_name": spec.selector_name,
        "feature_protocol": spec.feature_protocol,
        "transfer_protocol": spec.transfer_protocol,
        "selected_feature_count": int(feature_result["selected_feature_count"]),
        "selected_feature_count_by_classifier": feature_result.get("selected_feature_count_by_classifier", {}),
        "feature_best_classifier": feature_best_classifier,
        "feature_best_metrics": feature_best_metrics,
        "feature_summary": feature_summary,
        "feature_delta_rows": feature_delta_rows,
        "feature_summary_path": str(ensure_ai_output_path(paths["feature_summary_path"])),
        "feature_report_path": str(ensure_ai_output_path(paths["feature_report_path"])),
        "transfer_best_classifier": transfer_best_classifier,
        "transfer_best_metrics": transfer_best_metrics,
        "transfer_summary": transfer_summary,
        "transfer_delta_rows": transfer_delta_rows,
        "transfer_decision_rows": transfer_decision_rows,
        "transfer_summary_path": str(ensure_ai_output_path(paths["transfer_summary_path"])),
        "transfer_report_path": str(ensure_ai_output_path(paths["transfer_report_path"])),
    }


def main() -> None:
    """顺序执行四组联合实验，并生成总 JSON/Markdown 报告。"""
    args = build_parser().parse_args()
    ensure_artifact_dir()
    suite_summary_path = ensure_ai_output_path(args.suite_summary_path)
    suite_report_path = ensure_ai_output_path(args.suite_report_path)

    specs = _build_specs(args)
    valid_keys = {spec.key for spec in specs}
    selected_keys = set(valid_keys) if not args.only_experiment else {str(key) for key in args.only_experiment}
    unknown_keys = sorted(selected_keys - valid_keys)
    if unknown_keys:
        raise ValueError(f"only_experiment 包含未知实验键: {unknown_keys}")

    experiment_rows: List[Dict[str, Any]] = []
    for spec in specs:
        if spec.key in selected_keys:
            experiment_rows.append(_run_single_experiment(spec, args))
            continue
        recovered_row = _load_completed_experiment(spec, args)
        if recovered_row is None:
            print(f"[Suite][Recover] skip experiment={spec.display_name}，未找到可复用的已完成结果。")
            continue
        print(f"[Suite][Recover] reuse experiment={spec.display_name} from existing artifacts.")
        experiment_rows.append(recovered_row)

    suite_report_path = _write_suite_report(suite_report_path, args, experiment_rows)
    summary = {
        "generated_at": _timestamp_text(),
        "mode": str(args.mode),
        "classifiers": list(CLASSIFIER_ORDER),
        "feature_config": {
            "variance_threshold": float(args.variance_threshold),
            "selector_bins": int(args.selector_bins),
            "groupkfold_selector_max_features": int(args.groupkfold_selector_max_features),
            "groupkfold_feature_counts": [int(value) for value in args.groupkfold_feature_counts],
            "groupkfold_train_sample_cap_per_subject": int(args.groupkfold_train_sample_cap_per_subject),
            "loso_selector_max_features": int(args.loso_selector_max_features),
            "loso_feature_counts": [int(value) for value in args.loso_feature_counts],
            "loso_train_sample_cap_per_subject": int(args.loso_train_sample_cap_per_subject),
        },
        "transfer_config": {
            "target_train_ratio": float(args.target_train_ratio),
            "target_val_ratio": float(args.target_val_ratio),
            "target_test_ratio": float(args.target_test_ratio),
            "max_source_subjects": int(args.max_source_subjects),
            "source_sample_cap": int(args.source_sample_cap),
            "mmd_sample_cap": int(args.mmd_sample_cap),
            "mmd_prefix_top_k": int(args.mmd_prefix_top_k),
            "target_repeat_grid": [int(value) for value in args.target_repeat_grid],
            "source_target_ratio_grid": [float(value) for value in args.source_target_ratio_grid],
        },
        "experiments": experiment_rows,
        "suite_report_path": str(suite_report_path),
    }
    save_json(to_serializable(summary), suite_summary_path)
    print(f"[Suite] summary_path={suite_summary_path}")
    print(f"[Suite] report_path={suite_report_path}")


if __name__ == "__main__":
    main()
