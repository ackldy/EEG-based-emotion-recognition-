from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.common import ensure_ai_output_path, ensure_artifact_dir, save_json, to_serializable
from AI.feature_selection_pipeline import FeatureSelectionConfig, run_feature_selection_pipeline
from AI.reporting import format_metric_bars, format_metric_table, write_markdown_report
from AI.transfer_learning_pipeline import (
    CLASSIFIER_ORDER,
    TransferLearningConfig,
    run_transfer_learning_pipeline,
)


EXPERIMENT_NAME = "MIIFS_CANDD_STABLE"
ARTIFACT_BASE_DIR = Path("AI/artifacts/miifs_candd_stable")
FEATURE_SUMMARY_PATH = ARTIFACT_BASE_DIR / "feature_selection_summary_miifs_candd_stable.json"
FEATURE_REPORT_PATH = ARTIFACT_BASE_DIR / "feature_selection_report_miifs_candd_stable.md"
FEATURE_MASK_PATH = ARTIFACT_BASE_DIR / "feature_mask_miifs_candd_stable.pkl"
FEATURE_ARTIFACT_PATH = ARTIFACT_BASE_DIR / "feature_selection_cv_miifs_candd_stable.pkl"
TRANSFER_SUMMARY_PATH = ARTIFACT_BASE_DIR / "transfer_learning_summary_miifs_candd_stable.json"
TRANSFER_REPORT_PATH = ARTIFACT_BASE_DIR / "transfer_learning_report_miifs_candd_stable.md"
EXPERIMENT_SUMMARY_PATH = ARTIFACT_BASE_DIR / "experiment_summary_miifs_candd_stable.json"
EXPERIMENT_REPORT_PATH = ARTIFACT_BASE_DIR / "experiment_report_miifs_candd_stable.md"
DEBUG_MD_PATH = ARTIFACT_BASE_DIR / "debug_miifs_candd_stable.md"
REFERENCE_SUMMARY_PATH = Path("AI/artifacts/transfer_learning_summary_miifs_loso_k200_candD_full32.json")
LEGACY_FEATURE_ARTIFACT_PATH = Path(
    "AI/artifacts/miifs_candd_refined/feature_selection_cv_miifs_candd_refined_fs_vt1e3_b48_m224_candf_balanced_256_full32_fs_vt1e3_b48_m224.pkl"
)
LEGACY_FEATURE_SUMMARY_PATH = Path(
    "AI/artifacts/miifs_candd_refined/feature_selection_summary_miifs_candd_refined_fs_vt1e3_b48_m224_candf_balanced_256_full32_fs_vt1e3_b48_m224.json"
)
LEGACY_FEATURE_MASK_PATH = Path(
    "AI/artifacts/miifs_candd_refined/feature_mask_miifs_candd_refined_fs_vt1e3_b48_m224_candf_balanced_256_full32_fs_vt1e3_b48_m224.pkl"
)
OBSOLETE_REPORT_PATHS: Tuple[Path, ...] = (
    Path("AI/artifacts/miifs_candd_refined/miifs_candd_refined_summary.json"),
    Path("AI/artifacts/miifs_candd_refined/miifs_candd_refined_summary.md"),
)


@dataclass(frozen=True)
class StableFeatureSpec:
    """定义 MIIFS_CANDD_STABLE 阶段一固定使用的特征选择口径。"""

    variance_threshold: float = 1e-3
    selector_bins: int = 48
    selector_max_features: int = 224
    feature_grid: Tuple[int, ...] = (192, 208, 224)
    mask_feature_count: int = 224
    min_accept_acc: float = 0.60


@dataclass(frozen=True)
class StableTransferSpec:
    """定义 MIIFS_CANDD_STABLE 阶段二固定使用的监督迁移口径。"""

    max_source_subjects: int = 12
    source_sample_cap: int = 200
    mmd_sample_cap: int = 200
    mmd_prefix_top_k: int = 8
    target_repeat_grid: Tuple[int, ...] = (2, 4)
    source_target_ratio_grid: Tuple[float, ...] = (0.25, 0.5)
    source_positive_ratio_gap_threshold: float = 0.08
    gate_repeat_count: int = 3


FEATURE_SPEC = StableFeatureSpec()
TRANSFER_SPEC = StableTransferSpec()


class TeeTextIO(io.TextIOBase):
    """把控制台输出同时写到终端和内存缓冲区，便于落盘调试日志。"""

    def __init__(self, primary_stream: Any, mirror_stream: io.StringIO) -> None:
        """记录真实终端流与镜像缓冲区。"""
        self.primary_stream = primary_stream
        self.mirror_stream = mirror_stream

    def write(self, text: str) -> int:
        """同步写入真实终端和内存缓冲区。"""
        self.primary_stream.write(text)
        self.mirror_stream.write(text)
        return len(text)

    def flush(self) -> None:
        """同步刷新真实终端和镜像缓冲区。"""
        self.primary_stream.flush()
        self.mirror_stream.flush()

    def isatty(self) -> bool:
        """尽量保留原始终端流的交互属性。"""
        return bool(getattr(self.primary_stream, "isatty", lambda: False)())


def build_parser() -> argparse.ArgumentParser:
    """构建 MIIFS_CANDD_STABLE 实验入口参数。"""
    parser = argparse.ArgumentParser(description="运行 MIIFS_CANDD_STABLE，并生成中文 JSON/MD 报告与调试日志。")
    parser.add_argument("--seed", type=int, default=42, help="统一随机种子。")
    parser.add_argument("--target-train-ratio", type=float, default=0.60, help="目标域训练集比例。")
    parser.add_argument("--target-val-ratio", type=float, default=0.20, help="目标域验证集比例。")
    parser.add_argument("--target-test-ratio", type=float, default=0.20, help="目标域测试集比例。")
    parser.add_argument("--feature-n-splits", type=int, default=32, help="阶段一 LOSO 折数。")
    parser.add_argument("--summary-path", type=Path, default=EXPERIMENT_SUMMARY_PATH, help="实验总摘要 JSON 输出路径。")
    parser.add_argument("--report-path", type=Path, default=EXPERIMENT_REPORT_PATH, help="实验总报告 MD 输出路径。")
    parser.add_argument("--debug-path", type=Path, default=DEBUG_MD_PATH, help="完整调试日志 MD 输出路径。")
    parser.add_argument("--reference-summary-path", type=Path, default=REFERENCE_SUMMARY_PATH, help="上一轮 candD 结果 JSON，用于做对照。")
    parser.add_argument("--skip-cleanup", action="store_true", help="跳过删除旧的 refined 总结报告。")
    return parser


def _timestamp_text() -> str:
    """返回当前本地时间字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _json_block_lines(data: Any) -> List[str]:
    """把结构化对象渲染成 Markdown 可读的 JSON 代码块。"""
    return ["```json", json.dumps(to_serializable(data), ensure_ascii=False, indent=2), "```"]


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    """如果 JSON 产物已经存在就读取返回，否则返回 None。"""
    resolved_path = ensure_ai_output_path(path)
    if not resolved_path.exists():
        return None
    try:
        return json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _artifact_file_name(path: Path | str) -> str:
    """提取产物文件名，便于在报告中快速和 JSON/MD 对照。"""
    return Path(path).name


def _metric_mean(summary_block: Dict[str, Any], classifier_name: str, metric_name: str) -> float:
    """统一读取单个分类器汇总结果里的指标均值。"""
    metric_value = summary_block[classifier_name][metric_name]
    if isinstance(metric_value, dict):
        return float(metric_value.get("mean", 0.0))
    return float(metric_value)


def _summary_rows(summary_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把分类器汇总结果整理成条形图与表格复用的行结构。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        if classifier_name not in summary_block:
            continue
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
    """把分类器汇总结果整理成 Markdown 表格需要的字符串行。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in CLASSIFIER_ORDER:
        if classifier_name not in summary_block:
            continue
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


def _best_classifier_from_summary(summary_block: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    """按 ACC/F1 优先级返回当前结果里最稳的分类器。"""
    available = [name for name in CLASSIFIER_ORDER if name in summary_block]
    best_name = max(
        available,
        key=lambda name: (
            min(_metric_mean(summary_block, name, "ACC"), _metric_mean(summary_block, name, "F1")),
            _metric_mean(summary_block, name, "ACC"),
            _metric_mean(summary_block, name, "F1"),
            _metric_mean(summary_block, name, "BACC"),
            _metric_mean(summary_block, name, "MCC"),
        ),
    )
    return best_name, {
        "ACC": _metric_mean(summary_block, best_name, "ACC"),
        "F1": _metric_mean(summary_block, best_name, "F1"),
        "BACC": _metric_mean(summary_block, best_name, "BACC"),
        "MCC": _metric_mean(summary_block, best_name, "MCC"),
    }


def _total_transfer_fallback_counts(transfer_decision_rows: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    """统计本次迁移阶段总共保留迁移和回退了多少次。"""
    transfer_count = int(sum(int(row.get("TransferCount", 0)) for row in transfer_decision_rows))
    fallback_count = int(sum(int(row.get("FallbackCount", 0)) for row in transfer_decision_rows))
    return transfer_count, fallback_count


@contextmanager
def _capture_output() -> Generator[io.StringIO, None, None]:
    """在保留终端输出的同时，把 stdout/stderr 镜像到内存缓冲区。"""
    buffer = io.StringIO()
    tee_stdout = TeeTextIO(sys.stdout, buffer)
    tee_stderr = TeeTextIO(sys.stderr, buffer)
    with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
        yield buffer


def _build_feature_config(args: argparse.Namespace) -> FeatureSelectionConfig:
    """构造 MIIFS_CANDD_STABLE 阶段一固定使用的特征选择配置。"""
    return FeatureSelectionConfig(
        evaluation_protocol="loso_subject",
        selector_name="miifs",
        n_splits=int(args.feature_n_splits),
        max_folds=None,
        random_state=int(args.seed),
        variance_threshold=float(FEATURE_SPEC.variance_threshold),
        selector_max_features=int(FEATURE_SPEC.selector_max_features),
        selector_bins=int(FEATURE_SPEC.selector_bins),
        evaluation_feature_counts=[int(value) for value in FEATURE_SPEC.feature_grid],
        mask_feature_count=int(FEATURE_SPEC.mask_feature_count),
        min_accept_acc=float(FEATURE_SPEC.min_accept_acc),
        train_sample_cap_per_subject=None,
        artifact_path=FEATURE_ARTIFACT_PATH,
        summary_path=FEATURE_SUMMARY_PATH,
        mask_path=FEATURE_MASK_PATH,
        report_path=FEATURE_REPORT_PATH,
    )


def _build_transfer_config(args: argparse.Namespace) -> TransferLearningConfig:
    """构造 MIIFS_CANDD_STABLE 阶段二固定使用的监督迁移配置。"""
    return TransferLearningConfig(
        mask_path=FEATURE_MASK_PATH,
        feature_source_mode="mask",
        feature_source_label=EXPERIMENT_NAME,
        summary_path=TRANSFER_SUMMARY_PATH,
        report_path=TRANSFER_REPORT_PATH,
        random_state=int(args.seed),
        evaluation_protocol="loso",
        cv_splits=32,
        max_target_subjects=None,
        target_train_ratio=float(args.target_train_ratio),
        target_val_ratio=float(args.target_val_ratio),
        target_test_ratio=float(args.target_test_ratio),
        max_source_subjects=int(TRANSFER_SPEC.max_source_subjects),
        source_sample_cap=int(TRANSFER_SPEC.source_sample_cap),
        mmd_sample_cap=int(TRANSFER_SPEC.mmd_sample_cap),
        mmd_prefix_top_k=int(TRANSFER_SPEC.mmd_prefix_top_k),
        transfer_variant="supervised",
        target_repeat_grid=tuple(int(value) for value in TRANSFER_SPEC.target_repeat_grid),
        source_target_ratio_grid=tuple(float(value) for value in TRANSFER_SPEC.source_target_ratio_grid),
        source_positive_ratio_gap_threshold=float(TRANSFER_SPEC.source_positive_ratio_gap_threshold),
        gate_repeat_count=int(TRANSFER_SPEC.gate_repeat_count),
        reference_summary_path=args.reference_summary_path,
    )


def _feature_param_rows(feature_config: Dict[str, Any]) -> List[Dict[str, str]]:
    """整理阶段一关键参数及其中文作用，写入最终实验总报告。"""
    return [
        {
            "Parameter": "variance_threshold",
            "Value": f"{float(feature_config['variance_threshold']):.6f}",
            "Meaning": "先删除低方差特征，减少几乎不变的噪声维度。",
        },
        {
            "Parameter": "selector_bins",
            "Value": str(int(feature_config["selector_bins"])),
            "Meaning": "MIIFS 互信息估计使用的分箱数，影响互信息离散化精度。",
        },
        {
            "Parameter": "selector_max_features",
            "Value": str(int(feature_config["selector_max_features"])),
            "Meaning": "贪心搜索阶段允许探索到的最大特征数上限。",
        },
        {
            "Parameter": "evaluation_feature_counts",
            "Value": str(feature_config["evaluation_feature_counts"]),
            "Meaning": "阶段一会对比的候选特征数网格。",
        },
        {
            "Parameter": "mask_feature_count",
            "Value": str(int(feature_config["mask_feature_count"])),
            "Meaning": "最终冻结给迁移学习直接读取的特征数。",
        },
        {
            "Parameter": "n_splits",
            "Value": str(int(feature_config["n_splits"])),
            "Meaning": "阶段一 LOSO 折数；32 表示完整覆盖全部被试。",
        },
        {
            "Parameter": "min_accept_acc",
            "Value": f"{float(feature_config['min_accept_acc']):.4f}",
            "Meaning": "冻结掩码回放 ACC 的最低参考线，低于该值说明阶段一不够稳。",
        },
    ]


def _transfer_param_rows(transfer_config: Dict[str, Any]) -> List[Dict[str, str]]:
    """整理阶段二关键参数及其中文作用，写入最终实验总报告。"""
    return [
        {
            "Parameter": "target_split",
            "Value": f"{float(transfer_config['target_train_ratio']):.2f}/{float(transfer_config['target_val_ratio']):.2f}/{float(transfer_config['target_test_ratio']):.2f}",
            "Meaning": "目标域 train/val/test 固定切分比例，用来同时输出验证集与测试集结果。",
        },
        {
            "Parameter": "max_source_subjects",
            "Value": str(int(transfer_config["max_source_subjects"])),
            "Meaning": "每个目标被试最多引入多少个源被试参与迁移训练。",
        },
        {
            "Parameter": "source_sample_cap",
            "Value": str(int(transfer_config["source_sample_cap"])),
            "Meaning": "每个源被试最多抽样多少条样本进入迁移训练，平衡速度与稳定性。",
        },
        {
            "Parameter": "mmd_sample_cap",
            "Value": str(int(transfer_config["mmd_sample_cap"])),
            "Meaning": "估计跨域差异 MMD 时每个域最多使用多少条样本。",
        },
        {
            "Parameter": "mmd_prefix_top_k",
            "Value": str(int(transfer_config["mmd_prefix_top_k"])),
            "Meaning": "累计前缀 MMD 搜索上限；本稳定口径固定只保留前 8 个候选前缀。",
        },
        {
            "Parameter": "target_repeat_grid",
            "Value": str(list(transfer_config["target_repeat_grid"])),
            "Meaning": "目标域样本在混合训练中的重复倍数候选网格。",
        },
        {
            "Parameter": "source_target_ratio_grid",
            "Value": str(list(transfer_config["source_target_ratio_grid"])),
            "Meaning": "搜索源域有效权重与目标域有效权重比例的候选网格。",
        },
        {
            "Parameter": "source_positive_ratio_gap_threshold",
            "Value": f"{float(transfer_config['source_positive_ratio_gap_threshold']):.4f}",
            "Meaning": "源被试与目标训练集正类比例允许的最大差距，超出则先过滤。",
        },
        {
            "Parameter": "gate_repeat_count",
            "Value": str(int(transfer_config["gate_repeat_count"])),
            "Meaning": "主验证切分没通过 gate 时，额外复核的次数，用来降低误杀概率。",
        },
        {
            "Parameter": "transfer_variant",
            "Value": str(transfer_config["transfer_variant"]),
            "Meaning": "本次固定使用监督迁移，不启用 JDA 回路。",
        },
    ]


def _write_reused_feature_report(feature_result: Dict[str, Any], feature_config: FeatureSelectionConfig) -> Path:
    """用已存在的阶段一摘要重新生成 stable 版特征选择报告。"""
    pre_summary = dict(feature_result.get("pre_feature_baseline_summary", {}))
    selected_summary = dict(feature_result.get("feature_selection_cv_summary", {}))
    frozen_summary = dict(feature_result.get("frozen_mask_baseline_summary", {}))
    pre_rows = _summary_rows(pre_summary) if pre_summary else []
    selected_rows = _summary_rows(selected_summary) if selected_summary else []
    frozen_rows = _summary_rows(frozen_summary)
    feature_config_dict = to_serializable(asdict(feature_config))

    sections: List[Dict[str, Any]] = [
        {
            "title": "复用说明",
            "body_lines": [
                f"- 当前 stable 阶段一直接复用同口径历史结果: `{LEGACY_FEATURE_SUMMARY_PATH}`",
                "- 复用原因：阶段一口径完全一致，继续重跑只会额外消耗大量时间。",
                f"- 对应 JSON 摘要文件: `{_artifact_file_name(FEATURE_SUMMARY_PATH)}`",
                f"- 对应 Markdown 报告文件: `{_artifact_file_name(FEATURE_REPORT_PATH)}`",
            ],
        },
        {
            "title": "参数注释",
            "table": format_metric_table(
                _feature_param_rows(feature_config_dict),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "阶段一完整超参数",
            "body_lines": _json_block_lines(feature_config_dict),
        },
    ]

    if feature_result.get("classifier_mask_rows"):
        sections.append(
            {
                "title": "分类器专属 Mask",
                "table": format_metric_table(
                    feature_result["classifier_mask_rows"],
                    ["Classifier", "FeatureCount", "CV_ACC", "CV_F1", "Replay_ACC", "Replay_F1"],
                ),
            }
        )

    if pre_rows:
        sections.append(
            {
                "title": "特征选择前基线",
                "body_lines": format_metric_bars("未筛选特征 ACC", pre_rows, "ACC")
                + [""]
                + format_metric_bars("未筛选特征 F1", pre_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(pre_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            }
        )

    if selected_rows:
        sections.append(
            {
                "title": "折内候选最优结果",
                "body_lines": format_metric_bars("折内候选最优 ACC", selected_rows, "ACC")
                + [""]
                + format_metric_bars("折内候选最优 F1", selected_rows, "F1"),
                "table": format_metric_table(_summary_table_rows(selected_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
            }
        )

    sections.append(
        {
            "title": "冻结掩码回放基线",
            "body_lines": format_metric_bars("冻结 mask ACC", frozen_rows, "ACC")
            + [""]
            + format_metric_bars("冻结 mask F1", frozen_rows, "F1"),
            "table": format_metric_table(_summary_table_rows(frozen_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
        }
    )

    if feature_result.get("fold_overview_rows"):
        sections.append(
            {
                "title": "折概览",
                "table": format_metric_table(
                    feature_result["fold_overview_rows"],
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
            }
        )

    return write_markdown_report(
        path=ensure_ai_output_path(FEATURE_REPORT_PATH),
        title=f"{EXPERIMENT_NAME} 阶段一报告",
        intro_lines=[
            f"本报告对应 `{EXPERIMENT_NAME}` 的阶段一特征选择结果。",
            "该报告由同口径历史摘要重渲染而成，但输出文件已经切换为 stable 命名。",
        ],
        sections=sections,
    )


def _try_reuse_legacy_feature_stage(feature_config: FeatureSelectionConfig) -> Dict[str, Any] | None:
    """尝试复用同口径历史阶段一结果，并重写成 stable 命名。"""
    legacy_summary = _load_json_if_exists(LEGACY_FEATURE_SUMMARY_PATH)
    if not isinstance(legacy_summary, dict) or not legacy_summary.get("frozen_mask_baseline_summary"):
        return None

    legacy_config = dict(legacy_summary.get("config", {}))
    expected_config = to_serializable(asdict(feature_config))
    for key in (
        "evaluation_protocol",
        "selector_name",
        "n_splits",
        "max_folds",
        "random_state",
        "variance_threshold",
        "selector_max_features",
        "selector_bins",
        "evaluation_feature_counts",
        "mask_feature_count",
        "min_accept_acc",
        "train_sample_cap_per_subject",
    ):
        if legacy_config.get(key) != expected_config.get(key):
            return None

    for source_path, target_path in (
        (LEGACY_FEATURE_ARTIFACT_PATH, FEATURE_ARTIFACT_PATH),
        (LEGACY_FEATURE_MASK_PATH, FEATURE_MASK_PATH),
    ):
        source_resolved = ensure_ai_output_path(source_path)
        target_resolved = ensure_ai_output_path(target_path)
        if not source_resolved.exists():
            return None
        target_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_resolved, target_resolved)

    reused_summary = dict(legacy_summary)
    reused_summary["config"] = expected_config
    reused_summary["report_path"] = str(ensure_ai_output_path(FEATURE_REPORT_PATH))
    save_json(to_serializable(reused_summary), ensure_ai_output_path(FEATURE_SUMMARY_PATH))
    _write_reused_feature_report(reused_summary, feature_config)
    print(
        f"[{EXPERIMENT_NAME}][Feature][ReuseLegacy] "
        f"legacy_summary={LEGACY_FEATURE_SUMMARY_PATH} "
        f"stable_summary={FEATURE_SUMMARY_PATH}"
    )
    return reused_summary


def _diagnostic_lines(transfer_result: Dict[str, Any]) -> List[str]:
    """把迁移结果整理成便于快速阅读的问题诊断结论。"""
    lines: List[str] = []
    for row in transfer_result.get("transfer_decision_rows", []):
        lines.append(
            f"- {row['Classifier']}: 迁移保留 `{row['TransferCount']}` 次，回退 `{row['FallbackCount']}` 次；"
            f"迁移目标 `{row['TransferUnits']}`；回退目标 `{row['FallbackUnits']}`；"
            f"回退原因汇总 `{row['FallbackReasons']}`。"
        )
    for row in transfer_result.get("val_test_gap_rows", []):
        lines.append(
            f"- {row['Classifier']}: 验证集 ACC/F1=`{row['Val_ACC']}`/`{row['Val_F1']}`，"
            f"测试集 ACC/F1=`{row['Test_ACC']}`/`{row['Test_F1']}`，"
            f"落差 ACC/F1=`{row['Gap_ACC']}`/`{row['Gap_F1']}`。"
        )
    return lines


def _print_transfer_diagnostics(transfer_result: Dict[str, Any]) -> None:
    """把迁移保留/回退与验证测试落差打印到控制台，便于保存到调试日志。"""
    for row in transfer_result.get("gate_summary_rows", []):
        print(
            f"[{EXPERIMENT_NAME}][GateSummary] classifier={row['Classifier']} units={row['Units']} "
            f"transfer_kept={row['TransferKept']} target_only_gate={row['TargetOnlyGate']} "
            f"gate_rate={row['GateRate']} top_reason={row['TopGateReason']}"
        )
    for row in transfer_result.get("transfer_decision_rows", []):
        print(
            f"[{EXPERIMENT_NAME}][Decision] classifier={row['Classifier']} "
            f"transfer_count={row['TransferCount']} fallback_count={row['FallbackCount']} "
            f"transfer_units={row['TransferUnits']} fallback_units={row['FallbackUnits']} "
            f"fallback_reasons={row['FallbackReasons']}"
        )
    for row in transfer_result.get("val_test_gap_rows", []):
        print(
            f"[{EXPERIMENT_NAME}][ValTestGap] classifier={row['Classifier']} "
            f"val_ACC={row['Val_ACC']} test_ACC={row['Test_ACC']} gap_ACC={row['Gap_ACC']} "
            f"val_F1={row['Val_F1']} test_F1={row['Test_F1']} gap_F1={row['Gap_F1']}"
        )


def _write_debug_markdown(path: Path, started_at: str, finished_at: str, log_text: str) -> Path:
    """把完整控制台输出保存成 Markdown 调试日志文件。"""
    output_path = ensure_ai_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {EXPERIMENT_NAME} 调试日志",
        "",
        f"- 实验名: `{EXPERIMENT_NAME}`",
        f"- 开始时间: `{started_at}`",
        f"- 结束时间: `{finished_at}`",
        f"- 对应实验摘要 JSON: `{_artifact_file_name(EXPERIMENT_SUMMARY_PATH)}`",
        f"- 对应阶段一 JSON: `{_artifact_file_name(FEATURE_SUMMARY_PATH)}`",
        f"- 对应阶段二 JSON: `{_artifact_file_name(TRANSFER_SUMMARY_PATH)}`",
        "",
        "## 控制台输出",
        "",
        "```text",
        log_text.rstrip(),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_experiment_report(
    report_path: Path,
    args: argparse.Namespace,
    feature_result: Dict[str, Any],
    transfer_result: Dict[str, Any],
    debug_md_path: Path,
    removed_reports: Sequence[str],
) -> Path:
    """生成 MIIFS_CANDD_STABLE 的最终中文总报告。"""
    feature_summary = dict(feature_result["frozen_mask_baseline_summary"])
    transfer_summary = dict(transfer_result["transfer_summary"])
    feature_best_classifier, feature_best_metrics = _best_classifier_from_summary(feature_summary)
    transfer_best_classifier, transfer_best_metrics = _best_classifier_from_summary(transfer_summary)
    feature_rows = _summary_rows(feature_summary)
    transfer_rows = _summary_rows(transfer_summary)
    feature_config = dict(feature_result.get("config", {}))
    transfer_config = dict(transfer_result.get("config", {}))
    transfer_count, fallback_count = _total_transfer_fallback_counts(transfer_result.get("transfer_decision_rows", []))

    intro_lines = [
        f"本报告对应正式实验 `{EXPERIMENT_NAME}`。",
        "本次固定使用 `LR / KNN / DT` 三个分类器，并且统一输出验证集与测试集口径。",
        f"完整调试日志已经保存到 `{debug_md_path}`。",
    ]
    sections: List[Dict[str, Any]] = [
        {
            "title": "产物总览",
            "body_lines": [
                f"- 实验总摘要 JSON: `{_artifact_file_name(EXPERIMENT_SUMMARY_PATH)}`",
                f"- 实验总报告 MD: `{_artifact_file_name(report_path)}`",
                f"- 阶段一摘要 JSON: `{_artifact_file_name(FEATURE_SUMMARY_PATH)}`",
                f"- 阶段一报告 MD: `{_artifact_file_name(FEATURE_REPORT_PATH)}`",
                f"- 阶段二摘要 JSON: `{_artifact_file_name(TRANSFER_SUMMARY_PATH)}`",
                f"- 阶段二报告 MD: `{_artifact_file_name(TRANSFER_REPORT_PATH)}`",
                f"- 调试日志 MD: `{_artifact_file_name(debug_md_path)}`",
                f"- 清理旧报告数量: `{len(removed_reports)}`",
                *( [f"- 已删除旧报告: `{removed}`" for removed in removed_reports] if removed_reports else ["- 已删除旧报告: `无`"] ),
            ],
        },
        {
            "title": "统一实验口径",
            "body_lines": [
                f"- 实验名: `{EXPERIMENT_NAME}`",
                f"- 分类器集合: `{list(CLASSIFIER_ORDER)}`",
                f"- target_split: `{float(args.target_train_ratio):.2f}/{float(args.target_val_ratio):.2f}/{float(args.target_test_ratio):.2f}`",
                f"- 随机种子: `{int(args.seed)}`",
                "",
                "- 阶段一完整超参数 JSON：",
                *_json_block_lines(feature_config),
                "",
                "- 阶段二完整超参数 JSON：",
                *_json_block_lines(transfer_config),
            ],
        },
        {
            "title": "参数注释：阶段一",
            "table": format_metric_table(
                _feature_param_rows(feature_config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "参数注释：阶段二",
            "table": format_metric_table(
                _transfer_param_rows(transfer_config),
                [("Parameter", "参数名"), ("Value", "当前值"), ("Meaning", "作用")],
            ),
        },
        {
            "title": "阶段一结果",
            "body_lines": [
                f"- 最优分类器: `{feature_best_classifier}`",
                f"- 最优 ACC: `{feature_best_metrics['ACC']:.4f}`",
                f"- 最优 F1: `{feature_best_metrics['F1']:.4f}`",
                f"- 阶段一报告: `{ensure_ai_output_path(FEATURE_REPORT_PATH)}`",
                f"- 阶段一摘要 JSON: `{ensure_ai_output_path(FEATURE_SUMMARY_PATH)}`",
                "",
                *format_metric_bars("阶段一冻结 mask ACC", feature_rows, "ACC"),
                "",
                *format_metric_bars("阶段一冻结 mask F1", feature_rows, "F1"),
            ],
            "table": format_metric_table(_summary_table_rows(feature_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
        },
        {
            "title": "阶段二结果",
            "body_lines": [
                f"- 最优分类器: `{transfer_best_classifier}`",
                f"- 最优 ACC: `{transfer_best_metrics['ACC']:.4f}`",
                f"- 最优 F1: `{transfer_best_metrics['F1']:.4f}`",
                f"- 迁移保留总次数: `{transfer_count}`",
                f"- 回退总次数: `{fallback_count}`",
                f"- 阶段二报告: `{ensure_ai_output_path(TRANSFER_REPORT_PATH)}`",
                f"- 阶段二摘要 JSON: `{ensure_ai_output_path(TRANSFER_SUMMARY_PATH)}`",
                "",
                *format_metric_bars("阶段二 Final Transfer ACC", transfer_rows, "ACC"),
                "",
                *format_metric_bars("阶段二 Final Transfer F1", transfer_rows, "F1"),
            ],
            "table": format_metric_table(_summary_table_rows(transfer_summary), ["Classifier", "ACC", "F1", "BACC", "MCC"]),
        },
        {
            "title": "迁移与回退统计",
            "table": format_metric_table(
                transfer_result.get("transfer_decision_rows", []),
                ["Classifier", "TransferCount", "FallbackCount", "TransferUnits", "FallbackUnits", "FallbackReasons"],
            ),
        },
        {
            "title": "验证集与测试集对照",
            "table": format_metric_table(
                transfer_result.get("val_test_gap_rows", []),
                ["Classifier", "Val_ACC", "Test_ACC", "Gap_ACC", "Val_F1", "Test_F1", "Gap_F1"],
            ),
        },
        {
            "title": "问题诊断",
            "body_lines": _diagnostic_lines(transfer_result) or ["- 本次没有额外诊断结论。"],
        },
    ]

    if transfer_result.get("gate_summary_rows"):
        sections.append(
            {
                "title": "Gate 汇总",
                "table": format_metric_table(
                    transfer_result["gate_summary_rows"],
                    ["Classifier", "Units", "TransferKept", "TargetOnlyGate", "GateRate", "TopGateReason"],
                ),
            }
        )

    if transfer_result.get("hard_rows"):
        sections.append(
            {
                "title": "最难目标单元",
                "table": format_metric_table(
                    transfer_result["hard_rows"],
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
                        "Threshold",
                        "TopSources",
                    ],
                ),
            }
        )

    return write_markdown_report(
        path=ensure_ai_output_path(report_path),
        title=f"{EXPERIMENT_NAME} 总报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def _cleanup_obsolete_reports(skip_cleanup: bool) -> List[str]:
    """删除已经被正式 stable 结果替代的旧 refined 总结报告。"""
    if skip_cleanup:
        print(f"[{EXPERIMENT_NAME}][Cleanup] skipped=true")
        return []

    removed: List[str] = []
    for path in OBSOLETE_REPORT_PATHS:
        resolved_path = ensure_ai_output_path(path)
        if resolved_path.exists():
            resolved_path.unlink()
            removed.append(str(resolved_path))
            print(f"[{EXPERIMENT_NAME}][Cleanup] removed={resolved_path}")
    if not removed:
        print(f"[{EXPERIMENT_NAME}][Cleanup] removed=none")
    return removed


def _run_feature_stage(feature_config: FeatureSelectionConfig) -> Dict[str, Any]:
    """执行或复用阶段一特征选择结果。"""
    existing_summary = _load_json_if_exists(feature_config.summary_path)
    if isinstance(existing_summary, dict) and existing_summary.get("frozen_mask_baseline_summary"):
        print(f"[{EXPERIMENT_NAME}][Feature][Resume] summary_path={feature_config.summary_path}")
        return existing_summary

    reused_summary = _try_reuse_legacy_feature_stage(feature_config)
    if isinstance(reused_summary, dict):
        print(
            f"[{EXPERIMENT_NAME}][Feature] summary_json={_artifact_file_name(FEATURE_SUMMARY_PATH)} "
            f"report_md={_artifact_file_name(FEATURE_REPORT_PATH)}"
        )
        return reused_summary

    print(f"[{EXPERIMENT_NAME}][Feature][Start]")
    print(json.dumps(to_serializable(asdict(feature_config)), ensure_ascii=False, indent=2))
    result = run_feature_selection_pipeline(feature_config)
    feature_summary = dict(result["frozen_mask_baseline_summary"])
    best_classifier, best_metrics = _best_classifier_from_summary(feature_summary)
    print(
        f"[{EXPERIMENT_NAME}][StageDone] stage=feature best_classifier={best_classifier} "
        f"ACC={best_metrics['ACC']:.4f} F1={best_metrics['F1']:.4f}"
    )
    print(
        f"[{EXPERIMENT_NAME}][Feature] summary_json={_artifact_file_name(FEATURE_SUMMARY_PATH)} "
        f"report_md={_artifact_file_name(FEATURE_REPORT_PATH)}"
    )
    return result


def _run_transfer_stage(transfer_config: TransferLearningConfig) -> Dict[str, Any]:
    """执行或复用阶段二监督迁移结果。"""
    existing_summary = _load_json_if_exists(transfer_config.summary_path)
    if isinstance(existing_summary, dict) and existing_summary.get("transfer_summary"):
        print(f"[{EXPERIMENT_NAME}][Transfer][Resume] summary_path={transfer_config.summary_path}")
        _print_transfer_diagnostics(existing_summary)
        return existing_summary

    print(f"[{EXPERIMENT_NAME}][Transfer][Start]")
    print(json.dumps(to_serializable(asdict(transfer_config)), ensure_ascii=False, indent=2))
    result = run_transfer_learning_pipeline(transfer_config)
    transfer_summary = dict(result["transfer_summary"])
    best_classifier, best_metrics = _best_classifier_from_summary(transfer_summary)
    transfer_count, fallback_count = _total_transfer_fallback_counts(result.get("transfer_decision_rows", []))
    print(
        f"[{EXPERIMENT_NAME}][StageDone] stage=transfer best_classifier={best_classifier} "
        f"ACC={best_metrics['ACC']:.4f} F1={best_metrics['F1']:.4f} "
        f"transferred={transfer_count} fallback={fallback_count}"
    )
    _print_transfer_diagnostics(result)
    print(
        f"[{EXPERIMENT_NAME}][Transfer] summary_json={_artifact_file_name(TRANSFER_SUMMARY_PATH)} "
        f"report_md={_artifact_file_name(TRANSFER_REPORT_PATH)}"
    )
    return result


def main() -> None:
    """运行 MIIFS_CANDD_STABLE，并生成最终 JSON/MD 报告与调试日志。"""
    args = build_parser().parse_args()
    ensure_artifact_dir()
    ensure_ai_output_path(args.summary_path)
    ensure_ai_output_path(args.report_path)
    ensure_ai_output_path(args.debug_path)

    feature_config = _build_feature_config(args)
    transfer_config = _build_transfer_config(args)
    started_at = _timestamp_text()
    print(f"[{EXPERIMENT_NAME}] started_at={started_at}")

    try:
        with _capture_output() as buffer:
            feature_result = _run_feature_stage(feature_config)
            print(f"[{EXPERIMENT_NAME}] 阶段一实验跑完了。")
            transfer_result = _run_transfer_stage(transfer_config)
            print(f"[{EXPERIMENT_NAME}] 阶段二实验跑完了。")
    except Exception:
        failed_at = _timestamp_text()
        _write_debug_markdown(args.debug_path, started_at, failed_at, buffer.getvalue() if "buffer" in locals() else "")
        raise

    finished_at = _timestamp_text()
    debug_md_path = _write_debug_markdown(args.debug_path, started_at, finished_at, buffer.getvalue())
    removed_reports = _cleanup_obsolete_reports(bool(args.skip_cleanup))
    experiment_report_path = _write_experiment_report(args.report_path, args, feature_result, transfer_result, debug_md_path, removed_reports)

    feature_summary = dict(feature_result["frozen_mask_baseline_summary"])
    transfer_summary = dict(transfer_result["transfer_summary"])
    feature_best_classifier, feature_best_metrics = _best_classifier_from_summary(feature_summary)
    transfer_best_classifier, transfer_best_metrics = _best_classifier_from_summary(transfer_summary)

    experiment_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "generated_at": finished_at,
        "feature_summary_path": str(ensure_ai_output_path(FEATURE_SUMMARY_PATH)),
        "feature_report_path": str(ensure_ai_output_path(FEATURE_REPORT_PATH)),
        "transfer_summary_path": str(ensure_ai_output_path(TRANSFER_SUMMARY_PATH)),
        "transfer_report_path": str(ensure_ai_output_path(TRANSFER_REPORT_PATH)),
        "experiment_summary_path": str(ensure_ai_output_path(args.summary_path)),
        "experiment_report_path": str(ensure_ai_output_path(experiment_report_path)),
        "debug_md_path": str(ensure_ai_output_path(debug_md_path)),
        "reference_summary_path": str(ensure_ai_output_path(args.reference_summary_path)),
        "classifiers": list(CLASSIFIER_ORDER),
        "feature_spec": asdict(FEATURE_SPEC),
        "transfer_spec": asdict(TRANSFER_SPEC),
        "feature_config": feature_result.get("config", to_serializable(asdict(feature_config))),
        "transfer_config": transfer_result.get("config", to_serializable(asdict(transfer_config))),
        "feature_best_classifier": feature_best_classifier,
        "feature_best_metrics": feature_best_metrics,
        "transfer_best_classifier": transfer_best_classifier,
        "transfer_best_metrics": transfer_best_metrics,
        "feature_summary": feature_summary,
        "transfer_summary": transfer_summary,
        "gate_summary_rows": list(transfer_result.get("gate_summary_rows", [])),
        "transfer_decision_rows": list(transfer_result.get("transfer_decision_rows", [])),
        "val_test_gap_rows": list(transfer_result.get("val_test_gap_rows", [])),
        "hard_rows": list(transfer_result.get("hard_rows", [])),
        "diagnostic_lines": _diagnostic_lines(transfer_result),
        "removed_reports": list(removed_reports),
        "json_file_names": {
            "feature_summary_json": _artifact_file_name(FEATURE_SUMMARY_PATH),
            "transfer_summary_json": _artifact_file_name(TRANSFER_SUMMARY_PATH),
            "experiment_summary_json": _artifact_file_name(args.summary_path),
        },
    }
    save_json(to_serializable(experiment_summary), ensure_ai_output_path(args.summary_path))
    print(f"[{EXPERIMENT_NAME}] finished_at={finished_at}")
    print(f"[{EXPERIMENT_NAME}] experiment_summary_json={ensure_ai_output_path(args.summary_path)}")
    print(f"[{EXPERIMENT_NAME}] experiment_report_md={ensure_ai_output_path(experiment_report_path)}")
    print(f"[{EXPERIMENT_NAME}] debug_md={ensure_ai_output_path(debug_md_path)}")


if __name__ == "__main__":
    main()
