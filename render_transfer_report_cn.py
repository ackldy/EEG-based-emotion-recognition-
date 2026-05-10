from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.reporting import format_metric_bars, format_metric_table, write_markdown_report


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    """读取迁移学习汇总 JSON。"""
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _protocol_label(protocol: str) -> str:
    """把评估协议名称转换成中文。"""
    normalized = str(protocol).lower().strip()
    if normalized == "loso":
        return "留一被试"
    if normalized == "group_kfold":
        return "按被试分组 K 折"
    return normalized or "-"


def _transfer_variant_label(transfer_variant: str) -> str:
    """把迁移分支名称转换成中文。"""
    normalized = str(transfer_variant).lower().strip()
    if normalized == "enhanced_jda":
        return "增强多源域选择 + JDA"
    if normalized == "supervised":
        return "监督迁移"
    return normalized or "-"


def _alignment_method_label(alignment_method: str) -> str:
    """把域对齐方法名称转换成中文。"""
    normalized = str(alignment_method).lower().strip()
    if normalized == "jda":
        return "JDA 联合分布自适应"
    if normalized == "none":
        return "无额外域对齐"
    return normalized or "-"


def _selection_mode_label(selection_mode: str) -> str:
    """把最终选择模式转换成中文。"""
    normalized = str(selection_mode).lower().strip()
    if normalized == "transfer":
        return "保留迁移"
    if normalized == "target_only_gate":
        return "回退到目标域基线"
    return normalized or "-"


def _gate_reason_label(gate_reason: str) -> str:
    """把门控原因编码翻译成中文。"""
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
    """把带计数的门控原因字符串翻译成中文。"""
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


def _split_mode_label(split_mode: str) -> str:
    """把切分方式转换成中文。"""
    normalized = str(split_mode).lower().strip()
    if normalized == "subject_equal_count_stratified":
        return "单被试等量分层切分"
    return normalized or "-"


def _target_unit_label(target_unit: str) -> str:
    """把目标单元标识转换成中文。"""
    text = str(target_unit).strip()
    if text.startswith("subject_"):
        return "被试_" + text.split("_", 1)[1]
    return text or "-"


def _per_subject_split_label(per_subject_split: str) -> str:
    """把单被试切分说明转换成中文。"""
    return str(per_subject_split).replace("per_subject", "每被试")


def _prefix_rank_preview_label(preview_text: str) -> str:
    """把前缀池预览字符串转换成更接近中文阅读的形式。"""
    text = str(preview_text)
    text = text.replace("rank", "第")
    text = text.replace(":p", "名:前缀")
    text = text.replace("/mmd=", "/MMD=")
    return text


def _yes_no_label(flag_text: str) -> str:
    """把 Y/空字符串 标记转换成 是/否。"""
    return "是" if str(flag_text).strip().upper() == "Y" else "否"


def _metric_mean(summary_block: Dict[str, Any], classifier_name: str, metric_name: str) -> float:
    """从 mean/std 风格汇总中读取均值。"""
    metric_value = summary_block[classifier_name][metric_name]
    if isinstance(metric_value, dict):
        return float(metric_value.get("mean", 0.0))
    return float(metric_value)


def _summary_bar_rows(summary_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把迁移汇总转换成条形图数据。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in ["LR", "KNN", "DT"]:
        if classifier_name not in summary_block:
            continue
        rows.append(
            {
                "Classifier": classifier_name,
                "ACC": _metric_mean(summary_block, classifier_name, "ACC"),
                "F1": _metric_mean(summary_block, classifier_name, "F1"),
            }
        )
    return rows


def _summary_table_rows(summary_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把迁移汇总转换成紧凑表格数据。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in ["LR", "KNN", "DT"]:
        if classifier_name not in summary_block:
            continue
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


def _jda_diagnostic_rows(subject_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从单元级结果中汇总 JDA 诊断。"""
    rows: List[Dict[str, Any]] = []
    for classifier_name in ["LR", "KNN", "DT"]:
        classifier_rows = [row for row in subject_rows if str(row.get("Classifier")) == classifier_name]
        jda_rows = [row for row in classifier_rows if int(row.get("JDAIterations", 0)) > 0]
        if not jda_rows:
            continue
        rows.append(
            {
                "Classifier": classifier_name,
                "Units": len(jda_rows),
                "MeanJDAIter": f"{sum(int(row.get('JDAIterations', 0)) for row in jda_rows) / len(jda_rows):.2f}",
                "MeanEndMMD": f"{sum(float(row.get('JDALastMMD', 0.0)) for row in jda_rows) / len(jda_rows):.4f}",
                "MeanMMDDelta": f"{sum(float(row.get('JDAMMDDelta', 0.0)) for row in jda_rows) / len(jda_rows):+.4f}",
                "MeanPseudoChange": f"{sum(float(row.get('JDAPseudoChange', 0.0)) for row in jda_rows) / len(jda_rows):.4f}",
                "MeanConfidence": f"{sum(float(row.get('JDAConfidence', 0.0)) for row in jda_rows) / len(jda_rows):.4f}",
            }
        )
    return rows


def _localize_rows(summary: Dict[str, Any]) -> Dict[str, Any]:
    """为中文报告准备带中文展示字段的行数据。"""
    split_overview_rows = [
        {
            **row,
            "TargetUnitLabel": _target_unit_label(str(row.get("TargetUnit", ""))),
            "SplitModeLabel": _split_mode_label(str(row.get("SplitMode", ""))),
            "PerSubjectSplitLabel": _per_subject_split_label(str(row.get("PerSubjectSplit", ""))),
        }
        for row in summary.get("split_overview_rows", [])
    ]
    mmd_selection_rows = [
        {
            **row,
            "TargetUnitLabel": _target_unit_label(str(row.get("TargetUnit", ""))),
            "PrefixRankPreviewLabel": _prefix_rank_preview_label(str(row.get("PrefixRankPreview", ""))),
        }
        for row in summary.get("mmd_selection_rows", [])
    ]
    gate_summary_rows = [
        {
            **row,
            "TopGateReasonLabel": _gate_reason_label(str(row.get("TopGateReason", ""))),
        }
        for row in summary.get("gate_summary_rows", [])
    ]
    transfer_decision_rows = [
        {
            **row,
            "TransferUnitsLabel": ",".join(_target_unit_label(item) for item in str(row.get("TransferUnits", "")).split(",") if item),
            "FallbackUnitsLabel": ",".join(_target_unit_label(item) for item in str(row.get("FallbackUnits", "")).split(",") if item),
            "FallbackReasonsLabel": _reason_counter_text_to_chinese(str(row.get("FallbackReasons", ""))),
        }
        for row in summary.get("transfer_decision_rows", [])
    ]
    subject_rows = [
        {
            **row,
            "TargetUnitLabel": _target_unit_label(str(row.get("TargetUnit", ""))),
            "TransferVariantLabel": _transfer_variant_label(str(row.get("TransferVariant", summary.get("transfer_variant", "")))),
            "AlignmentMethodLabel": _alignment_method_label(str(row.get("AlignmentMethod", ""))),
            "SelectionModeLabel": _selection_mode_label(str(row.get("SelectionMode", ""))),
            "GateReasonLabel": _gate_reason_label(str(row.get("GateReason", ""))),
            "GateRepeatPassedLabel": _yes_no_label(str(row.get("GateRepeatPassed", ""))),
        }
        for row in summary.get("subject_rows", [])
    ]
    hard_rows = [
        {
            **row,
            "TargetUnitLabel": _target_unit_label(str(row.get("TargetUnit", ""))),
            "TransferVariantLabel": _transfer_variant_label(str(row.get("TransferVariant", summary.get("transfer_variant", "")))),
        }
        for row in summary.get("subject_rows", [])
    ]
    hard_rows = sorted(
        hard_rows,
        key=lambda row: (float(row.get("Transfer_ACC", 0.0)), float(row.get("Transfer_F1", 0.0))),
    )[: min(10, len(hard_rows))]
    return {
        "split_overview_rows": split_overview_rows,
        "mmd_selection_rows": mmd_selection_rows,
        "gate_summary_rows": gate_summary_rows,
        "transfer_decision_rows": transfer_decision_rows,
        "subject_rows": subject_rows,
        "hard_rows": hard_rows,
        "jda_diag_rows": _jda_diagnostic_rows(subject_rows),
    }


def build_report(summary: Dict[str, Any], report_path: Path) -> Path:
    """根据迁移学习汇总结果生成全中文 Markdown 报告。"""
    config = summary.get("config", {})
    transfer_variant = str(summary.get("transfer_variant", "supervised"))
    transfer_variant_label = _transfer_variant_label(transfer_variant)
    localized = _localize_rows(summary)
    source_only_bar_rows = _summary_bar_rows(summary["source_only_summary"])
    target_only_bar_rows = _summary_bar_rows(summary["target_only_summary"])
    raw_transfer_bar_rows = _summary_bar_rows(summary["raw_transfer_summary"])
    final_transfer_bar_rows = _summary_bar_rows(summary["final_transfer_summary"])

    intro_lines = [
        "本报告固定使用目标域 `60% 训练 / 20% 验证 / 20% 测试` 划分。",
        "对比基线包括只用源域训练的“源域基线”，以及只用目标域训练的“目标域基线”。",
        f"当前特征入口为 `{summary.get('feature_source_label', '')}`。",
        f"当前迁移训练分支为 `{transfer_variant_label}`。",
    ]
    if str(summary.get("jda_config_text", "")).strip():
        intro_lines.append(f"JDA 配置: `{summary['jda_config_text']}`")
    intro_lines.append("“原始迁移结果”表示迁移候选在门控之前的测试结果，“最终迁移结果”表示和目标域基线比较并执行保留/回退后的最终结果。")

    sections: List[Dict[str, Any]] = [
        {
            "title": "运行配置",
            "body_lines": [
                f"- 特征入口标签: `{summary.get('feature_source_label', '')}`",
                f"- 特征入口模式: `{'冻结掩码' if str(summary.get('feature_source_mode', 'mask')) == 'mask' else '全特征'}`",
                f"- 掩码文件: `{config.get('mask_path', '')}`",
                f"- 评估协议: `{_protocol_label(str(config.get('evaluation_protocol', '')) )}`",
                f"- 目标单元上限: `{config.get('max_target_subjects', 'all')}`",
                f"- 最多源被试数: `{config.get('max_source_subjects', '')}`",
                f"- 源域采样上限: `{config.get('source_sample_cap', '')}`",
                f"- MMD 采样上限: `{config.get('mmd_sample_cap', '')}`",
                f"- MMD 前缀搜索范围: `{config.get('mmd_prefix_top_k', '')}`",
                f"- 目标域强度网格: `{config.get('target_repeat_grid', [])}`",
                f"- 源/目标强度比例网格: `{config.get('source_target_ratio_grid', [])}`",
                f"- 迁移分支: `{transfer_variant_label}`",
                f"- 源域正类比例过滤阈值: `{float(config.get('source_positive_ratio_gap_threshold', 0.0)):.4f}`",
                f"- 门控复核次数: `{config.get('gate_repeat_count', '')}`",
                *( [f"- JDA 配置: `{summary['jda_config_text']}`"] if str(summary.get("jda_config_text", "")).strip() else [] ),
            ],
        },
        {
            "title": "阶段一参考",
            "body_lines": [
                f"- 阶段一最佳分类器: `{summary.get('stage1_best_classifier_name', '-')}`",
                f"- 阶段一最佳 ACC: `{float(summary.get('stage1_best_classifier_metrics', {}).get('ACC', 0.0)):.4f}`",
                f"- 阶段一最佳 F1: `{float(summary.get('stage1_best_classifier_metrics', {}).get('F1', 0.0)):.4f}`",
            ],
            "table": format_metric_table(
                summary.get("stage1_classifier_rows", []),
                [
                    ("Classifier", "分类器"),
                    ("FeatureCount", "特征数"),
                    ("Stage1_ACC", "阶段一 ACC"),
                    ("Stage1_F1", "阶段一 F1"),
                ],
            ),
        },
        {
            "title": "目标域切分概览",
            "table": format_metric_table(
                localized["split_overview_rows"],
                [
                    ("Classifier", "分类器"),
                    ("MaskFeatureCount", "特征数"),
                    ("TargetUnitLabel", "目标单元"),
                    ("TargetSubjects", "目标被试"),
                    ("Train", "训练样本数"),
                    ("Val", "验证样本数"),
                    ("Test", "测试样本数"),
                    ("SourceFilterKept", "保留源被试"),
                    ("SplitModeLabel", "切分方式"),
                    ("PerSubjectSplitLabel", "单被试切分"),
                    ("TrainPosRatio", "训练正类比例"),
                    ("ValPosRatio", "验证正类比例"),
                    ("TestPosRatio", "测试正类比例"),
                    ("SourceRankPreview", "源域排序预览"),
                    ("ChosenPrefix", "最优前缀长度"),
                    ("ChosenMMD", "最优前缀 MMD"),
                ],
            ),
        },
        {
            "title": "累计前缀 MMD 排序",
            "table": format_metric_table(
                localized["mmd_selection_rows"],
                [
                    ("Classifier", "分类器"),
                    ("MaskFeatureCount", "特征数"),
                    ("TargetUnitLabel", "目标单元"),
                    ("BestMMDPrefix", "最优 MMD 前缀"),
                    ("BestMMD", "最优 MMD"),
                    ("SourceFilterKept", "保留源被试"),
                    ("BestMMDSources", "最优源被试组合"),
                    ("PrefixRankPreviewLabel", "前缀池预览"),
                    ("SourceRankPreview", "源域排序预览"),
                    ("WeightPreview", "权重预览"),
                ],
            ),
        },
    ]

    if summary.get("legacy_reference_rows"):
        sections.append(
            {
                "title": "上一轮迁移结果",
                "body_lines": [f"- 参考文件: `{summary.get('legacy_reference_path', '')}`"],
                "table": format_metric_table(
                    summary.get("legacy_reference_rows", []),
                    [("Classifier", "分类器"), ("Prev_ACC", "上一轮 ACC"), ("Prev_F1", "上一轮 F1")],
                ),
            }
        )

    if summary.get("reference_delta_rows"):
        sections.append(
            {
                "title": "相对上一轮的变化",
                "table": format_metric_table(
                    summary.get("reference_delta_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("Prev_ACC", "上一轮 ACC"),
                        ("Current_ACC", "当前 ACC"),
                        ("Delta_ACC", "ACC 变化"),
                        ("Prev_F1", "上一轮 F1"),
                        ("Current_F1", "当前 F1"),
                        ("Delta_F1", "F1 变化"),
                    ],
                ),
            }
        )

    if localized["gate_summary_rows"]:
        sections.append(
            {
                "title": "门控统计",
                "table": format_metric_table(
                    localized["gate_summary_rows"],
                    [
                        ("Classifier", "分类器"),
                        ("Units", "目标单元数"),
                        ("TransferKept", "保留迁移数"),
                        ("TargetOnlyGate", "回退数"),
                        ("GateRate", "回退率"),
                        ("TopGateReasonLabel", "主要回退原因"),
                    ],
                ),
            }
        )

    if localized["transfer_decision_rows"]:
        sections.append(
            {
                "title": "迁移/回退明细",
                "table": format_metric_table(
                    localized["transfer_decision_rows"],
                    [
                        ("Classifier", "分类器"),
                        ("TransferCount", "保留迁移数"),
                        ("FallbackCount", "回退数"),
                        ("TransferUnitsLabel", "保留迁移单元"),
                        ("FallbackUnitsLabel", "回退单元"),
                        ("FallbackReasonsLabel", "回退原因"),
                    ],
                ),
            }
        )

    if summary.get("val_test_gap_rows"):
        sections.append(
            {
                "title": "验证到测试落差",
                "table": format_metric_table(
                    summary.get("val_test_gap_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("Val_ACC", "验证集 ACC"),
                        ("Test_ACC", "测试集 ACC"),
                        ("Gap_ACC", "ACC 落差"),
                        ("Val_F1", "验证集 F1"),
                        ("Test_F1", "测试集 F1"),
                        ("Gap_F1", "F1 落差"),
                    ],
                ),
            }
        )

    if summary.get("prefix_choice_rows"):
        sections.append(
            {
                "title": "前缀选择分布",
                "table": format_metric_table(
                    summary.get("prefix_choice_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("TopPrefixRank", "最常见前缀排名"),
                        ("TopPrefixSize", "最常见前缀长度"),
                        ("TopChoiceCount", "出现次数"),
                        ("MeanDesiredRatio", "平均目标比例"),
                        ("MeanActualRatio", "平均实际比例"),
                    ],
                ),
            }
        )

    if summary.get("source_frequency_rows"):
        sections.append(
            {
                "title": "高频源域",
                "table": format_metric_table(
                    summary.get("source_frequency_rows", []),
                    [("Classifier", "分类器"), ("SourceId", "源被试编号"), ("ChosenCount", "被选次数")],
                ),
            }
        )

    if localized["jda_diag_rows"]:
        sections.append(
            {
                "title": "JDA 诊断",
                "table": format_metric_table(
                    localized["jda_diag_rows"],
                    [
                        ("Classifier", "分类器"),
                        ("Units", "目标单元数"),
                        ("MeanJDAIter", "平均迭代轮数"),
                        ("MeanEndMMD", "平均最终 MMD"),
                        ("MeanMMDDelta", "平均 MMD 变化"),
                        ("MeanPseudoChange", "平均伪标签变化"),
                        ("MeanConfidence", "平均置信度"),
                    ],
                ),
            }
        )

    if summary.get("method_comparison_rows"):
        sections.append(
            {
                "title": "四路方法对照",
                "body_lines": [
                    "- “源域基线”：只用源域训练。",
                    "- “目标域基线”：只用目标域训练。",
                    "- “原始迁移结果”：迁移候选未经门控回退前的原始测试结果。",
                    "- “最终迁移结果”：执行门控后真正进入最终汇总的结果。",
                ],
                "table": format_metric_table(
                    summary.get("method_comparison_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("SourceOnly_ACC", "源域基线 ACC"),
                        ("SourceOnly_F1", "源域基线 F1"),
                        ("TargetOnly_ACC", "目标域基线 ACC"),
                        ("TargetOnly_F1", "目标域基线 F1"),
                        ("RawTransfer_ACC", "原始迁移 ACC"),
                        ("RawTransfer_F1", "原始迁移 F1"),
                        ("FinalTransfer_ACC", "最终迁移 ACC"),
                        ("FinalTransfer_F1", "最终迁移 F1"),
                    ],
                ),
            }
        )

    sections.extend(
        [
            {
                "title": "源域基线",
                "body_lines": format_metric_bars("源域基线 ACC", source_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("源域基线 F1", source_only_bar_rows, "F1"),
                "table": format_metric_table(source_only_bar_rows, [("Classifier", "分类器"), ("ACC", "ACC"), ("F1", "F1")]),
            },
            {
                "title": "目标域基线",
                "body_lines": format_metric_bars("目标域基线 ACC", target_only_bar_rows, "ACC")
                + [""]
                + format_metric_bars("目标域基线 F1", target_only_bar_rows, "F1"),
                "table": format_metric_table(target_only_bar_rows, [("Classifier", "分类器"), ("ACC", "ACC"), ("F1", "F1")]),
            },
            {
                "title": "原始迁移结果",
                "body_lines": format_metric_bars("原始迁移 ACC", raw_transfer_bar_rows, "ACC")
                + [""]
                + format_metric_bars("原始迁移 F1", raw_transfer_bar_rows, "F1"),
                "table": format_metric_table(
                    _summary_table_rows(summary["raw_transfer_summary"]),
                    [("Classifier", "分类器"), ("ACC", "ACC"), ("F1", "F1"), ("BACC", "BACC"), ("MCC", "MCC")],
                ),
            },
            {
                "title": "最终迁移结果",
                "body_lines": format_metric_bars("最终迁移 ACC", final_transfer_bar_rows, "ACC")
                + [""]
                + format_metric_bars("最终迁移 F1", final_transfer_bar_rows, "F1"),
                "table": format_metric_table(
                    _summary_table_rows(summary["final_transfer_summary"]),
                    [("Classifier", "分类器"), ("ACC", "ACC"), ("F1", "F1"), ("BACC", "BACC"), ("MCC", "MCC")],
                ),
            },
            {
                "title": "相对源域基线的提升",
                "table": format_metric_table(
                    summary.get("gain_vs_source_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("SourceOnly_ACC", "源域基线 ACC"),
                        ("Transfer_ACC", "最终迁移 ACC"),
                        ("Delta_ACC", "ACC 提升"),
                        ("SourceOnly_F1", "源域基线 F1"),
                        ("Transfer_F1", "最终迁移 F1"),
                        ("Delta_F1", "F1 提升"),
                    ],
                ),
            },
            {
                "title": "相对目标域基线的提升",
                "table": format_metric_table(
                    summary.get("gain_vs_target_rows", []),
                    [
                        ("Classifier", "分类器"),
                        ("TargetOnly_ACC", "目标域基线 ACC"),
                        ("Transfer_ACC", "最终迁移 ACC"),
                        ("Delta_ACC", "ACC 提升"),
                        ("TargetOnly_F1", "目标域基线 F1"),
                        ("Transfer_F1", "最终迁移 F1"),
                        ("Delta_F1", "F1 提升"),
                    ],
                ),
            },
            {
                "title": "目标单元明细",
                "table": format_metric_table(
                    localized["subject_rows"],
                    [
                        ("TargetUnitLabel", "目标单元"),
                        ("Classifier", "分类器"),
                        ("MaskFeatureCount", "特征数"),
                        ("SourceOnly_ACC", "源域基线 ACC"),
                        ("TargetOnly_ACC", "目标域基线 ACC"),
                        ("RawTransfer_ACC", "原始迁移 ACC"),
                        ("RawTransfer_F1", "原始迁移 F1"),
                        ("Val_ACC", "验证集 ACC"),
                        ("Val_F1", "验证集 F1"),
                        ("Transfer_ACC", "最终迁移 ACC"),
                        ("Transfer_F1", "最终迁移 F1"),
                        ("DeltaVsSource_ACC", "相对源域基线 ACC"),
                        ("DeltaVsTarget_ACC", "相对目标域基线 ACC"),
                        ("RawDeltaVsTarget_ACC", "原始迁移相对目标域 ACC"),
                        ("RawDeltaVsTarget_F1", "原始迁移相对目标域 F1"),
                        ("PrefixMMD", "前缀 MMD"),
                        ("TransferVariantLabel", "迁移分支"),
                        ("AlignmentMethodLabel", "对齐方法"),
                        ("JDAIterations", "JDA 迭代轮数"),
                        ("JDALastMMD", "JDA 最终 MMD"),
                        ("JDAMMDDelta", "JDA MMD 变化"),
                        ("JDAPseudoChange", "JDA 伪标签变化"),
                        ("JDAConfidence", "JDA 平均置信度"),
                        ("SelectionModeLabel", "最终选择"),
                        ("GateReasonLabel", "门控原因"),
                        ("GateRepeatPassedLabel", "重复复核通过"),
                        ("GateRepeatMeanGain_ACC", "重复复核平均 ACC 增益"),
                        ("GateRepeatMeanGain_F1", "重复复核平均 F1 增益"),
                        ("Threshold", "阈值"),
                        ("TopSources", "源被试组合"),
                    ],
                ),
            },
            {
                "title": "最难目标单元",
                "table": format_metric_table(
                    localized["hard_rows"],
                    [
                        ("TargetUnitLabel", "目标单元"),
                        ("Classifier", "分类器"),
                        ("MaskFeatureCount", "特征数"),
                        ("RawTransfer_ACC", "原始迁移 ACC"),
                        ("RawTransfer_F1", "原始迁移 F1"),
                        ("Transfer_ACC", "最终迁移 ACC"),
                        ("Transfer_F1", "最终迁移 F1"),
                        ("DeltaVsSource_ACC", "相对源域基线 ACC"),
                        ("DeltaVsTarget_ACC", "相对目标域基线 ACC"),
                        ("PrefixMMD", "前缀 MMD"),
                        ("TransferVariantLabel", "迁移分支"),
                        ("JDALastMMD", "JDA 最终 MMD"),
                        ("JDAPseudoChange", "JDA 伪标签变化"),
                        ("Threshold", "阈值"),
                        ("TopSources", "源被试组合"),
                    ],
                ),
            },
        ]
    )

    return write_markdown_report(
        path=report_path,
        title=f"{transfer_variant_label} 报告",
        intro_lines=intro_lines,
        sections=sections,
    )


def build_parser() -> argparse.ArgumentParser:
    """构建中文报告重写脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="把迁移学习 summary.json 重写成全中文 Markdown 报告。")
    parser.add_argument("--summary-path", type=Path, required=True, help="迁移学习汇总 JSON 路径。")
    parser.add_argument("--report-path", type=Path, default=None, help="输出 Markdown 路径；默认覆盖 summary 里的 report_path。")
    return parser


def main() -> None:
    """读取 summary 并重写成中文 Markdown 报告。"""
    args = build_parser().parse_args()
    summary = _load_summary(args.summary_path)
    report_path = args.report_path if args.report_path is not None else Path(summary.get("report_path", "AI/artifacts/transfer_learning_report_cn.md"))
    output_path = build_report(summary=summary, report_path=report_path)
    print(f"[TransferReportCN] summary_path={args.summary_path}")
    print(f"[TransferReportCN] report_path={output_path}")


if __name__ == "__main__":
    main()
