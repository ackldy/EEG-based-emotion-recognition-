from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI.transfer_learning_pipeline import TransferLearningConfig, run_transfer_learning_pipeline


def build_parser() -> argparse.ArgumentParser:
    """构建迁移学习主程序的命令行参数。"""
    parser = argparse.ArgumentParser(description="运行迁移学习实验，并生成 JSON 与 Markdown 可视化报告。")
    parser.add_argument("--mask-path", type=Path, default=Path("artifacts/miifs_mask_latest.pkl"))
    parser.add_argument(
        "--feature-source-mode",
        type=str,
        choices=["mask", "full"],
        default="mask",
        help="特征入口模式；mask 表示读取冻结掩码，full 表示直接使用全部特征。",
    )
    parser.add_argument(
        "--feature-source-label",
        type=str,
        default="MIIFS",
        help="特征入口标签，会写入迁移学习报告和汇总结果。",
    )
    parser.add_argument("--summary-path", type=Path, default=Path("AI/artifacts/transfer_learning_summary_latest.json"))
    parser.add_argument("--report-path", type=Path, default=Path("AI/artifacts/transfer_learning_report_latest.md"))
    parser.add_argument("--reference-summary-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-protocol", type=str, choices=["loso", "group_kfold"], default="loso")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--max-target-subjects",
        type=int,
        default=5,
        help="调试阶段默认只跑前 5 个目标单元；传 0 表示跑完整实验。",
    )
    parser.add_argument("--target-train-ratio", type=float, default=0.60)
    parser.add_argument("--target-val-ratio", type=float, default=0.20)
    parser.add_argument("--target-test-ratio", type=float, default=0.20)
    parser.add_argument("--max-source-subjects", type=int, default=6)
    parser.add_argument("--source-sample-cap", type=int, default=128)
    parser.add_argument("--mmd-sample-cap", type=int, default=160)
    parser.add_argument(
        "--mmd-prefix-top-k",
        type=int,
        default=0,
        help="MMD 前缀搜索上限；小于等于 0 表示把前 1、前 2、前 3... 全部累计前缀都交给验证集复排。",
    )
    parser.add_argument(
        "--transfer-variant",
        type=str,
        choices=["supervised", "enhanced_jda", "twod_cnn", "dann"],   # 新增 dann
        default="supervised",
        help="迁移训练分支；supervised 为当前默认监督迁移，enhanced_jda 为增强多源域选择 + JDA，twod_cnn 为 TwoD_CNN 特征提取器，dann 为 DANN 域自适应。",
    )
    parser.add_argument(
        "--target-repeat-grid",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="目标域强度网格；对 LR/DT 表示目标样本权重，对 KNN 表示目标样本重复次数。",
    )
    parser.add_argument(
        "--source-target-ratio-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="源域与目标域有效训练强度的目标比例网格，用于约束迁移时两侧的相对影响力。",
    )
    parser.add_argument("--source-positive-ratio-gap-threshold", type=float, default=0.08)
    parser.add_argument("--gate-repeat-count", type=int, default=3)

    # JDA 参数（保持不变）
    parser.add_argument("--jda-dim", type=int, default=32, help="JDA 潜空间维度。")
    parser.add_argument("--jda-iterations", type=int, default=6, help="JDA 伪标签迭代轮数。")
    parser.add_argument("--jda-components", type=int, default=48, help="JDA 进入线性投影前的工作维度。")
    parser.add_argument("--jda-lambda", type=float, default=1.0, help="JDA 结构保持项权重。")
    parser.add_argument("--jda-reg", type=float, default=1e-6, help="JDA 数值稳定正则项。")
    parser.add_argument(
        "--jda-pseudo-labeler",
        type=str,
        choices=["1nn", "knn", "lr", "svm"],
        default="1nn",
        help="JDA 在潜空间里生成目标域伪标签时使用的分类器。",
    )
    parser.add_argument("--jda-pseudo-neighbors", type=int, default=3, help="当伪标签器为 knn 时使用的邻居数。")
    parser.add_argument("--jda-pseudo-change-tol", type=float, default=1e-3, help="JDA 伪标签变化收敛阈值。")
    parser.add_argument("--jda-mmd-delta-tol", type=float, default=1e-3, help="JDA 潜空间 MMD 变化收敛阈值。")
    parser.add_argument("--jda-confidence-delta-tol", type=float, default=5e-3, help="JDA 置信度变化收敛阈值。")
    parser.add_argument("--jda-early-stop-patience", type=int, default=2, help="JDA 连续稳定多少轮后允许早停。")
    parser.add_argument("--jda-min-iterations", type=int, default=2, help="JDA 允许早停前至少跑多少轮。")
    parser.add_argument(
        "--jda-pseudo-keep-ratio-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.25, 0.4],
        help="论文式 JDA 中，高置信伪标签目标样本参与自训练时的保留比例网格。",
    )
    parser.add_argument(
        "--jda-pseudo-target-repeat-grid",
        type=int,
        nargs="+",
        default=[1, 2],
        help="论文式 JDA 中，高置信伪标签目标样本参与训练时的重复次数网格。",
    )

    # TwoD_CNN 参数（如果你之前添加过，请保留；否则一并添加）
    parser.add_argument("--twod-cnn-encoding-dim", type=int, default=100, help="TwoD_CNN 输出特征维度。")
    parser.add_argument("--twod-cnn-epochs", type=int, default=20, help="TwoD_CNN 训练轮数。")
    parser.add_argument("--twod-cnn-batch-size", type=int, default=32, help="TwoD_CNN 批次大小。")

    # === 新增 DANN 相关参数 ===
    parser.add_argument("--dann-encoding-dim", type=int, default=64, help="DANN 特征提取器输出维度。")
    parser.add_argument("--dann-lambda", type=float, default=1.0, help="DANN 梯度反转层系数（域适应强度）。")
    parser.add_argument("--dann-epochs", type=int, default=20, help="DANN 训练轮数。")
    parser.add_argument("--dann-batch-size", type=int, default=32, help="DANN 训练批次大小。")
    # ===========================

    return parser

def main() -> None:
    """解析命令行参数并启动迁移学习流程。"""
    args = build_parser().parse_args()
    max_target_subjects = None if args.max_target_subjects is None or args.max_target_subjects <= 0 else args.max_target_subjects
    config = TransferLearningConfig(
        mask_path=args.mask_path,
        feature_source_mode=args.feature_source_mode,
        feature_source_label=args.feature_source_label,
        summary_path=args.summary_path,
        report_path=args.report_path,
        random_state=args.seed,
        evaluation_protocol=args.evaluation_protocol,
        cv_splits=args.cv_splits,
        max_target_subjects=max_target_subjects,
        target_train_ratio=args.target_train_ratio,
        target_val_ratio=args.target_val_ratio,
        target_test_ratio=args.target_test_ratio,
        max_source_subjects=args.max_source_subjects,
        source_sample_cap=args.source_sample_cap,
        mmd_sample_cap=args.mmd_sample_cap,
        mmd_prefix_top_k=args.mmd_prefix_top_k,
        transfer_variant=args.transfer_variant,
        target_repeat_grid=tuple(args.target_repeat_grid),
        source_target_ratio_grid=tuple(args.source_target_ratio_grid),
        jda_dim=int(args.jda_dim),
        jda_iterations=int(args.jda_iterations),
        jda_n_components=int(args.jda_components),
        jda_lambda=float(args.jda_lambda),
        jda_reg=float(args.jda_reg),
        jda_pseudo_labeler=str(args.jda_pseudo_labeler),
        jda_pseudo_neighbors=int(args.jda_pseudo_neighbors),
        jda_pseudo_change_tol=float(args.jda_pseudo_change_tol),
        jda_mmd_delta_tol=float(args.jda_mmd_delta_tol),
        jda_confidence_delta_tol=float(args.jda_confidence_delta_tol),
        jda_early_stop_patience=int(args.jda_early_stop_patience),
        jda_min_iterations=int(args.jda_min_iterations),
        jda_pseudo_keep_ratio_grid=tuple(args.jda_pseudo_keep_ratio_grid),
        jda_pseudo_target_repeat_grid=tuple(args.jda_pseudo_target_repeat_grid),
        reference_summary_path=args.reference_summary_path,
        source_positive_ratio_gap_threshold=float(args.source_positive_ratio_gap_threshold),
        gate_repeat_count=int(args.gate_repeat_count),
        # TwoD_CNN 参数（如果有）
        twod_cnn_encoding_dim=args.twod_cnn_encoding_dim,
        twod_cnn_epochs=args.twod_cnn_epochs,
        twod_cnn_batch_size=args.twod_cnn_batch_size,
        # === 新增 DANN 参数传递 ===
        dann_encoding_dim=args.dann_encoding_dim,
        dann_lambda=args.dann_lambda,
        dann_epochs=args.dann_epochs,
        dann_batch_size=args.dann_batch_size,
    )
    run_transfer_learning_pipeline(config)


if __name__ == "__main__":
    main()
