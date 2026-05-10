import sys
import json
import csv
import itertools
from pathlib import Path
import types

# ========== 1. 设置项目根目录 ==========
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ========== 2. 模拟缺失的 wwt.runtime_safe 模块 ==========
wwt_runtime_safe = types.ModuleType("wwt.runtime_safe")
def apply_safe_runtime_env():
    pass
wwt_runtime_safe.apply_safe_runtime_env = apply_safe_runtime_env
sys.modules["wwt.runtime_safe"] = wwt_runtime_safe

# ========== 3. 导入项目模块 ==========
from AI.common import ensure_artifact_dir
from AI.transfer_learning_pipeline import TransferLearningConfig, run_transfer_learning_pipeline


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
    return items


def try_load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}


def extract_score(summary: dict):
    """
    尽量兼容不同 summary 结构，优先提取 final transfer 的 ACC/F1。
    若结构不同，会回退到常见字段。
    """
    candidates = []

    # 常见的整体汇总路径尝试
    possible_paths = [
        ("final_transfer",),
        ("final",),
        ("metrics", "final_transfer"),
        ("summary", "final_transfer"),
        ("aggregate", "final_transfer"),
    ]

    for path in possible_paths:
        cur = summary
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, dict):
            candidates.append(cur)

    # 若有 classifiers 维度，按 LR/KNN/DT 聚合
    classifier_block_keys = [
        "classifiers",
        "classifier_results",
        "results_by_classifier"
        "per_classifier",
    ]
    classifier_metrics = {}

    for blk_key in classifier_block_keys:
        blk = summary.get(blk_key)
        if isinstance(blk, dict):
            for clf_name, clf_val in blk.items():
                if not isinstance(clf_val, dict):
                    continue
                # 优先 final transfer
                for sub_key in ["final_transfer", "final", "transfer", "metrics"]:
                    maybe = clf_val.get(sub_key)
                    if isinstance(maybe, dict):
                        acc = safe_float(
                            maybe.get("acc", maybe.get("ACC", maybe.get("accuracy")))
                        )
                        f1 = safe_float(
                            maybe.get("f1", maybe.get("F1"))
                        )
                        classifier_metrics[clf_name] = {"acc": acc, "f1": f1}
                        break
                else:
                    acc = safe_float(
                        clf_val.get("acc", clf_val.get("ACC", clf_val.get("accuracy")))
                    )
                    f1 = safe_float(
                        clf_val.get("f1", clf_val.get("F1"))
                    )
                    if acc or f1:
                        classifier_metrics[clf_name] = {"acc": acc, "f1": f1}

    # 若有 classifier 级别结果，优先用它们做平均
    if classifier_metrics:
        accs = [v["acc"] for v in classifier_metrics.values()]
        f1s = [v["f1"] for v in classifier_metrics.values()]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        score = 0.5 * mean_acc + 0.5 * mean_f1
        return {
            "mean_acc": mean_acc,
            "mean_f1": mean_f1,
            "score": score,
            "classifier_metrics": classifier_metrics,
            "source": "classifier_average",
        }

    # 否则尝试从 candidates 抽
    for c in candidates:
        acc = safe_float(c.get("acc", c.get("ACC", c.get("accuracy"))))
        f1 = safe_float(c.get("f1", c.get("F1")))
        if acc or f1:
            score = 0.5 * acc + 0.5 * f1
            return {
                "mean_acc": acc,
                "mean_f1": f1,
                "score": score,
                "classifier_metrics": {},
                "source": "final_transfer_block",
            }

    # 最后暴力扫描常见字段
    flat = flatten_dict(summary)
    acc_keys = [k for k in flat if "final" in k.lower() and ("acc" in k.lower() or "accuracy" in k.lower())]
    f1_keys = [k for k in flat if "final" in k.lower() and "f1" in k.lower()]

    acc = safe_float(flat[acc_keys[0]]) if acc_keys else 0.0
    f1 = safe_float(flat[f1_keys[0]]) if f1_keys else 0.0
    score = 0.5 * acc + 0.5 * f1

    return {
        "mean_acc": acc,
        "mean_f1": f1,
        "score": score,
        "classifier_metrics": {},
        "source": "fallback_flat_scan",
    }


def main():
    ensure_artifact_dir()

    mask_path = Path("artifacts/feature_mask_miifs_loso_k192_full32_multisource.pkl")
    if not mask_path.exists():
        raise FileNotFoundError(
            f"真实 MIIFS mask 文件不存在: {mask_path}\n"
            f"请先运行特征选择（main_feature_selection.py）生成 mask。"
        )

    # ========== 固定中心参数 ==========
    fixed_params = dict(
        mask_path=mask_path,
        feature_source_mode="mask",
        feature_source_label="MIIFS_Selected_Optimized",
        random_state=42,
        evaluation_protocol="loso",
        max_target_subjects=5,
        max_source_subjects=24,
        source_sample_cap=160,
        mmd_sample_cap=200,
        mmd_prefix_top_k=8,
        transfer_variant="dann",
        target_repeat_grid=(1, 2, 3),
        source_target_ratio_grid=(0.25, 0.5, 0.75, 1.0),
        dann_encoding_dim=64,
        dann_batch_size=64,
        source_positive_ratio_gap_threshold=0.18,
        positive_ratio_weight_strength=0.5,
        use_positive_ratio_source_weighting=True,
    )

    # ========== 小规模搜索空间：12组 ==========
    search_space = {
        "dann_lambda": [0.2, 0.5, 1.0],
        "dann_epochs": [30, 50],
        "gate_repeat_count": [1, 3],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = list(itertools.product(*values))

    all_results = []

    print("\n" + "=" * 100)
    print(f"🚀 开始搜索 DANN 参数，共 {len(combinations)} 组")
    print("=" * 100 + "\n")

    for run_idx, combo in enumerate(combinations, start=1):
        variable_params = dict(zip(keys, combo))

        tag = (
            f"lam{variable_params['dann_lambda']}"
            f"_ep{variable_params['dann_epochs']}"
            f"_gate{variable_params['gate_repeat_count']}"
        )

        summary_path = Path(f"AI/artifacts/dann_search_summary_{tag}.json")
        report_path = Path(f"AI/artifacts/dann_search_report_{tag}.md")

        print("-" * 100)
        print(f"[{run_idx}/{len(combinations)}] 运行参数组: {tag}")
        print(variable_params)

        config_kwargs = dict(fixed_params)
        config_kwargs.update(variable_params)
        config_kwargs["summary_path"] = summary_path
        config_kwargs["report_path"] = report_path

        try:
            config = TransferLearningConfig(**config_kwargs)
            run_transfer_learning_pipeline(config)

            summary = try_load_json(summary_path)
            extracted = extract_score(summary)

            row = {
                "run_idx": run_idx,
                "tag": tag,
                **variable_params,
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "mean_acc": extracted["mean_acc"],
                "mean_f1": extracted["mean_f1"],
                "score": extracted["score"],
                "extract_source": extracted["source"],
                "classifier_metrics": extracted["classifier_metrics"],
                "status": "ok",
            }

            print(
                f"✅ 完成: score={row['score']:.4f}, "
                f"mean_acc={row['mean_acc']:.4f}, mean_f1={row['mean_f1']:.4f}"
            )

        except Exception as e:
            row = {
                "run_idx": run_idx,
                "tag": tag,
                **variable_params,
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "mean_acc": 0.0,
                "mean_f1": 0.0,
                "score": 0.0,
                "extract_source": "error",
                "classifier_metrics": {},
                "status": "error",
                "error": str(e),
            }
            print(f"❌ 失败: {e}")

        all_results.append(row)

    # 排序
    all_results_sorted = sorted(
        all_results,
        key=lambda x: (x["status"] == "ok", x["score"], x["mean_f1"], x["mean_acc"]),
        reverse=True,
    )

    # 保存 JSON
    json_out = Path("AI/artifacts/dann_search_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results_sorted, f, ensure_ascii=False, indent=2)

    # 保存 CSV
    csv_out = Path("AI/artifacts/dann_search_results.csv")
    csv_fields = [
        "run_idx",
        "tag",
        "status",
        "dann_lambda",
        "dann_epochs",
        "gate_repeat_count",
        "mean_acc",
        "mean_f1",
        "score",
        "extract_source",
        "summary_path",
        "report_path",
        "error",
    ]
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_results_sorted:
            csv_row = {k: row.get(k, "") for k in csv_fields}
            writer.writerow(csv_row)

    print("\n" + "=" * 100)
    print("🏁 搜索完成，Top 5 结果：")
    print("=" * 100)

    for i, row in enumerate(all_results_sorted[:5], start=1):
        print(
            f"Top {i}: {row['tag']} | status={row['status']} | "
            f"score={row['score']:.4f} | acc={row['mean_acc']:.4f} | f1={row['mean_f1']:.4f}"
        )

    # 输出最佳参数建议
    best_ok = next((r for r in all_results_sorted if r["status"] == "ok"), None)
    if best_ok:
        print("\n✅ 当前搜索下的最佳参数：")
        print(json.dumps({
            "dann_lambda": best_ok["dann_lambda"],
            "dann_epochs": best_ok["dann_epochs"],
            "gate_repeat_count": best_ok["gate_repeat_count"],
            "mean_acc": best_ok["mean_acc"],
            "mean_f1": best_ok["mean_f1"],
            "score": best_ok["score"],
            "summary_path": best_ok["summary_path"],
            "report_path": best_ok["report_path"],
        }, ensure_ascii=False, indent=2))
    else:
        print("\n⚠️ 没有成功跑完的参数组。")

    print(f"\n结果 JSON: {json_out}")
    print(f"结果 CSV : {csv_out}")


if __name__ == "__main__":
    main()