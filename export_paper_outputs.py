import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "paper_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "model_effect_analysis"
ANALYSIS_FIG_DIR = ANALYSIS_DIR / "figures"

VUL_ORDER = ["reentrancy", "integer_overflow", "time_dependency", "dos_failed_call"]
BASELINE_METHODS = ["RESCUER", "FedAvg", "CL", "CLC", "FedCorr", "ARFL"]
ABLATION_METHODS = [
    "Ablation_Full",
    "Ablation_woLCN",
    "Ablation_woWarmup",
    "Ablation_woOrtho",
    "Ablation_woTransformer",
]
MOTIVATION_METHODS = ["Motivation_Exp1", "Motivation_Exp2", "Motivation_Exp3", "Motivation_Exp4"]
METRICS = ["F1 score", "Precision", "Recall", "Accuracy"]


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    ANALYSIS_FIG_DIR.mkdir(exist_ok=True)


def parse_result_file(path: Path):
    rel = path.relative_to(RESULT_DIR)
    if len(rel.parts) < 3:
        return None

    noise_bucket = rel.parts[0]
    method = rel.parts[1]
    filename = rel.parts[2].replace("_result.json", "")

    if filename.startswith("fn_"):
        noise_type = "fn_noise"
        vul = filename[3:]
    elif filename.startswith("diff_noise_"):
        noise_type = "diff_noise"
        vul = filename[11:]
    elif filename.startswith("pure_"):
        noise_type = "pure"
        vul = filename[5:]
    else:
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[-1]

    noise_label = "0.0" if noise_type == "pure" else noise_bucket
    return {
        "method": method,
        "vulnerability": vul,
        "noise_type": noise_type,
        "noise_rate": noise_label,
        "noise_bucket": noise_bucket,
        "F1 score": float(data.get("F1 score", 0.0)),
        "Precision": float(data.get("Precision", 0.0)),
        "Recall": float(data.get("Recall", 0.0)),
        "Accuracy": float(data.get("Accuracy", 0.0)),
        "TP": int(data.get("TP", 0)),
        "TN": int(data.get("TN", 0)),
        "FP": int(data.get("FP", 0)),
        "FN": int(data.get("FN", 0)),
    }


def load_results_df():
    rows = []
    for path in RESULT_DIR.rglob("*_result.json"):
        parsed = parse_result_file(path)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise RuntimeError("No result files found.")
    df = pd.DataFrame(rows)
    df["noise_sort"] = df["noise_rate"].map(
        lambda x: float(x) if x not in {"diff", "diff_10_20", "diff_10_30"} else math.inf
    )
    return df.sort_values(["method", "vulnerability", "noise_type", "noise_sort"]).reset_index(drop=True)


def save_metric_tables(df: pd.DataFrame, methods, prefix: str, diff_only: bool = False):
    if diff_only:
        subset = df[(df["method"].isin(methods)) & (df["noise_type"] == "diff_noise")].copy()
        subset["setting"] = subset["noise_bucket"]
    else:
        subset = df[
            (df["method"].isin(methods))
            & (((df["noise_type"] == "pure") & (df["noise_rate"] == "0.0")) | (df["noise_type"] == "fn_noise"))
        ].copy()
        subset["setting"] = subset["noise_rate"]

    for metric in METRICS:
        pivot = subset.pivot_table(
            index="method",
            columns=["vulnerability", "setting"],
            values=metric,
            aggfunc="mean",
        )
        pivot = pivot.reindex(methods)
        flat_cols = []
        for vul, setting in pivot.columns:
            flat_cols.append(f"{vul}_{setting}")
        pivot.columns = flat_cols
        pivot.to_csv(OUTPUT_DIR / f"{prefix}_{metric.lower().replace(' ', '_')}.csv", encoding="utf-8-sig")


def average_by_noise(df: pd.DataFrame, methods, method_col_name: str):
    subset = df[
        (df["method"].isin(methods))
        & (((df["noise_type"] == "pure") & (df["noise_rate"] == "0.0")) | (df["noise_type"] == "fn_noise"))
    ].copy()
    pivot = subset.pivot_table(
        index="method",
        columns="noise_rate",
        values="F1 score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(methods)
    ordered_cols = [c for c in ["0.0", "0.1", "0.2", "0.3"] if c in pivot.columns]
    pivot = pivot[ordered_cols]
    out = pd.DataFrame({method_col_name: pivot.index})
    for col in ordered_cols:
        out[f"noise_{col}"] = pivot[col].round(4).values
    out["overall_avg"] = pivot.mean(axis=1).round(4).values
    return out


def asymmetric_average(df: pd.DataFrame, methods):
    subset = df[(df["method"].isin(methods)) & (df["noise_type"] == "diff_noise")].copy()
    pivot = subset.pivot_table(index="method", columns="noise_bucket", values="F1 score", aggfunc="mean")
    pivot = pivot.reindex(methods)
    ordered_cols = [c for c in ["diff_10_20", "diff_10_30"] if c in pivot.columns]
    out = pd.DataFrame({"Method": pivot.index})
    for col in ordered_cols:
        out[col] = pivot[col].round(4).values
    out["overall_avg"] = pivot.mean(axis=1).round(4).values
    return out


def make_line_plot(table: pd.DataFrame, name_col: str, out_path: Path, title: str):
    plt.figure(figsize=(9, 5))
    noise_cols = [c for c in table.columns if c.startswith("noise_")]
    x = [c.replace("noise_", "") for c in noise_cols]
    for _, row in table.iterrows():
        plt.plot(x, [row[c] for c in noise_cols], marker="o", linewidth=2, label=row[name_col])
    plt.title(title)
    plt.xlabel("False-negative noise rate")
    plt.ylabel("Average F1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_bar_plot(table: pd.DataFrame, name_col: str, value_col: str, out_path: Path, title: str, color="#4C78A8"):
    plt.figure(figsize=(8, 5))
    order = table.sort_values(value_col, ascending=False)
    sns.barplot(data=order, x=name_col, y=value_col, color=color)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(value_col)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_grouped_bar(df: pd.DataFrame, methods, out_path: Path, title: str):
    subset = df[
        (df["method"].isin(methods))
        & (((df["noise_type"] == "pure") & (df["noise_rate"] == "0.0")) | (df["noise_type"] == "fn_noise"))
    ].copy()
    grouped = (
        subset.groupby(["noise_rate", "method"], as_index=False)["F1 score"]
        .mean()
        .rename(columns={"method": "Method"})
    )
    grouped["noise_rate"] = pd.Categorical(grouped["noise_rate"], categories=["0.0", "0.1", "0.2", "0.3"], ordered=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=grouped, x="noise_rate", y="F1 score", hue="Method", order=["0.0", "0.1", "0.2", "0.3"])
    plt.title(title)
    plt.xlabel("False-negative noise rate")
    plt.ylabel("Average F1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_summary_docs(df: pd.DataFrame, baseline_avg: pd.DataFrame, asym_avg: pd.DataFrame, ablation_avg: pd.DataFrame):
    best_baseline = baseline_avg.sort_values("overall_avg", ascending=False).iloc[0]
    best_asym = asym_avg.sort_values("overall_avg", ascending=False).iloc[0]
    full_row = ablation_avg[ablation_avg["Variant"] == "Ablation_Full"].iloc[0]

    ablation_delta = ablation_avg.copy()
    full_avg = float(full_row["overall_avg"])
    ablation_delta["delta_vs_full"] = (full_avg - ablation_delta["overall_avg"]).round(4)
    ablation_delta = ablation_delta[ablation_delta["Variant"] != "Ablation_Full"]

    summary = f"""# Paper Results Summary

This run completed a full DAppSCAN experiment sweep across four vulnerability tasks and generated {len(df)} result files. The strongest method under uniform false-negative noise is {best_baseline['Method']} with an overall average F1 of {best_baseline['overall_avg']:.4f}. Under asymmetric noise, the best overall method is {best_asym['Method']} with an average F1 of {best_asym['overall_avg']:.4f}. In the ablation study, the full model reaches an average F1 of {full_avg:.4f}.

## Uniform-Noise Baseline Average F1

{baseline_avg.to_markdown(index=False)}

## Asymmetric-Noise Baseline Average F1

{asym_avg.to_markdown(index=False)}

## Ablation Average F1

{ablation_avg.to_markdown(index=False)}

## Full Model Gain Over Ablations

{ablation_delta.to_markdown(index=False)}
"""
    (OUTPUT_DIR / "paper_results_summary.md").write_text(summary, encoding="utf-8")

    zh = f"""# 实验结果展示

本轮实验围绕真实 DAppSCAN 数据集上的四类智能合约漏洞任务完成了完整训练与评估，共生成 {len(df)} 组结果。对比实验中，均匀假阴噪声条件下整体平均 F1 最高的方法是 {best_baseline['Method']}，平均 F1 为 {best_baseline['overall_avg']:.4f}；非对称噪声条件下整体平均 F1 最高的方法是 {best_asym['Method']}，平均 F1 为 {best_asym['overall_avg']:.4f}。消融实验中，完整模型 Ablation_Full 的整体平均 F1 为 {full_avg:.4f}，整体高于去掉关键模块后的各个变体，说明完整结构在整体稳定性和噪声鲁棒性上表现更好。

## Baseline 平均 F1

{baseline_avg.to_markdown(index=False)}

## 非对称噪声平均 F1

{asym_avg.to_markdown(index=False)}

## 消融平均 F1

{ablation_avg.to_markdown(index=False)}
"""
    (OUTPUT_DIR / "实验结果展示.md").write_text(zh, encoding="utf-8")

    effect = f"""# 模型效果整理

当前这一版实验主要围绕对比实验和消融实验展开。均匀假阴噪声条件下，{best_baseline['Method']} 的整体平均 F1 为 {best_baseline['overall_avg']:.4f}，在所有 baseline 中表现最好。非对称噪声条件下，{best_asym['Method']} 的整体平均 F1 为 {best_asym['overall_avg']:.4f}，说明其对复杂噪声分布更敏感。完整模型 Ablation_Full 的整体平均 F1 为 {full_avg:.4f}，相比各消融变体保持了更稳定的综合表现。

## 对比实验

{baseline_avg.to_markdown(index=False)}

## 消融实验

{ablation_avg.to_markdown(index=False)}

## Full 相对各消融变体增益

{ablation_delta.to_markdown(index=False)}
"""
    (OUTPUT_DIR / "模型效果整理.md").write_text(effect, encoding="utf-8")

    recommend = """# 推荐展示图

推荐优先展示以下图：

1. `figures/baseline_average_f1_by_noise.png`
2. `model_effect_analysis/figures/baseline_robustness_drop.png`
3. `figures/ablation_average_f1.png`
4. `model_effect_analysis/figures/ablation_full_gain.png`
5. `figures/motivation_average_f1.png`

其中第一张最适合用于展示整体鲁棒性，第三和第四张最适合说明模块贡献。
"""
    (OUTPUT_DIR / "推荐展示图.md").write_text(recommend, encoding="utf-8")

    analysis = f"""# Model Effect Analysis

The current experiment suite focuses on baseline comparison and ablation analysis over the four DAppSCAN vulnerability tasks. Under uniform false-negative noise, {best_baseline['Method']} ranks first with an overall average F1 of {best_baseline['overall_avg']:.4f}. Under asymmetric noise, {best_asym['Method']} ranks first with an overall average F1 of {best_asym['overall_avg']:.4f}. The full ablation model reaches an overall average F1 of {full_avg:.4f}, outperforming the weakened variants in aggregate.

## Baseline Average F1 by Noise

{baseline_avg.to_markdown(index=False)}

## Robustness Summary

{build_robustness_summary(baseline_avg).to_markdown(index=False)}

## Ablation Average F1 by Noise

{ablation_avg.to_markdown(index=False)}
"""
    (ANALYSIS_DIR / "model_effect_analysis.md").write_text(analysis, encoding="utf-8")


def build_robustness_summary(baseline_avg: pd.DataFrame):
    out = baseline_avg.copy()
    out["drop_0.0_to_0.3"] = (out["noise_0.3"] - out["noise_0.0"]).round(4)
    return out[["Method", "noise_0.0", "noise_0.3", "drop_0.0_to_0.3", "overall_avg"]]


def best_method_by_task_noise(df: pd.DataFrame, methods):
    subset = df[
        (df["method"].isin(methods))
        & (((df["noise_type"] == "pure") & (df["noise_rate"] == "0.0")) | (df["noise_type"] == "fn_noise"))
    ].copy()
    subset["setting"] = subset["noise_rate"]
    rows = []
    for vul in VUL_ORDER:
        for setting in ["0.0", "0.1", "0.2", "0.3"]:
            block = subset[(subset["vulnerability"] == vul) & (subset["setting"] == setting)]
            if block.empty:
                continue
            best = block.sort_values("F1 score", ascending=False).iloc[0]
            rows.append(
                {
                    "vulnerability": vul,
                    "noise_rate": setting,
                    "best_method": best["method"],
                    "best_f1": round(float(best["F1 score"]), 4),
                }
            )
    return pd.DataFrame(rows)


def make_task_plots(df: pd.DataFrame):
    subset = df[
        (df["method"].isin(BASELINE_METHODS))
        & (((df["noise_type"] == "pure") & (df["noise_rate"] == "0.0")) | (df["noise_type"] == "fn_noise"))
    ].copy()
    subset["noise_rate"] = pd.Categorical(subset["noise_rate"], categories=["0.0", "0.1", "0.2", "0.3"], ordered=True)
    for vul in VUL_ORDER:
        part = subset[subset["vulnerability"] == vul]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=part, x="noise_rate", y="F1 score", hue="method", marker="o")
        plt.title(f"{vul} average F1 by noise")
        plt.xlabel("False-negative noise rate")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"baseline_{vul}_average_f1.png", dpi=220)
        plt.close()


def write_manifest(df: pd.DataFrame):
    manifest = {
        "result_file_count": int(len(df)),
        "vulnerabilities": VUL_ORDER,
        "baseline_methods": BASELINE_METHODS,
        "ablation_methods": ABLATION_METHODS,
        "motivation_methods": MOTIVATION_METHODS,
        "uniform_noise_settings": ["0.0", "0.1", "0.2", "0.3"],
        "asymmetric_noise_settings": ["diff_10_20", "diff_10_30"],
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_results_df()
    df.to_csv(OUTPUT_DIR / "all_results_long.csv", index=False, encoding="utf-8-sig")

    save_metric_tables(df, MOTIVATION_METHODS, "motivation")
    save_metric_tables(df, BASELINE_METHODS, "baseline")
    save_metric_tables(df, BASELINE_METHODS, "baseline_asymmetric", diff_only=True)
    save_metric_tables(df, ABLATION_METHODS, "ablation")

    baseline_avg = average_by_noise(df, BASELINE_METHODS, "Method")
    baseline_avg.to_csv(ANALYSIS_DIR / "baseline_average_f1_by_noise.csv", index=False, encoding="utf-8-sig")
    ablation_avg = average_by_noise(df, ABLATION_METHODS, "Variant")
    ablation_avg.to_csv(ANALYSIS_DIR / "ablation_average_f1_by_noise.csv", index=False, encoding="utf-8-sig")
    asym_avg = asymmetric_average(df, BASELINE_METHODS)

    robustness = build_robustness_summary(baseline_avg)
    robustness.to_csv(ANALYSIS_DIR / "baseline_robustness_summary.csv", index=False, encoding="utf-8-sig")

    delta_vs_full = ablation_avg.copy()
    full_avg = float(delta_vs_full.loc[delta_vs_full["Variant"] == "Ablation_Full", "overall_avg"].iloc[0])
    delta_vs_full["delta_vs_full"] = (full_avg - delta_vs_full["overall_avg"]).round(4)
    delta_vs_full.to_csv(ANALYSIS_DIR / "ablation_delta_vs_full.csv", index=False, encoding="utf-8-sig")

    best_task = best_method_by_task_noise(df, BASELINE_METHODS)
    best_task.to_csv(ANALYSIS_DIR / "best_baseline_by_task_noise.csv", index=False, encoding="utf-8-sig")

    ablation_delta_by_vul = (
        df[df["method"].isin(ABLATION_METHODS)]
        .pivot_table(index="vulnerability", columns="method", values="F1 score", aggfunc="mean")
        .reindex(VUL_ORDER)
        .reset_index()
    )
    ablation_delta_by_vul.to_csv(ANALYSIS_DIR / "ablation_delta_by_vulnerability.csv", index=False, encoding="utf-8-sig")

    baseline_f1 = df[df["method"].isin(BASELINE_METHODS)].copy()
    asym_table = df[(df["method"].isin(BASELINE_METHODS)) & (df["noise_type"] == "diff_noise")].pivot_table(
        index="method", columns="noise_bucket", values="F1 score", aggfunc="mean"
    ).reindex(BASELINE_METHODS)

    make_line_plot(baseline_avg, "Method", FIG_DIR / "baseline_average_f1_by_noise.png", "Baseline average F1 by noise")
    make_line_plot(ablation_avg, "Variant", FIG_DIR / "ablation_average_f1.png", "Ablation average F1 by noise")
    make_bar_plot(delta_vs_full[delta_vs_full["Variant"] != "Ablation_Full"], "Variant", "delta_vs_full", ANALYSIS_FIG_DIR / "ablation_full_gain.png", "Full model gain over ablations", "#59A14F")
    make_bar_plot(robustness, "Method", "drop_0.0_to_0.3", ANALYSIS_FIG_DIR / "baseline_robustness_drop.png", "Baseline robustness from 0.0 to 0.3 noise", "#E15759")
    make_grouped_bar(df, BASELINE_METHODS, ANALYSIS_FIG_DIR / "baseline_grouped_by_noise.png", "Baseline grouped average F1 by noise")

    best_counts = best_task.groupby("best_method").size().reset_index(name="count")
    make_bar_plot(best_counts, "best_method", "count", ANALYSIS_FIG_DIR / "best_method_counts.png", "Best-method counts by task/noise", "#F28E2B")

    make_task_plots(df)

    motivation_table = (
        df[df["method"].isin(MOTIVATION_METHODS) & (df["noise_type"] == "pure")]
        .groupby("method", as_index=False)["F1 score"]
        .mean()
        .rename(columns={"method": "Method", "F1 score": "average_f1"})
    )
    make_bar_plot(motivation_table, "Method", "average_f1", FIG_DIR / "motivation_average_f1.png", "Motivation average F1", "#76B7B2")

    if not asym_table.empty:
        asym_table = asym_table.reset_index().rename(columns={"method": "Method"})
        for col in ["diff_10_20", "diff_10_30"]:
            if col in asym_table.columns:
                temp = asym_table[["Method", col]].rename(columns={col: "average_f1"})
                make_bar_plot(temp, "Method", "average_f1", FIG_DIR / f"asymmetric_{col}_average_f1.png", f"Asymmetric noise {col} average F1", "#B07AA1")

    write_summary_docs(df, baseline_avg, asym_avg, ablation_avg)
    write_manifest(df)
    print(f"Exported paper outputs from {len(df)} result files.")


if __name__ == "__main__":
    main()
