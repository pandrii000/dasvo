import json
from itertools import product
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from dasvo.datasets import list_sequences
from dasvo.settings import SETTINGS
from scripts.table_utils import format_mean_std_bold_best

plt.style.use("seaborn-v0_8-whitegrid")


def _finite_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _metric_stats(values: pd.Series, *, n_boot: int, confidence: float, seed: int) -> dict[str, float | int]:
    finite = _finite_array(values)
    n = int(len(finite))
    if n == 0:
        return {
            "finite_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    median = float(np.median(finite))
    iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    se = float(std / np.sqrt(n)) if n > 1 else 0.0

    if n_boot > 0 and n > 1:
        rng = np.random.default_rng(seed)
        boot_means = rng.choice(finite, size=(n_boot, n), replace=True).mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
        ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        "finite_count": n,
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": iqr,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _format_mean_std(row: pd.Series, prefix: str) -> str:
    mean = row[f"{prefix}_mean"]
    std = row[f"{prefix}_std"]
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _format_median_iqr(row: pd.Series, prefix: str) -> str:
    median = row[f"{prefix}_median"]
    iqr = row[f"{prefix}_iqr"]
    if pd.isna(median):
        return "--"
    return f"{median:.3f} [{iqr:.3f}]"


def _format_ci(row: pd.Series, prefix: str) -> str:
    lower = row[f"{prefix}_ci_lower"]
    upper = row[f"{prefix}_ci_upper"]
    if pd.isna(lower) or pd.isna(upper):
        return "--"
    return f"[{lower:.3f}, {upper:.3f}]"


def _write_detailed_baseline_table(df: pd.DataFrame, output_dir: Path) -> None:
    baseline = df[df["degradation"] == "none"].sort_values(["frontend", "backend", "frame_stride"]).copy()
    if baseline.empty:
        return

    metric_specs = [
        ("ATE", "ate", "m"),
        ("RPE", "rpe", "m/frame"),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\caption{Baseline ATE/RPE summary with finite-only statistics and 95\\% bootstrap confidence intervals. All baseline aggregates contain 32 finite sequence-level estimates.}",
        "\\label{tab:metrics_detailed}",
        "\\begin{adjustwidth}{-\\extralength}{0cm}",
        "\\centering",
        "\\footnotesize",
    ]

    for metric_label, prefix, unit in metric_specs:
        lines.extend(
            [
                f"\\textbf{{{metric_label} ({unit})}}\\\\[2pt]",
                "\\begin{tabular}{llrlll}",
                "\\toprule",
                "frontend & backend & stride & mean $\\pm$ std & median [IQR] & 95\\% CI \\\\",
                "\\midrule",
            ]
        )
        for _, row in baseline.iterrows():
            lines.append(
                " & ".join(
                    [
                        str(row["frontend"]),
                        str(row["backend"]),
                        str(int(row["frame_stride"])),
                        _format_mean_std(row, prefix=prefix),
                        _format_median_iqr(row, prefix=prefix),
                        _format_ci(row, prefix=prefix),
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\vspace{6pt}"])

    lines.extend(["\\end{adjustwidth}", "\\end{table}", ""])
    (output_dir / "vo_metrics_summary_detailed.tex").write_text("\n".join(lines))


def generate_tables(df: pd.DataFrame, output_dir: Path) -> None:
    # Table 1: Baseline Performance across Motion Scales (Strides)
    baseline_df = df[df["degradation"] == "none"]
    pivot_baseline = format_mean_std_bold_best(
        baseline_df,
        index_col="frame_stride",
        columns_cols=["frontend", "backend"],
        mean_col="ate_mean",
        std_col="ate_std",
    )
    pivot_baseline.to_latex(
        output_dir / "vo_metrics_summary.tex",
        escape=False,
        position="H",
        caption="Baseline ATE (m) across Frame Strides.",
        label="tab:baseline_ate",
    )
    
    # Table 2: Robustness to Severe Visual Degradations
    robust_df = df[df["frame_stride"] == 4]
    pivot_robust = format_mean_std_bold_best(
        robust_df,
        index_col="degradation",
        columns_cols=["frontend", "backend"],
        mean_col="ate_mean",
        std_col="ate_std",
    )
    pivot_robust.to_latex(
        output_dir / "robustness_metrics.tex",
        escape=False,
        caption="Robustness to Visual Degradations (ATE in m, Stride 4).",
        label="tab:robustness_ate",
    )
    
    # Table 3: Baseline Relative Pose Error (RPE) for Local Stability.
    # Keep symmetric with Table 1 (all strides), so reviewers can compare ATE/RPE row-by-row.
    pivot_rpe = format_mean_std_bold_best(
        baseline_df,
        index_col="frame_stride",
        columns_cols=["frontend", "backend"],
        mean_col="rpe_mean",
        std_col="rpe_std",
    )
    pivot_rpe.to_latex(
        output_dir / "vo_metrics_by_sequence.tex",
        escape=False,
        caption="Baseline RPE (m/frame) for Local Stability.",
        label="tab:rpe_stability",
    )

    # Table 3b: PnP-only ATE under Sim(3) alignment.
    # Sim(3) absorbs metric scale so the contrast with the SE(3) PnP ATE in
    # Table 1 isolates the trajectory-shape component of the depth-assisted
    # advantage. Essential is already Sim(3)-aligned in Table 1 by design and
    # is therefore omitted here to avoid implying a separate Sim3 measurement.
    pnp_baseline = baseline_df[baseline_df["backend"] == "pnp"]
    if not pnp_baseline.empty and pnp_baseline["ate_sim3_mean"].notna().any():
        pivot_sim3 = format_mean_std_bold_best(
            pnp_baseline,
            index_col="frame_stride",
            columns_cols=["frontend"],
            mean_col="ate_sim3_mean",
            std_col="ate_sim3_std",
        )
        pivot_sim3.to_latex(
            output_dir / "vo_metrics_summary_ate_sim3.tex",
            escape=False,
            caption=(
                "PnP ATE (m) under Sim(3) alignment across Frame Strides. "
                "Sim(3) absorbs metric scale, so the contrast with Table~\\ref{tab:baseline_ate} "
                "isolates the trajectory-shape component of the PnP advantage from metric-scale drift."
            ),
            label="tab:baseline_ate_sim3",
        )

    # Table 3c: Rotational RPE (deg/frame).
    if "rpe_rot_mean" in baseline_df.columns and baseline_df["rpe_rot_mean"].notna().any():
        pivot_rpe_rot = format_mean_std_bold_best(
            baseline_df,
            index_col="frame_stride",
            columns_cols=["frontend", "backend"],
            mean_col="rpe_rot_mean",
            std_col="rpe_rot_std",
        )
        pivot_rpe_rot.to_latex(
            output_dir / "vo_metrics_rotational_rpe.tex",
            escape=False,
            caption="Baseline Rotational RPE (deg/frame) for Angular Stability.",
            label="tab:rpe_rotational",
        )

    # The combined ATE/RPE detailed table is superseded by the per-metric
    # detailed tables produced by `wave1_revision_metrics.py` (`make revision`).
    # Remove any stale copy so `make assets` does not re-stage it.
    legacy_detailed = output_dir / "vo_metrics_summary_detailed.tex"
    if legacy_detailed.exists():
        legacy_detailed.unlink()
        logger.info(f"Removed legacy combined detailed table {legacy_detailed}")

    logger.info("Generated LaTeX tables.")


def generate_figures(df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = SETTINGS.paths.figures_root
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 2: ATE vs. Frame Stride (Line Chart)
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df[df["degradation"].isin(["none", "blur_heavy"])],
        x="frame_stride",
        y="ate_mean",
        hue="backend",
        style="degradation",
        markers=True,
        dashes=True,
    )
    plt.title("ATE vs. Frame Stride")
    plt.xlabel("Frame Stride")
    plt.ylabel("ATE (m)")
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_stride_degradation.png", dpi=300)
    plt.close()
    
    # Figure 3: RPE Distribution across Degradations (Grouped Bar Chart)
    plt.figure(figsize=(12, 6))
    stride2_df = df[df["frame_stride"] == 2]
    if not stride2_df.empty:
        sns.barplot(
            data=stride2_df,
            x="degradation",
            y="rpe_mean",
            hue="backend",
        )
        plt.title("RPE Distribution across Degradations (Stride 2)")
        plt.xlabel("Degradation")
        plt.ylabel("RPE (m/frame)")
        plt.tight_layout()
        plt.savefig(figures_dir / "rpe_distribution.png", dpi=300)
    plt.close()
    
    # Figure 4: Radar/Spider Chart of Overall Robustness Profile
    stride2_df = df[df["frame_stride"] == 2]
    if not stride2_df.empty:
        categories = stride2_df["degradation"].unique()
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for backend in ["essential", "pnp"]:
            backend_df = stride2_df[stride2_df["backend"] == backend].groupby("degradation")["ate_mean"].mean()
            # Inverse ATE for radar chart (larger is better)
            values = [1.0 / (backend_df.get(cat, np.nan) + 1e-6) for cat in categories]
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=backend)
            ax.fill(angles, values, alpha=0.25)
            
        plt.xticks(angles[:-1], categories)
        plt.title("Robustness Profile (Inverse ATE at Stride 2)")
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(figures_dir / "robustness_radar.png", dpi=300)
    plt.close()
    
    logger.info("Generated analytical figures.")


def generate_trajectory_figure() -> None:
    # Figure 1: Trajectory Comparison under Degradation (Top-Down 2D)
    # We will pick a specific sequence, e.g. abandonedfactory_easy_p011, stride 4, blur_mild
    seq_id = "abandonedfactory_easy_p011"
    stride = "4"
    deg = "blur_mild"
    
    gt_file = SETTINGS.paths.data_root / "abandonedfactory" / "Easy" / "P011" / "pose_left.txt"
    if not gt_file.exists():
        logger.warning(f"GT file {gt_file} not found. Skipping trajectory figure.")
        return
        
    try:
        gt = np.loadtxt(gt_file)[:, :3]
        gt = gt[::int(stride)]
    except Exception as e:
        logger.warning(f"Failed to load GT: {e}")
        return
        
    from dasvo.evaluation import align_umeyama
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    idx = 0
    for fe in ["klt", "orb"]:
        ax = axes[idx]
        ax.plot(gt[:, 0], gt[:, 1], "k--", label="Ground Truth", linewidth=2)
        
        for be, color in zip(["essential", "pnp"], ["#1f77b4", "#ff7f0e"]):
            traj_file = SETTINGS.paths.outputs_root / fe / be / deg / stride / f"{seq_id}_traj.txt"
            if traj_file.exists():
                est = np.loadtxt(traj_file)
                if len(est) > 0:
                    min_len = min(len(gt), len(est))
                    gt_trunc = gt[:min_len]
                    est_trunc = est[:min_len]
                    with_scale = (be in SETTINGS.evaluation.monocular_scale_align_backends)
                    est_aligned, _ = align_umeyama(gt_trunc, est_trunc, with_scale=with_scale)
                    ax.plot(est_aligned[:, 0], est_aligned[:, 1], label=f"{be.upper()}", color=color)
        
        ax.set_title(f"Frontend: {fe.upper()} ({deg}, stride {stride})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.axis("equal")
        idx += 1
            
    plt.tight_layout()
    figures_dir = SETTINGS.paths.figures_root
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "trajectory_comparison.png", dpi=300)
    plt.close()
    logger.info("Generated trajectory comparison figure.")


def generate_trajectory_degradation_grid() -> None:
    seq_id = "abandonedfactory_easy_p011"
    stride = "4"
    
    gt_file = SETTINGS.paths.data_root / "abandonedfactory" / "Easy" / "P011" / "pose_left.txt"
    if not gt_file.exists():
        logger.warning(f"GT file {gt_file} not found. Skipping grid trajectory figure.")
        return
        
    try:
        gt = np.loadtxt(gt_file)[:, :3]
        gt = gt[::int(stride)]
    except Exception as e:
        logger.warning(f"Failed to load GT: {e}")
        return

    from dasvo.evaluation import align_umeyama

    degradations = ["none", "blur_mild", "blur_heavy", "gaussian_noise", "jpeg_low", "low_light"]
    degradations = [d for d in degradations if d in SETTINGS.experiment.degradations]
    
    pipelines = ["klt", "orb"]
    
    if not degradations:
        return

    fig, axes = plt.subplots(len(degradations), len(pipelines), figsize=(6 * len(pipelines), 4 * len(degradations)))
    
    if len(degradations) == 1 and len(pipelines) == 1:
        axes = np.array([[axes]])
    elif len(degradations) == 1:
        axes = np.expand_dims(axes, 0)
    elif len(pipelines) == 1:
        axes = np.expand_dims(axes, 1)
        
    for r, deg in enumerate(degradations):
        for c, fe in enumerate(pipelines):
            ax = axes[r, c]
            ax.plot(gt[:, 0], gt[:, 1], "k--", label="Ground Truth", alpha=0.5, linewidth=2)
            
            title = f"{fe.upper()} | {deg}"
            
            for be, color in zip(["essential", "pnp"], ["#1f77b4", "#ff7f0e"]):
                traj_file = SETTINGS.paths.outputs_root / fe / be / deg / stride / f"{seq_id}_traj.txt"
                ate_val = None
                
                res_file = SETTINGS.paths.tables_root / fe / be / deg / stride / f"{seq_id}_results.json"
                if res_file.exists():
                    try:
                        with open(res_file, "r") as f:
                            res_data = json.load(f)
                            ate_val = res_data.get("ate")
                    except Exception:
                        pass
                
                if traj_file.exists():
                    try:
                        est = np.loadtxt(traj_file)
                        if len(est) > 0:
                            min_len = min(len(gt), len(est))
                            gt_trunc = gt[:min_len]
                            est_trunc = est[:min_len]
                            with_scale = (be in SETTINGS.evaluation.monocular_scale_align_backends)
                            est_aligned, _ = align_umeyama(gt_trunc, est_trunc, with_scale=with_scale)
                            ax.plot(est_aligned[:, 0], est_aligned[:, 1], label=f"{be.upper()}", color=color)
                    except Exception:
                        pass
                
                if ate_val is not None:
                    title += f"\n{be.upper()} ATE: {ate_val:.3f}m"
                else:
                    title += f"\n{be.upper()} ATE: N/A"
                
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.axis("equal")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
                
    plt.tight_layout()
    figures_dir = SETTINGS.paths.figures_root
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "trajectory_degradation_grid.png", dpi=300)
    plt.close()
    logger.info("Generated trajectory degradation grid figure.")


def generate_comments_placeholders(output_dir: Path, figures_dir: Path) -> None:
    comments_dir = output_dir / "comments"
    comments_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "aggregated_results.csv"
    
    placeholders = {
        "vo_metrics_summary": f"Table 1: Baseline Performance across Motion Scales (Strides).\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "robustness_metrics": f"Table 2: Robustness to Severe Visual Degradations.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "vo_metrics_by_sequence": f"Table 3: Relative Pose Error (RPE) for Local Stability.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "trajectory_comparison": f"Figure 1: Trajectory Comparison under Degradation (Top-Down 2D).\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "trajectory_degradation_grid": f"Figure: Trajectory Degradation Grid for a single sequence.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "robustness_stride_degradation": f"Figure 2: ATE vs. Frame Stride (Line Chart).\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "rpe_distribution": f"Figure 3: RPE Distribution across Degradations (Grouped Bar Chart).\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "robustness_radar": f"Figure 4: Radar/Spider Chart of Overall Robustness Profile.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
    }
    
    for name, content in placeholders.items():
        (comments_dir / f"{name}_comments.md").write_text(content)
        
    logger.info("Generated comment placeholders.")


def copy_to_publication() -> None:
    import shutil

    pub_tables = SETTINGS.paths.paper_tables_root
    pub_figures = SETTINGS.paths.paper_figures_root
    pub_tables.mkdir(parents=True, exist_ok=True)
    pub_figures.mkdir(parents=True, exist_ok=True)

    # Retired artifacts: combined detailed table is superseded by the per-metric
    # detailed tables produced by `wave1_revision_metrics.py`. Remove any stale
    # copies before mirroring so they cannot reappear after a re-run.
    for stale_name in ("vo_metrics_summary_detailed.tex",):
        for root in (SETTINGS.paths.tables_root, pub_tables):
            stale = root / stale_name
            if stale.exists():
                stale.unlink()
                logger.info(f"Removed retired artifact {stale}")

    # Copy tables
    for tex_file in SETTINGS.paths.tables_root.glob("*.tex"):
        shutil.copy(tex_file, pub_tables / tex_file.name)
    for csv_file in SETTINGS.paths.tables_root.glob("*.csv"):
        shutil.copy(csv_file, pub_tables / csv_file.name)
        
    # Copy figures
    for png_file in SETTINGS.paths.figures_root.glob("*.png"):
        shutil.copy(png_file, pub_figures / png_file.name)
        
    # Copy comments
    pub_comments = SETTINGS.paths.paper_project_root / "comments"
    pub_comments.mkdir(parents=True, exist_ok=True)
    comments_dir = SETTINGS.paths.tables_root / "comments"
    if comments_dir.exists():
        for md_file in comments_dir.glob("*.md"):
            shutil.copy(md_file, pub_comments / md_file.name)
            
    logger.info("Copied assets to publication directory.")


def main() -> None:
    results = []
    
    jobs = []
    for fe in SETTINGS.experiment.frontends:
        for be in SETTINGS.experiment.backends:
            for deg in SETTINGS.experiment.degradations:
                for stride in SETTINGS.experiment.frame_strides:
                    eval_dir = SETTINGS.paths.tables_root / fe / be / deg / str(stride)
                    if not eval_dir.exists():
                        continue
                    jobs.extend(list(eval_dir.glob("*_results.json")))
                        
    for result_file in tqdm(jobs, desc="Aggregating results"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.error(f"Failed to read {result_file}: {e}")
                            
    sequences = [sequence.sequence_id for sequence in list_sequences()]
    key_cols = ["frontend", "backend", "degradation", "frame_stride", "sequence"]

    expected_df = pd.DataFrame(
        [
            {
                "frontend": fe,
                "backend": be,
                "degradation": deg,
                "frame_stride": stride,
                "sequence": sequence_id,
            }
            for fe, be, deg, stride, sequence_id in product(
                SETTINGS.experiment.frontends,
                SETTINGS.experiment.backends,
                SETTINGS.experiment.degradations,
                SETTINGS.experiment.frame_strides,
                sequences,
            )
        ]
    )

    if results:
        raw_df = pd.DataFrame(results)
        raw_df["frame_stride"] = raw_df["frame_stride"].astype(int)
        raw_df = raw_df.drop_duplicates(key_cols, keep="last")
        df = expected_df.merge(raw_df, on=key_cols, how="left")
    else:
        logger.warning("No result JSON files found. Writing expected-grid failure summary.")
        df = expected_df.copy()

    for metric in ("ate", "rpe", "ate_sim3", "rpe_rot"):
        if metric not in df:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if "failed" not in df:
        df["failed"] = False

    df["observed"] = df["ate"].notna() | df["rpe"].notna()
    df["ate_finite"] = np.isfinite(df["ate"].to_numpy(dtype=float))
    df["rpe_finite"] = np.isfinite(df["rpe"].to_numpy(dtype=float))
    df["failed"] = (
        df["failed"].fillna(False).astype(bool)
        | ~df["observed"]
        | ~df["ate_finite"]
        | ~df["rpe_finite"]
    )

    group_cols = ["frontend", "backend", "degradation", "frame_stride"]
    agg_rows = []
    for group_index, (group_key, group) in enumerate(df.groupby(group_cols, sort=True)):
        ate_stats = _metric_stats(
            group["ate"],
            n_boot=SETTINGS.evaluation.bootstrap_samples,
            confidence=SETTINGS.evaluation.confidence_level,
            seed=SETTINGS.evaluation.bootstrap_seed + group_index,
        )
        rpe_stats = _metric_stats(
            group["rpe"],
            n_boot=SETTINGS.evaluation.bootstrap_samples,
            confidence=SETTINGS.evaluation.confidence_level,
            seed=SETTINGS.evaluation.bootstrap_seed + 10000 + group_index,
        )
        ate_sim3_stats = _metric_stats(
            group["ate_sim3"],
            n_boot=SETTINGS.evaluation.bootstrap_samples,
            confidence=SETTINGS.evaluation.confidence_level,
            seed=SETTINGS.evaluation.bootstrap_seed + 20000 + group_index,
        )
        rpe_rot_stats = _metric_stats(
            group["rpe_rot"],
            n_boot=SETTINGS.evaluation.bootstrap_samples,
            confidence=SETTINGS.evaluation.confidence_level,
            seed=SETTINGS.evaluation.bootstrap_seed + 30000 + group_index,
        )
        sequences_expected = int(len(group))
        sequences_observed = int(group["observed"].sum())
        failure_count = int(group["failed"].sum())

        row = dict(zip(group_cols, group_key, strict=True))
        row.update(
            {
                "sequences_expected": sequences_expected,
                "sequences_observed": sequences_observed,
                "failure_count": failure_count,
                "failure_rate": float(failure_count / sequences_expected) if sequences_expected else float("nan"),
            }
        )
        for prefix, stats in (
            ("ate", ate_stats),
            ("rpe", rpe_stats),
            ("ate_sim3", ate_sim3_stats),
            ("rpe_rot", rpe_rot_stats),
        ):
            for name, value in stats.items():
                row[f"{prefix}_{name}"] = value
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    
    output_dir = SETTINGS.paths.tables_root
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sequence_csv_path = output_dir / "per_sequence_results.csv"
    df.to_csv(per_sequence_csv_path, index=False)
    logger.info(f"Saved per-sequence results to {per_sequence_csv_path}")
    
    csv_path = output_dir / "aggregated_results.csv"
    agg_df.to_csv(csv_path, index=False)
    logger.info(f"Saved aggregated results to {csv_path}")
    
    md_path = output_dir / "aggregated_results.md"
    agg_df.to_markdown(md_path, index=False)
    logger.info(f"Saved aggregated results markdown to {md_path}")
    
    generate_tables(agg_df, output_dir)
    generate_figures(agg_df, output_dir)
    generate_trajectory_figure()
    generate_trajectory_degradation_grid()
    generate_comments_placeholders(output_dir, SETTINGS.paths.figures_root)
    
    if SETTINGS.publication.copy_to_paper_assets:
        copy_to_publication()


if __name__ == "__main__":
    fire.Fire(main)
