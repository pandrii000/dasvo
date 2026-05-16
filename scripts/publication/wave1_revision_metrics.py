"""
Wave 1 revision post-processing: normalized ATE, coverage, Holm-Bonferroni.

Reads existing per-sequence ``_results.json`` artifacts and ground-truth pose files,
computes additional metrics requested in the round-2 peer review (normalized ATE
relative to GT trajectory length, finite-aggregation coverage per cell, and
multiple-comparison-corrected Wilcoxon p-values across the baseline grid), and
writes updated LaTeX assets without touching the main evaluation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon

from dasvo.datasets import list_sequences
from dasvo.settings import SETTINGS


STRIDES = tuple(SETTINGS.experiment.frame_strides)
FRONTENDS = tuple(SETTINGS.experiment.frontends)
BACKENDS = tuple(SETTINGS.experiment.backends)
N_EXPECTED_SEQUENCES = 32


@dataclass(frozen=True)
class CellKey:
    frontend: str
    backend: str
    degradation: str
    frame_stride: int


def _load_per_sequence_records(tables_root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for json_path in tables_root.rglob("*_results.json"):
        try:
            with json_path.open() as f:
                records.append(json.load(f))
        except Exception as exc:
            logger.warning(f"Skipping unreadable {json_path}: {exc}")
    if not records:
        raise RuntimeError(f"No per-sequence result JSONs found under {tables_root}")
    df = pd.DataFrame(records)
    df["frame_stride"] = df["frame_stride"].astype(int)
    for col in ("ate", "rpe", "ate_sim3", "rpe_rot"):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _gt_length_strided(pose_path: Path, stride: int) -> float:
    poses = np.loadtxt(pose_path)
    if poses.ndim != 2 or poses.shape[0] < 2:
        return float("nan")
    xyz = poses[:, :3][:: max(1, int(stride))]
    if len(xyz) < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def _build_gt_length_table() -> pd.DataFrame:
    rows: list[dict] = []
    for seq in list_sequences():
        if not seq.pose_exist():
            logger.warning(f"Missing pose file for {seq.sequence_id}; skipping.")
            continue
        for stride in STRIDES:
            rows.append(
                {
                    "sequence": seq.sequence_id,
                    "frame_stride": int(stride),
                    "gt_length_m": _gt_length_strided(seq.pose_path, stride),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_boot: int, confidence: float, seed: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(finite, size=(n_boot, finite.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.percentile(means, 100.0 * alpha)), float(np.percentile(means, 100.0 * (1.0 - alpha)))


def _aggregate_cells(df: pd.DataFrame, n_boot: int, confidence: float, seed: int) -> pd.DataFrame:
    group_cols = ["frontend", "backend", "degradation", "frame_stride"]
    rows: list[dict] = []
    for group_index, (key, group) in enumerate(df.groupby(group_cols, sort=True)):
        cell = dict(zip(group_cols, key, strict=True))
        for metric in ("ate", "rpe", "ate_pct", "ate_sim3", "rpe_rot"):
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            n = int(finite.size)
            cell[f"{metric}_n_finite"] = n
            if n == 0:
                cell.update(
                    {
                        f"{metric}_mean": float("nan"),
                        f"{metric}_std": float("nan"),
                        f"{metric}_median": float("nan"),
                        f"{metric}_iqr": float("nan"),
                        f"{metric}_ci_lower": float("nan"),
                        f"{metric}_ci_upper": float("nan"),
                    }
                )
                continue
            cell[f"{metric}_mean"] = float(np.mean(finite))
            cell[f"{metric}_std"] = float(np.std(finite, ddof=1)) if n > 1 else 0.0
            cell[f"{metric}_median"] = float(np.median(finite))
            cell[f"{metric}_iqr"] = float(np.percentile(finite, 75) - np.percentile(finite, 25))
            ci_l, ci_u = _bootstrap_ci(
                values=finite,
                n_boot=n_boot,
                confidence=confidence,
                seed=seed + group_index * 31 + hash(metric) % 1009,
            )
            cell[f"{metric}_ci_lower"] = ci_l
            cell[f"{metric}_ci_upper"] = ci_u
        cell["sequences_expected"] = N_EXPECTED_SEQUENCES
        cell["coverage_str"] = f"{cell['ate_n_finite']}/{N_EXPECTED_SEQUENCES}"
        rows.append(cell)
    return pd.DataFrame(rows)


def _paired_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["degradation"] == "none"].copy()
    out: list[dict] = []
    for fe in FRONTENDS:
        for stride in STRIDES:
            ess = baseline[
                (baseline["frontend"] == fe)
                & (baseline["backend"] == "essential")
                & (baseline["frame_stride"] == stride)
            ]
            pnp = baseline[
                (baseline["frontend"] == fe)
                & (baseline["backend"] == "pnp")
                & (baseline["frame_stride"] == stride)
            ]
            merged = ess.merge(pnp, on="sequence", suffixes=("_e", "_p"))
            for metric in ("ate", "rpe"):
                paired = merged[[f"{metric}_e", f"{metric}_p"]].dropna()
                paired = paired[np.isfinite(paired[f"{metric}_e"]) & np.isfinite(paired[f"{metric}_p"])]
                row = {
                    "frontend": fe,
                    "frame_stride": int(stride),
                    "metric": metric,
                    "n_pairs": int(len(paired)),
                    "median_diff_e_minus_p": float("nan"),
                    "p_raw_greater": float("nan"),
                    "p_raw_two_sided": float("nan"),
                }
                if len(paired) >= 5:
                    diffs = paired[f"{metric}_e"].to_numpy() - paired[f"{metric}_p"].to_numpy()
                    row["median_diff_e_minus_p"] = float(np.median(diffs))
                    try:
                        # one-sided: Essential is larger than PnP, i.e. PnP improves
                        row["p_raw_greater"] = float(wilcoxon(diffs, alternative="greater").pvalue)
                        row["p_raw_two_sided"] = float(wilcoxon(diffs, alternative="two-sided").pvalue)
                    except Exception as exc:
                        logger.warning(f"Wilcoxon failed for {fe} stride {stride} {metric}: {exc}")
                out.append(row)
    tests_df = pd.DataFrame(out)
    _apply_holm(tests_df, p_col="p_raw_greater", out_col="p_holm_greater")
    _apply_holm(tests_df, p_col="p_raw_two_sided", out_col="p_holm_two_sided")
    return tests_df


def _apply_holm(tests_df: pd.DataFrame, p_col: str, out_col: str) -> None:
    """Adds a Holm-Bonferroni corrected p-value column in place."""
    mask = tests_df[p_col].notna()
    p_values = tests_df.loc[mask, p_col].to_numpy(dtype=float)
    m = p_values.size
    if m == 0:
        tests_df[out_col] = float("nan")
        return
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    # Holm step-down: adjusted_(k) = max_{j<=k} min(1, (m - j) * p_(j))  (j is 0-indexed)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_p * (m - np.arange(m)))
    )
    holm = np.empty(m, dtype=float)
    holm[order] = adjusted_sorted
    tests_df[out_col] = float("nan")
    tests_df.loc[mask, out_col] = holm


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_pvalue(value: float, bold: bool = False) -> str:
    """LaTeX-ready p-value cell. Bolding wraps math properly via \\boldsymbol."""
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / 10**exponent
        body = f"{mantissa:.2f}\\times10^{{{exponent}}}"
        if bold:
            return f"$\\boldsymbol{{{body}}}$"
        return f"${body}$"
    if bold:
        return f"\\textbf{{{value:.4f}}}"
    return f"{value:.4f}"


def _baseline_view(agg: pd.DataFrame) -> pd.DataFrame:
    return (
        agg[agg["degradation"] == "none"]
        .sort_values(["frontend", "backend", "frame_stride"])
        .copy()
    )


def _best_backend(pair: pd.DataFrame, mean_col: str) -> str:
    means = pair.set_index("backend")[mean_col].to_dict()
    return min(means, key=lambda b: means[b] if np.isfinite(means[b]) else np.inf)


def _write_ate_table(agg: pd.DataFrame, output_path: Path) -> None:
    """Detailed ATE table: absolute + normalized (% of GT length) + coverage + CI."""
    baseline = _baseline_view(agg)
    if baseline.empty:
        logger.warning("No baseline rows; skipping ATE detailed table.")
        return

    lines: list[str] = [
        "\\begin{table}[H]",
        (
            "\\caption{Baseline ATE on the 32-sequence TartanAir validation split: "
            "absolute mean$\\pm$std (m), normalized mean (\\% of GT trajectory length), "
            "finite-aggregation coverage (n/32), median with IQR, and 95\\% bootstrap "
            "confidence interval over finite per-sequence values. \\textbf{Bold}: "
            "within each (frontend, stride) pair, the backend with the lower absolute ATE mean.}"
        ),
        "\\label{tab:metrics_detailed_ate}",
        "\\footnotesize",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrlllll}",
        "\\toprule",
        (
            "frontend & backend & stride & mean $\\pm$ std (m) & "
            "ATE\\(_{\\%}\\) mean (\\%) & coverage & median [IQR] (m) & "
            "95\\% CI (m) \\\\"
        ),
        "\\midrule",
    ]

    for (_fe, _stride), pair in baseline.groupby(["frontend", "frame_stride"], sort=True):
        best = _best_backend(pair, "ate_mean")
        for _, row in pair.iterrows():
            is_best = row["backend"] == best
            b_open = "\\textbf{" if is_best else ""
            b_close = "}" if is_best else ""
            mean_std = f"{_fmt(row['ate_mean'])} $\\pm$ {_fmt(row['ate_std'])}"
            pct_mean = (
                f"{_fmt(row['ate_pct_mean'], digits=2)} $\\pm$ "
                f"{_fmt(row['ate_pct_std'], digits=2)}"
            )
            median_iqr = f"{_fmt(row['ate_median'])} [{_fmt(row['ate_iqr'])}]"
            ci = f"[{_fmt(row['ate_ci_lower'])}, {_fmt(row['ate_ci_upper'])}]"
            cells = [
                str(row["frontend"]),
                str(row["backend"]),
                str(int(row["frame_stride"])),
                f"{b_open}{mean_std}{b_close}",
                f"{b_open}{pct_mean}{b_close}",
                row["coverage_str"],
                f"{b_open}{median_iqr}{b_close}",
                f"{b_open}{ci}{b_close}",
            ]
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))
    logger.info(f"Wrote {output_path}")


def _write_rpe_table(agg: pd.DataFrame, output_path: Path) -> None:
    """Detailed RPE table: mean+/-std + coverage + median/IQR + 95% CI."""
    baseline = _baseline_view(agg)
    if baseline.empty:
        logger.warning("No baseline rows; skipping RPE detailed table.")
        return

    lines: list[str] = [
        "\\begin{table}[H]",
        (
            "\\caption{Baseline RPE (m/frame) on the 32-sequence TartanAir validation split: "
            "mean$\\pm$std, finite-aggregation coverage (n/32), median with IQR, and "
            "95\\% bootstrap confidence interval. \\textbf{Bold}: within each "
            "(frontend, stride) pair, the backend with the lower RPE mean.}"
        ),
        "\\label{tab:metrics_detailed_rpe}",
        "\\footnotesize",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrllll}",
        "\\toprule",
        (
            "frontend & backend & stride & mean $\\pm$ std (m/frame) & "
            "coverage & median [IQR] (m/frame) & 95\\% CI (m/frame) \\\\"
        ),
        "\\midrule",
    ]

    for (_fe, _stride), pair in baseline.groupby(["frontend", "frame_stride"], sort=True):
        best = _best_backend(pair, "rpe_mean")
        for _, row in pair.iterrows():
            is_best = row["backend"] == best
            b_open = "\\textbf{" if is_best else ""
            b_close = "}" if is_best else ""
            mean_std = f"{_fmt(row['rpe_mean'])} $\\pm$ {_fmt(row['rpe_std'])}"
            median_iqr = f"{_fmt(row['rpe_median'])} [{_fmt(row['rpe_iqr'])}]"
            ci = f"[{_fmt(row['rpe_ci_lower'])}, {_fmt(row['rpe_ci_upper'])}]"
            cells = [
                str(row["frontend"]),
                str(row["backend"]),
                str(int(row["frame_stride"])),
                f"{b_open}{mean_std}{b_close}",
                row["coverage_str"],
                f"{b_open}{median_iqr}{b_close}",
                f"{b_open}{ci}{b_close}",
            ]
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))
    logger.info(f"Wrote {output_path}")


def _write_ate_sim3_table(agg: pd.DataFrame, output_path: Path) -> None:
    """Detailed PnP-only ATE under Sim(3) alignment, side-by-side with SE(3) ATE."""
    baseline = _baseline_view(agg)
    pnp = baseline[baseline["backend"] == "pnp"].copy()
    if pnp.empty or pnp["ate_sim3_mean"].isna().all():
        logger.warning("No PnP Sim(3) rows; skipping ATE Sim3 detailed table.")
        return

    lines: list[str] = [
        "\\begin{table}[H]",
        (
            "\\caption{Baseline PnP ATE under Sim(3) alignment on the 32-sequence "
            "TartanAir validation split: mean$\\pm$std (m), finite-aggregation coverage "
            "(n/32), median with IQR, and 95\\% bootstrap confidence interval. "
            "Reported side-by-side with the SE(3) PnP ATE from Table~\\ref{tab:metrics_detailed_ate} "
            "to isolate trajectory-shape error from metric-scale drift. \\textbf{Bold}: within "
            "each stride, the frontend with the lower Sim(3) ATE mean (i.e. lower "
            "trajectory-shape error after metric-scale is absorbed).}"
        ),
        "\\label{tab:metrics_detailed_ate_sim3}",
        "\\footnotesize",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrlllll}",
        "\\toprule",
        (
            "frontend & backend & stride & ATE SE(3) (m) & ATE Sim(3) (m) & "
            "coverage & Sim(3) median [IQR] & Sim(3) 95\\% CI \\\\"
        ),
        "\\midrule",
    ]

    pnp_sorted = pnp.sort_values(["frame_stride", "frontend"]).copy()
    best_sim3_per_stride: dict[int, str] = {}
    for stride_val, stride_group in pnp_sorted.groupby("frame_stride", sort=True):
        finite = stride_group.dropna(subset=["ate_sim3_mean"])
        if finite.empty:
            continue
        best_row = finite.loc[finite["ate_sim3_mean"].idxmin()]
        best_sim3_per_stride[int(stride_val)] = str(best_row["frontend"])

    for _, row in pnp_sorted.iterrows():
        is_best = best_sim3_per_stride.get(int(row["frame_stride"])) == str(row["frontend"])
        b_open = "\\textbf{" if is_best else ""
        b_close = "}" if is_best else ""
        se3 = f"{_fmt(row['ate_mean'])} $\\pm$ {_fmt(row['ate_std'])}"
        sim3 = f"{_fmt(row['ate_sim3_mean'])} $\\pm$ {_fmt(row['ate_sim3_std'])}"
        median_iqr = f"{_fmt(row['ate_sim3_median'])} [{_fmt(row['ate_sim3_iqr'])}]"
        ci = f"[{_fmt(row['ate_sim3_ci_lower'])}, {_fmt(row['ate_sim3_ci_upper'])}]"
        cells = [
            str(row["frontend"]),
            str(row["backend"]),
            str(int(row["frame_stride"])),
            se3,
            f"{b_open}{sim3}{b_close}",
            row["coverage_str"],
            f"{b_open}{median_iqr}{b_close}",
            f"{b_open}{ci}{b_close}",
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))
    logger.info(f"Wrote {output_path}")


def _write_rpe_rot_table(agg: pd.DataFrame, output_path: Path) -> None:
    """Detailed Rotational RPE table, mirroring the translational-RPE table layout."""
    baseline = _baseline_view(agg)
    if baseline.empty or baseline["rpe_rot_mean"].isna().all():
        logger.warning("No rotational RPE rows; skipping detailed table.")
        return

    lines: list[str] = [
        "\\begin{table}[H]",
        (
            "\\caption{Baseline Rotational RPE (deg/frame) on the 32-sequence TartanAir "
            "validation split: mean$\\pm$std, finite-aggregation coverage (n/32), median "
            "with IQR, and 95\\% bootstrap confidence interval. \\textbf{Bold}: within "
            "each (frontend, stride) pair, the backend with the lower rotational RPE mean.}"
        ),
        "\\label{tab:metrics_detailed_rpe_rot}",
        "\\footnotesize",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrllll}",
        "\\toprule",
        (
            "frontend & backend & stride & mean $\\pm$ std (deg/frame) & "
            "coverage & median [IQR] (deg/frame) & 95\\% CI (deg/frame) \\\\"
        ),
        "\\midrule",
    ]

    for (_fe, _stride), pair in baseline.groupby(["frontend", "frame_stride"], sort=True):
        best = _best_backend(pair, "rpe_rot_mean")
        for _, row in pair.iterrows():
            is_best = row["backend"] == best
            b_open = "\\textbf{" if is_best else ""
            b_close = "}" if is_best else ""
            mean_std = f"{_fmt(row['rpe_rot_mean'])} $\\pm$ {_fmt(row['rpe_rot_std'])}"
            median_iqr = f"{_fmt(row['rpe_rot_median'])} [{_fmt(row['rpe_rot_iqr'])}]"
            ci = f"[{_fmt(row['rpe_rot_ci_lower'])}, {_fmt(row['rpe_rot_ci_upper'])}]"
            cells = [
                str(row["frontend"]),
                str(row["backend"]),
                str(int(row["frame_stride"])),
                f"{b_open}{mean_std}{b_close}",
                row["coverage_str"],
                f"{b_open}{median_iqr}{b_close}",
                f"{b_open}{ci}{b_close}",
            ]
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))
    logger.info(f"Wrote {output_path}")


def _write_wilcoxon_tex(tests_df: pd.DataFrame, output_path: Path) -> None:
    """Renders a LaTeX table of Wilcoxon outcomes with Holm-Bonferroni correction."""
    if tests_df.empty:
        logger.warning("No Wilcoxon rows; skipping table.")
        return

    sort_cols = ["metric", "frontend", "frame_stride"]
    tdf = tests_df.sort_values(sort_cols).copy()

    lines: list[str] = [
        "\\begin{table}[H]",
        (
            "\\caption{Paired Wilcoxon signed-rank tests across the 16 baseline cells "
            "(2 frontends $\\times$ 4 strides $\\times$ 2 metrics), comparing per-sequence "
            "errors of Essential vs. PnP on shared sequences with finite values for both "
            "backends. One-sided alternative: error of Essential greater than PnP "
            "(i.e. PnP improves accuracy). Raw and Holm--Bonferroni corrected p-values "
            "are reported jointly across the family of 16 tests. \\textbf{Bold}: corrected "
            "$p<0.05$ under the one-sided test.}"
        ),
        "\\label{tab:wilcoxon_holm}",
        "\\footnotesize",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        (
            "metric & frontend & stride & $n$ & median($E{-}P$) & "
            "$p_{\\text{raw}}$ (1-s.) & $p_{\\text{Holm}}$ (1-s.) & "
            "$p_{\\text{Holm}}$ (2-s.) \\\\"
        ),
        "\\midrule",
    ]

    for _, row in tdf.iterrows():
        significant = (
            isinstance(row["p_holm_greater"], float)
            and np.isfinite(row["p_holm_greater"])
            and row["p_holm_greater"] < 0.05
        )
        p_raw = _fmt_pvalue(row["p_raw_greater"])
        p_holm_g = _fmt_pvalue(row["p_holm_greater"], bold=significant)
        p_holm_2 = _fmt_pvalue(row["p_holm_two_sided"])
        cells = [
            row["metric"].upper(),
            str(row["frontend"]),
            str(int(row["frame_stride"])),
            str(int(row["n_pairs"])),
            _fmt(row["median_diff_e_minus_p"]),
            p_raw,
            p_holm_g,
            p_holm_2,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))
    logger.info(f"Wrote {output_path}")


def main(n_boot: int = 1000, confidence: float = 0.95, seed: int = 0) -> None:
    tables_root = SETTINGS.paths.tables_root
    df = _load_per_sequence_records(tables_root)
    logger.info(f"Loaded {len(df)} per-sequence records")

    gt = _build_gt_length_table()
    df = df.merge(gt, on=["sequence", "frame_stride"], how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        df["ate_pct"] = 100.0 * df["ate"] / df["gt_length_m"]
    df.loc[~np.isfinite(df["ate_pct"]), "ate_pct"] = np.nan

    agg = _aggregate_cells(df, n_boot=n_boot, confidence=confidence, seed=seed)
    tests_df = _paired_wilcoxon_tests(df)

    out_dir = tables_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wave1_per_sequence_with_pct.csv").write_text(df.to_csv(index=False))
    agg.to_csv(out_dir / "wave1_aggregated_with_pct.csv", index=False)
    tests_df.to_csv(out_dir / "wave1_wilcoxon_holm.csv", index=False)

    paper_tables = SETTINGS.paths.paper_tables_root
    paper_tables.mkdir(parents=True, exist_ok=True)
    _write_ate_table(agg, paper_tables / "vo_metrics_summary_detailed_ate.tex")
    _write_rpe_table(agg, paper_tables / "vo_metrics_summary_detailed_rpe.tex")
    _write_ate_sim3_table(agg, paper_tables / "vo_metrics_summary_detailed_ate_sim3.tex")
    _write_rpe_rot_table(agg, paper_tables / "vo_metrics_summary_detailed_rpe_rot.tex")
    _write_wilcoxon_tex(tests_df, paper_tables / "wave1_wilcoxon_holm.tex")

    legacy = paper_tables / "vo_metrics_summary_detailed.tex"
    if legacy.exists():
        legacy.unlink()
        logger.info(f"Removed legacy combined table {legacy}")
    legacy_md = paper_tables / "wave1_wilcoxon_holm.md"
    if legacy_md.exists():
        legacy_md.unlink()
        logger.info(f"Removed legacy markdown {legacy_md}")

    logger.info("Wave 1 metrics complete.")


if __name__ == "__main__":
    fire.Fire(main)
