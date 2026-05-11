import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from dasvo.settings import SETTINGS

plt.style.use("seaborn-v0_8-whitegrid")


def generate_eda_assets() -> None:
    data_root = SETTINGS.paths.data_root
    figures_dir = SETTINGS.paths.figures_root
    tables_dir = SETTINGS.paths.tables_root
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        logger.error(f"Dataset root not found: {data_root}")
        return

    records = []
    for pose_path in sorted(data_root.rglob("pose_left.txt")):
        try:
            parts = pose_path.relative_to(data_root).parts
            if len(parts) < 3:
                continue
            env, difficulty, sequence = parts[:3]
        except ValueError:
            continue
            
        image_dir = pose_path.parent / "image_left"
        depth_dir = pose_path.parent / "depth_left"
        
        if not image_dir.exists() or not depth_dir.exists():
            continue
            
        image_files = sorted(image_dir.glob("*.png"))
        depth_files = sorted(depth_dir.glob("*.npy"))
        
        try:
            pose_rows = sum(1 for _ in pose_path.open("r", encoding="utf-8"))
        except Exception:
            pose_rows = 0
            
        records.append({
            "environment": env,
            "difficulty": difficulty,
            "sequence": sequence,
            "sequence_id": f"{env}/{difficulty}/{sequence}",
            "image_count": len(image_files),
            "depth_count": len(depth_files),
            "pose_rows": pose_rows,
            "image_dir": image_dir,
            "depth_dir": depth_dir,
            "pose_path": pose_path,
        })

    if not records:
        logger.warning("No sequences found for EDA.")
        return

    df = pd.DataFrame(records).sort_values(["environment", "difficulty", "sequence"]).reset_index(drop=True)
    df["is_aligned"] = (df["image_count"] == df["depth_count"]) & (df["depth_count"] == df["pose_rows"])

    # 1. Summary Stats Table
    summary = pd.Series({
        "sequences": int(len(df)),
        "environments": int(df["environment"].nunique()),
        "easy_sequences": int((df["difficulty"] == "Easy").sum()),
        "hard_sequences": int((df["difficulty"] == "Hard").sum()),
        "total_frames": int(df["pose_rows"].sum()),
        "min_sequence_length": int(df["pose_rows"].min()),
        "max_sequence_length": int(df["pose_rows"].max()),
        "mean_sequence_length": float(df["pose_rows"].mean()),
        "all_sequences_aligned": bool(df["is_aligned"].all()),
    })
    summary_df = summary.to_frame("value")
    summary_df.to_latex(
        tables_dir / "eda_stats.tex",
        float_format="%.2f",
        caption="TartanAir Dataset Subset Summary",
        label="tab:eda_stats",
    )
    
    csv_path = tables_dir / "eda_stats.csv"
    summary_df.to_csv(csv_path)

    # 2. Sequences by Difficulty and Frames by Environment
    environment_summary = (
        df.groupby("environment", as_index=False)
        .agg(
            sequences=("sequence_id", "count"),
            total_frames=("pose_rows", "sum"),
            min_frames=("pose_rows", "min"),
            max_frames=("pose_rows", "max"),
            mean_frames=("pose_rows", "mean"),
        )
        .sort_values("total_frames", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    difficulty_counts = df["difficulty"].value_counts().reindex(["Easy", "Hard"]).fillna(0)
    axes[0].bar(difficulty_counts.index, difficulty_counts.values, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Sequences by Difficulty")
    axes[0].set_ylabel("Sequences")

    env_plot = environment_summary.sort_values("total_frames")
    axes[1].barh(env_plot["environment"], env_plot["total_frames"], color="#54A24B")
    axes[1].set_title("Frames by Environment")
    axes[1].set_xlabel("Frames")

    plt.tight_layout()
    plt.savefig(figures_dir / "eda_difficulty_env.png", dpi=300)
    plt.close()

    # 3. Sequence Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(df["pose_rows"], bins=15, color="#72B7B2", edgecolor="black")
    axes[0].set_title("Sequence Length Distribution")
    axes[0].set_xlabel("Frames per Sequence")
    axes[0].set_ylabel("Count")

    plot_df = df.sort_values("pose_rows").reset_index(drop=True)
    axes[1].bar(range(len(plot_df)), plot_df["pose_rows"], color="#E45756")
    axes[1].set_title("Per-Sequence Frame Counts")
    axes[1].set_xlabel("Sequence index (sorted)")
    axes[1].set_ylabel("Frames")

    plt.tight_layout()
    plt.savefig(figures_dir / "eda_sequence_lengths.png", dpi=300)
    plt.close()

    # 4. Depth Distribution
    sampled_depth_paths = []
    for _, row in df.iterrows():
        sequence_depths = sorted(row["depth_dir"].glob("*.npy"))
        if not sequence_depths:
            continue
        step = max(1, len(sequence_depths) // 8)
        sampled_depth_paths.extend(sequence_depths[::step][:8])

    depth_means = []
    for depth_path in sampled_depth_paths:
        try:
            depth_means.append(float(np.load(depth_path).mean()))
        except Exception:
            pass

    if depth_means:
        plt.figure(figsize=(8, 4))
        plt.hist(depth_means, bins=20, color="#B279A2", edgecolor="black")
        plt.title("Distribution of Sampled Mean Depth Values")
        plt.xlabel("Mean depth per sampled frame")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "eda_depth_distribution.png", dpi=300)
        plt.close()
        
        pd.Series(depth_means).describe().to_csv(tables_dir / "eda_depth_distribution.csv")

    # Generate Comment Placeholders
    comments_dir = tables_dir / "comments"
    comments_dir.mkdir(parents=True, exist_ok=True)
    
    placeholders = {
        "eda_stats": f"Table: TartanAir Dataset Subset Summary.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "eda_difficulty_env": f"Figure: Sequences by Difficulty and Frames by Environment.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "eda_sequence_lengths": f"Figure: Sequence Length Distribution.\n\nCSV Data: {csv_path}\n\n<!-- LLM Analysis Placeholder -->\n",
        "eda_depth_distribution": f"Figure: Distribution of Sampled Mean Depth Values.\n\nCSV Data: {tables_dir / 'eda_depth_distribution.csv'}\n\n<!-- LLM Analysis Placeholder -->\n",
    }
    
    for name, content in placeholders.items():
        (comments_dir / f"{name}_comments.md").write_text(content)
        
    logger.info("Generated EDA assets and comment placeholders.")


def main() -> None:
    generate_eda_assets()
    
    if SETTINGS.publication.copy_to_paper_assets:
        import shutil
        pub_tables = SETTINGS.paths.paper_tables_root
        pub_figures = SETTINGS.paths.paper_figures_root
        pub_comments = SETTINGS.paths.paper_project_root / "comments"
        pub_tables.mkdir(parents=True, exist_ok=True)
        pub_figures.mkdir(parents=True, exist_ok=True)
        pub_comments.mkdir(parents=True, exist_ok=True)
        
        for tex_file in SETTINGS.paths.tables_root.glob("eda_*.tex"):
            shutil.copy(tex_file, pub_tables / tex_file.name)
        for png_file in SETTINGS.paths.figures_root.glob("eda_*.png"):
            shutil.copy(png_file, pub_figures / png_file.name)
        for md_file in (SETTINGS.paths.tables_root / "comments").glob("eda_*.md"):
            shutil.copy(md_file, pub_comments / md_file.name)
            
        logger.info("Copied EDA assets to publication directory.")


if __name__ == "__main__":
    fire.Fire(main)
