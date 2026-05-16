import pandas as pd


def _latex_escape_text(value):
    if isinstance(value, str):
        if value == "frame_stride":
            return "stride"
        return value.replace("_", "\\_")
    return value


def format_mean_std_bold_best(df: pd.DataFrame, index_col: str, columns_cols: list[str], mean_col: str, std_col: str, ascending: bool = True) -> pd.DataFrame:
    """
    Pivots the dataframe, formats cells as 'mean \pm std', and bolds the best mean.

    When ``columns_cols`` is a 2-level header (e.g. ``[frontend, backend]``),
    the best cell is selected within each first-level group (per-frontend),
    so the reader can see which backend wins inside each frontend column-pair.
    For single-level column headers, the best cell is the row-global min/max.
    """
    pivot_mean = df.pivot_table(index=index_col, columns=columns_cols, values=mean_col)
    pivot_std = df.pivot_table(index=index_col, columns=columns_cols, values=std_col)

    formatted_df = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)

    multi_level = isinstance(pivot_mean.columns, pd.MultiIndex) and pivot_mean.columns.nlevels >= 2

    def _best(values: pd.Series) -> float:
        if values.dropna().empty:
            return float("nan")
        return values.min() if ascending else values.max()

    for idx in pivot_mean.index:
        row_means = pivot_mean.loc[idx]
        row_stds = pivot_std.loc[idx]

        if multi_level:
            best_by_group = {
                group_key: _best(row_means.xs(group_key, level=0))
                for group_key in row_means.index.get_level_values(0).unique()
            }

        for col in pivot_mean.columns:
            mean_val = row_means[col]
            std_val = row_stds[col]

            if pd.isna(mean_val):
                formatted_df.loc[idx, col] = "-"
                continue

            cell_str = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
            if multi_level:
                target = best_by_group.get(col[0], float("nan"))
            else:
                target = _best(row_means)
            if pd.notna(target) and mean_val == target:
                cell_str = f"\\textbf{{{cell_str}}}"
            formatted_df.loc[idx, col] = cell_str

    formatted_df.index = formatted_df.index.map(_latex_escape_text)
    formatted_df.index.name = _latex_escape_text(index_col)
    if isinstance(formatted_df.columns, pd.MultiIndex):
        formatted_df.columns = pd.MultiIndex.from_tuples(
            tuple(_latex_escape_text(value) for value in column)
            for column in formatted_df.columns
        )
        formatted_df.columns.names = [_latex_escape_text(name) for name in pivot_mean.columns.names]
    else:
        formatted_df.columns = formatted_df.columns.map(_latex_escape_text)
        formatted_df.columns.name = _latex_escape_text(pivot_mean.columns.name)

    return formatted_df
