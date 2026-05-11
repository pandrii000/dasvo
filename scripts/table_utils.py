import pandas as pd


def _latex_escape_text(value):
    if isinstance(value, str):
        if value == "frame_stride":
            return "stride"
        return value.replace("_", "\\_")
    return value


def format_mean_std_bold_best(df: pd.DataFrame, index_col: str, columns_cols: list[str], mean_col: str, std_col: str, ascending: bool = True) -> pd.DataFrame:
    """
    Pivots the dataframe, formats cells as 'mean \pm std', and bolds the best mean in each row.
    """
    pivot_mean = df.pivot_table(index=index_col, columns=columns_cols, values=mean_col)
    pivot_std = df.pivot_table(index=index_col, columns=columns_cols, values=std_col)
    
    formatted_df = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
    
    for idx in pivot_mean.index:
        row_means = pivot_mean.loc[idx]
        row_stds = pivot_std.loc[idx]
        
        if ascending:
            best_val = row_means.min()
        else:
            best_val = row_means.max()
            
        for col in pivot_mean.columns:
            mean_val = row_means[col]
            std_val = row_stds[col]
            
            if pd.isna(mean_val):
                formatted_df.loc[idx, col] = "-"
            else:
                cell_str = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
                if mean_val == best_val:
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
