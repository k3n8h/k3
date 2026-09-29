"""Data analysis over CSV/XLSX files in the workspace."""
import pandas as pd

from bot.registry import tool
from bot.tools.code import safe_path


def load(path: str) -> pd.DataFrame:
    p = safe_path(path)
    return pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)


@tool("Summarize a CSV/XLSX file in the workspace: shape, columns, dtypes, describe() stats and head.")
def describe_data(path: str) -> dict:
    df = load(path)
    return {"shape": list(df.shape), "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "stats": df.describe(include="all").fillna("").to_dict(), "head": df.head(5).to_dict("records")}


@tool("Run a pandas query on a data file: filter with a pandas .query() expression, then optionally "
      "group_by a column and aggregate `column` with agg (sum, mean, count, min, max).")
def query_data(path: str, filter: str = "", group_by: str = "", column: str = "", agg: str = "mean") -> list:
    df = load(path)
    if filter:
        df = df.query(filter)
    if group_by:
        df = getattr(df.groupby(group_by)[column], agg)().reset_index()
    return df.head(100).to_dict("records")


@tool("Make a bar/line/hist chart PNG in the workspace from a data file. kind: bar, line or hist.")
def make_chart(path: str, x: str, y: str = "", kind: str = "bar", out: str = "chart.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == "hist":
        ax.hist(df[x].dropna())
    else:
        getattr(ax, "bar" if kind == "bar" else "plot")(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    dest = safe_path(out)
    fig.savefig(dest)
    plt.close(fig)
    return f"saved {out}"
