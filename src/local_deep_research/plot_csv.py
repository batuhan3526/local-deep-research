"""Chart a CSV column without running model-written code (final mode: analysts may not execute arbitrary Python).

usage: plot_csv.py <research_dir> <data/file.csv> --x COL --y COL[,COL2] [--kind bar|barh|line|scatter]
                   [--title "..."] --out fig/<name>.png
Paths are relative to the research dir and must stay inside it (data/ in, fig/ out).
"""
import argparse
import pathlib
import sys

try:
    import matplotlib

    matplotlib.use("Agg")  # no display needed (servers, Windows services, CI)
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as _e:  # optional "reports" extra: say what is missing instead of a traceback
    plt = pd = None
    _MISSING = _e.name


def inside(root, rel, sub, suffix):
    p = (root / rel).resolve()
    if p.parent != (root / sub).resolve() or p.suffix != suffix:
        sys.exit(f"{rel}: must be {sub}/<name>{suffix} inside {root}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path); ap.add_argument("csv")
    ap.add_argument("--x", required=True); ap.add_argument("--y", required=True)
    ap.add_argument("--kind", default="bar", choices=["bar", "barh", "line", "scatter"])
    ap.add_argument("--title", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if plt is None:
        sys.exit(f"plot_csv needs matplotlib and pandas ({_MISSING} is missing): "
                 "pip install 'local-deep-research[reports]'")
    root = a.root.resolve()
    if root.name != "research" and (root / "research").is_dir():
        root = root / "research"
    src, out = inside(root, a.csv, "data", ".csv"), inside(root, a.out, "fig", ".png")
    df = pd.read_csv(src, nrows=5000)
    ys = a.y.split(",")
    for c in [a.x, *ys]:
        if c not in df.columns:
            sys.exit(f"column {c!r} not in {list(df.columns)}")
    fig, ax = plt.subplots(figsize=(9, 5))
    if a.kind == "scatter":
        for y in ys:
            ax.scatter(df[a.x], pd.to_numeric(df[y], errors="coerce"), label=y)
    else:
        df.assign(**{y: pd.to_numeric(df[y], errors="coerce") for y in ys}).plot(x=a.x, y=ys, kind=a.kind, ax=ax)
    ax.set_title(a.title[:120]); ax.legend() if len(ys) > 1 else None
    fig.tight_layout(); out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
