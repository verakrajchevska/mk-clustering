from pathlib import Path
import re
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


FIXED_YLIMS = {
    "purity":       (0.25, 0.90),
    "NMI":          (0.25, 0.90),
    "V":            (0.25, 0.90),
    "homogeneity":  (0.25, 0.90),
    "completeness": (0.25, 0.90),
    "silhouette":   (-0.04, 0.02),
}

WRITE_ZOOMED = True
ZOOM_PAD_FRAC = 0.06  

FAMILY_COLORS = {
    "baseline": "#C0392B",      
    "raw":      "#2E86C1",      
    "lemma":    "#27AE60",     
}
PARALLEL_ALPHA = 0.95
SERIAL_ALPHA   = 0.78
PARALLEL_LW    = 2.4
SERIAL_LW      = 2.0
MARKER_SIZE    = 5.5

ORDER = [
    ("baseline", "serial"),
    ("baseline", "parallel"),
    ("raw",      "serial"),
    ("raw",      "parallel"),
    ("lemma",    "serial"),
    ("lemma",    "parallel"),
]
ORDER_INDEX = {k: i for i, k in enumerate(ORDER)}


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\\-]+", "-", s)

def _family(slug: str) -> str:
    s = slug.lower()
    if s.startswith("baseline-"):
        return "baseline"
    if "lemma" in s:
        return "lemma"
    return "raw"

def _lighten(hex_color: str, amount: float = 0.45) -> str:
    """
    Blend color towards white by `amount` (0..1).
    """
    r, g, b = to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def _pair_style(slug: str, method: str):
    fam = _family(slug)
    base = FAMILY_COLORS.get(fam, "#555555")
    if method == "parallel":
        return dict(color=base, alpha=PARALLEL_ALPHA, linewidth=PARALLEL_LW)
    else:  
        return dict(color=_lighten(base, 0.42), alpha=SERIAL_ALPHA, linewidth=SERIAL_LW)

def _legend_sort_key(row):
    fam = _family(row.vecslug)
    return (ORDER_INDEX.get((fam, row.method), 999), row.vecslug)

def _titlecase_metric(m: str) -> str:
    return {
        "NMI": "NMI",
        "V": "V-measure",
        "silhouette": "Silhouette",
        "purity": "Purity",
        "homogeneity": "Homogeneity",
        "completeness": "Completeness",
    }.get(m, m.title())

def _plot_metric(df, metric, outdir: Path, fixed_ylim=True):
    """Draw one figure with all (vecslug,method) lines for the given metric."""
    if metric not in df.columns:
        return

    if fixed_ylim:
        figdir = outdir
        ylim = FIXED_YLIMS.get(metric)
        suffix = ""
    else:
        figdir = _ensure_dir(outdir / "zoomed")
        vals = df[metric].dropna()
        if not len(vals):
            return
        vmin, vmax = float(vals.min()), float(vals.max())
        pad = max(1e-6, (vmax - vmin) * ZOOM_PAD_FRAC)
        ylim = (vmin - pad, vmax + pad)
        suffix = "_zoom"

    plt.figure()
    groups = (
        df[["vecslug", "method"]]
        .drop_duplicates()
        .sort_values(["vecslug", "method"])
        .itertuples(index=False)
    )
    groups = sorted(groups, key=_legend_sort_key)

    for g in groups:
        sub = df[(df["vecslug"] == g.vecslug) & (df["method"] == g.method)].sort_values("K")
        if sub[metric].notna().sum() == 0:
            continue
        style = _pair_style(g.vecslug, g.method)
        label = f"{g.vecslug} · {g.method}"
        plt.plot(
            sub["K"], sub[metric],
            marker="o", markersize=MARKER_SIZE, label=label, **style
        )

    plt.xlabel("K")
    plt.ylabel(_titlecase_metric(metric))
    plt.title(f"{_titlecase_metric(metric)} vs K")
    if ylim:
        plt.ylim(*ylim)
    if metric.lower() == "silhouette":
        plt.axhline(0, linestyle="--", linewidth=1, color="#666666", alpha=0.8)
    plt.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(figdir / f"{metric.lower()}_vs_K{suffix}.png", dpi=200)
    plt.close()

def run(paths, cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_dir = Path(cfg["paths"]["eval_dir"]) / "metrics"
    figdir   = _ensure_dir(Path("reports/figures"))

    qfile = eval_dir / "ud_upos_metrics.csv"
    if not qfile.exists():
        print(f"[plots] Missing {qfile} — run mkcli eval-intrinsic first.")
        return

    q = pd.read_csv(qfile)
    if "vecslug" not in q.columns:
        q["vecslug"] = "unspecified"
    if "method" not in q.columns:
        q["method"] = "serial"

    metrics = ["purity", "NMI", "V", "homogeneity", "completeness", "silhouette"]
    for m in metrics:
        _plot_metric(q, m, figdir, fixed_ylim=True)
    if WRITE_ZOOMED:
        for m in metrics:
            _plot_metric(q, m, figdir, fixed_ylim=False)


    b_long = eval_dir / "benchmarks.csv"
    if not b_long.exists():
        print(f"[plots] Skipping runtime: {b_long} not found. Run mkcli bench-speed.")
        print(f"Wrote figs to {figdir}")
        return

    dfb = pd.read_csv(b_long)

    plt.figure()
    for slug, g in dfb.groupby("vecslug"):
        wide = g.pivot_table(index="K", columns="method", values="secs", aggfunc="min").reset_index()
        for method in ["serial", "parallel"]:
            if method in wide.columns:
                style = _pair_style(slug, method)
                plt.plot(
                    wide["K"], wide[method],
                    marker="o", markersize=MARKER_SIZE,
                    label=f"{slug} · {method}", **style
                )
    plt.xlabel("K"); plt.ylabel("Seconds"); plt.title("Runtime vs K (all models)")
    plt.tight_layout(); plt.legend(fontsize=9, frameon=False)
    plt.savefig(figdir / "runtime_vs_K.png", dpi=200); plt.close()

    plt.figure()
    for slug, g in dfb.groupby("vecslug"):
        wide = g.pivot_table(index="K", columns="method", values="secs", aggfunc="min").reset_index()
        if {"serial", "parallel"} <= set(wide.columns):
            sp = wide.copy()
            sp["speedup"] = sp["serial"] / sp["parallel"]
            style = _pair_style(slug, "parallel")  
            plt.plot(
                sp["K"], sp["speedup"],
                marker="o", markersize=MARKER_SIZE,
                label=slug, **style
            )
    plt.xlabel("K"); plt.ylabel("Speedup (serial/parallel)")
    plt.title("Parallel speedup vs K (all models)")
    plt.tight_layout(); plt.legend(fontsize=9, frameon=False)
    plt.savefig(figdir / "speedup_vs_K.png", dpi=200); plt.close()

    print(f"Wrote figures to {figdir}")


