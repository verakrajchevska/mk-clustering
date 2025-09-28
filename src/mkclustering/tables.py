from __future__ import annotations
from pathlib import Path
import re
import yaml
import pandas as pd
from mkclustering.paths import Paths

COLS = {
    "suffix":        ["suffix_max_purity","max_suffix_purity","suffix_purity_max","max-suffix-purity"],
    "pos_purity":    ["pos_purity","pos-purity"],
    "pos_entropy":   ["pos_entropy","pos-entropy"],
    "pos_coverage":  ["pos_coverage","pos-coverage","coverage","POS coverage"],
    "msi":           ["msi","MSI","msi_pos_minus_suffix","MSI (pos-suf)"],
}

def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def _summarize_cluster_stats(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for f in (eval_dir / "metrics").glob("cluster_stats_*_K*.csv"):
        m = re.match(r"cluster_stats_(.+)_(serial|parallel|sklearn)_K(\d+)\.csv", f.name)
        if not m:
            continue
        slug, method, K = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(f)

        c_suffix = _pick(df, COLS["suffix"])
        c_pospur = _pick(df, COLS["pos_purity"])
        c_posent = _pick(df, COLS["pos_entropy"])
        c_poscov = _pick(df, COLS["pos_coverage"])
        c_msi    = _pick(df, COLS["msi"])

        if c_msi is None and c_pospur and c_suffix:
            msi_mean = (df[c_pospur] - df[c_suffix]).mean()
        elif c_msi is not None:
            msi_mean = df[c_msi].mean()
        else:
            msi_mean = float("nan")

        rows.append({
            "vecslug": slug,
            "method": method,
            "K": int(K),
            "suffix_mean": df[c_suffix].mean() if c_suffix else float("nan"),
            "pos_purity_mean": df[c_pospur].mean() if c_pospur else float("nan"),
            "pos_entropy_mean": df[c_posent].mean() if c_posent else float("nan"),
            "pos_coverage_mean": df[c_poscov].mean() if c_poscov else float("nan"),
            "msi_mean": msi_mean,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["vecslug","method","K"])

def _summarize_alignment(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for f in (eval_dir / "metrics").glob("alignment_*_K*.csv"):
        mK = re.search(r"_K(\d+)\.csv$", f.name)
        K = int(mK.group(1)) if mK else None
        df = pd.read_csv(f)

        if "cosine" not in df.columns and "centroid_cosine" in df.columns:
            df["cosine"] = df["centroid_cosine"]
        if "jaccard" not in df.columns and "mean_jaccard" in df.columns:
            df["jaccard"] = df["mean_jaccard"]

        rows.append({
            "file": f.name,
            "K": K,
            "mean_centroid_cosine": df["cosine"].mean() if "cosine" in df.columns else float("nan"),
            "median_centroid_cosine": df["cosine"].median() if "cosine" in df.columns else float("nan"),
            "mean_jaccard": df["jaccard"].mean() if "jaccard" in df.columns else float("nan"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["K","file"]) if not out.empty else out

def _to_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data found._"
    fmt = df.copy()
    for c in cols:
        if c in fmt.columns and fmt[c].dtype.kind in "fc":
            fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
    return fmt[cols].to_markdown(index=False)

def _to_latex(df: pd.DataFrame, cols: list[str], pretty_names: dict[str,str]) -> str:
    """Return a LaTeX tabular string with pretty column headers."""
    sub = df[cols].copy()
    sub = sub.rename(columns=pretty_names)

    for c in sub.columns:
        if c.lower() in {"k"}:
            sub[c] = pd.to_numeric(sub[c], errors="ignore", downcast="integer")

    return sub.to_latex(index=False, escape=True, float_format="%.3f")

def run(paths: Paths, cfg_path: str = "configs/default.yaml"):
    outdir = Path("reports/tables")
    outdir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["paths"]["eval_dir"])

    stats = _summarize_cluster_stats(eval_dir)
    stats_csv = outdir / "cluster_stats_summary.csv"
    stats_md  = outdir / "cluster_stats_summary.md"
    stats_tex = outdir / "cluster_stats_summary.tex"

    if not stats.empty:
        stats.to_csv(stats_csv, index=False)

        md_cols   = ["vecslug","method","K","suffix_mean","pos_purity_mean","msi_mean","pos_entropy_mean","pos_coverage_mean"]
        md_string = _to_markdown(stats, md_cols)
        stats_md.write_text(md_string, encoding="utf-8")

        pretty = {
            "vecslug": "Model",
            "method": "Method",
            "K": "K",
            "suffix_mean": "Suffix purity (\\(\\uparrow\\))",
            "pos_purity_mean": "POS purity (\\(\\uparrow\\))",
            "msi_mean": "MSI (POS$-$suffix, \\(\\uparrow\\) semantic)",
            "pos_entropy_mean": "POS entropy (\\(\\uparrow\\) mixed)",
            "pos_coverage_mean": "POS coverage",
        }
        tex_cols = ["vecslug","method","K","suffix_mean","pos_purity_mean","msi_mean","pos_entropy_mean","pos_coverage_mean"]
        tex_string = _to_latex(stats, tex_cols, pretty)
        stats_tex.write_text(tex_string, encoding="utf-8")

        print(f"[tables] wrote {stats_csv}")
        print(f"[tables] wrote {stats_md}")
        print(f"[tables] wrote {stats_tex}")
    else:
        print("[tables] no cluster_stats_* files found.")

    al = _summarize_alignment(eval_dir)
    if not al.empty:
        al_csv = outdir / "alignment_summary.csv"
        al_md  = outdir / "alignment_summary.md"
        al_tex = outdir / "alignment_summary.tex"

        al.to_csv(al_csv, index=False)

        md_cols = ["K","mean_centroid_cosine","median_centroid_cosine","mean_jaccard","file"]
        al_md.write_text(_to_markdown(al, md_cols), encoding="utf-8")

        pretty_al = {
            "K": "K",
            "mean_centroid_cosine": "Mean centroid cosine",
            "median_centroid_cosine": "Median centroid cosine",
            "mean_jaccard": "Mean Jaccard",
            "file": "File",
        }
        al_tex.write_text(_to_latex(al, md_cols, pretty_al), encoding="utf-8")

        print(f"[tables] wrote {al_csv}")
        print(f"[tables] wrote {al_md}")
        print(f"[tables] wrote {al_tex}")
