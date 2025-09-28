from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

YLIM_MSI = (-0.45, 0.22)  
HIGHLIGHT_K = 200        

ORDER = [
    ("baseline-cc.mk.300", "serial"),
    ("baseline-cc.mk.300", "parallel"),
    ("trained-mk_skipgram", "serial"),
    ("trained-mk_skipgram", "parallel"),
    ("trained-mk_lemma_skipgram", "serial"),
    ("trained-mk_lemma_skipgram", "parallel"),
]
COLOR = {
    "baseline-cc.mk.300": "C3",            
    "trained-mk_skipgram": "C0",         
    "trained-mk_lemma_skipgram": "C2",     
}
ALPHA   = {"serial": 0.60, "parallel": 1.0}    
LW      = {"serial": 1.8, "parallel": 2.4}
MARKER  = {"serial": "o", "parallel": "o"}
MSIZE   = 5.5

def _read_cluster_stats(eval_dir: Path) -> pd.DataFrame:
    """Collect mean MSI / POS / suffix stats for each (slug, method, K)."""
    rows = []
    for p in (eval_dir / "metrics").glob("cluster_stats_*_K*.csv"):
        m = re.match(r"cluster_stats_(.+)_(serial|parallel)_K(\d+)\.csv", p.name)
        if not m:
            continue
        slug, method, K = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(p)

        suf_col = "max_suffix_purity" if "max_suffix_purity" in df.columns else "suffix_purity"
        rows.append({
            "vecslug": slug,
            "method": method,
            "K": K,
            "msi": df["msi"].mean(),
            "pos_purity": df["pos_purity"].mean(),
            "suffix_purity": df[suf_col].mean(),
        })
    return pd.DataFrame(rows)

def plot_msi_vs_k(eval_dir: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    agg = _read_cluster_stats(eval_dir)
    if agg.empty:
        print("[plots] No cluster_stats_*.csv found in eval/metrics — run mkcli cluster-stats first.")
        return

    agg_sorted = agg.sort_values(["vecslug", "method", "K"])
    agg_sorted.to_csv(outdir / "cluster_stats_summary.csv", index=False)

    latex = (agg_sorted
             .pivot_table(index="K", columns=["vecslug","method"], values="msi")
             .round(3)
             .to_latex(na_rep="–", multicolumn=True, multicolumn_format='c'))
    (outdir / "cluster_stats_summary.tex").write_text(latex, encoding="utf-8")
    print("\n[LaTeX] MSI table written to reports/figures/cluster_stats_summary.tex\n")

    plt.figure()
    for slug, method in ORDER:
        sub = agg_sorted[(agg_sorted["vecslug"]==slug) & (agg_sorted["method"]==method)]
        if sub.empty:
            continue
        sub = sub.sort_values("K")
        lab = f"{slug} · {method}"
        plt.plot(
            sub["K"], sub["msi"],
            marker=MARKER[method], markersize=MSIZE,
            linewidth=LW[method], alpha=ALPHA[method],
            label=lab, color=COLOR.get(slug, None)
        )

    plt.axhline(0.0, linestyle="--", linewidth=1, color="0.6")
    if HIGHLIGHT_K is not None:
        plt.axvline(HIGHLIGHT_K, linestyle="--", linewidth=1, color="0.85")

    plt.xlabel("K")
    plt.ylabel("MSI (POS − suffix; positive = semantic)")
    plt.title("Semantic–Morphology Index vs K")
    plt.ylim(*YLIM_MSI)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "msi_vs_K.png", dpi=200)
    plt.close()

def run_msi(cfg_path="configs/default.yaml"):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    eval_dir = Path(cfg["paths"]["eval_dir"])
    figdir = Path("reports/figures")
    plot_msi_vs_k(eval_dir, figdir)

if __name__ == "__main__":
    run_msi()
