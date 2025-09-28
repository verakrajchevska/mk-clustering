from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def _read_meta(kdir: Path) -> dict:
    meta = {}
    log = kdir / "run.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta

def _load_kv(vec_path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(vec_path, binary=False)

def _get_cluster_words_and_X(kdir: Path, cid: int, max_words: int, seed: int):
    asg = pd.read_csv(kdir / "assignments.csv")  
    sub = asg[asg["cluster"] == cid]
    if sub.empty:
        raise SystemExit(f"[viz] cluster {cid} not found in {kdir}")
    if max_words and len(sub) > max_words:
        sub = sub.sample(max_words, random_state=seed)

    meta = _read_meta(kdir)
    vec_path = meta.get("vec", "")
    if not vec_path:
        raise SystemExit(f"[viz] missing 'vec=' line in {kdir}/run.log")
    kv = _load_kv(vec_path)

    words = [w for w in sub["word"].tolist() if w in kv.key_to_index]
    if len(words) < 2:
        raise SystemExit("[viz] fewer than 2 words available for this cluster.")
    X = kv[words].astype(np.float32, copy=False)

    if meta.get("normed", "0") == "1":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return words, X, meta

def _project_2d(X: np.ndarray, algo: str = "pca", seed: int = 42) -> np.ndarray:
    if algo.lower() == "tsne":
        perplexity = max(5, min(30, (X.shape[0] - 1) // 3 or 5))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=seed, learning_rate="auto")
        return tsne.fit_transform(X)
    return PCA(n_components=2, random_state=seed).fit_transform(X)

def plot_single_cluster(kdir: Path, cid: int, outpath: Path,
                        max_words: int = 40, algo: str = "pca", seed: int = 42,
                        annotate: bool = True, show_centroid: bool = False):
    words, X, meta = _get_cluster_words_and_X(kdir, cid, max_words, seed)
    Y = _project_2d(X, algo=algo, seed=seed)

    plt.figure(figsize=(7, 6))
    plt.scatter(Y[:, 0], Y[:, 1], s=28, alpha=0.75)
    if annotate:
        for (x, y), w in zip(Y, words):
            plt.text(x + 0.01, y + 0.01, w, fontsize=9, alpha=0.9)
    if show_centroid:
        c = Y.mean(axis=0)
        plt.scatter([c[0]], [c[1]], s=90, marker="X")

    title = f"{meta.get('vecslug','?')} · {meta.get('method','?')} · K={meta.get('K','?')} · cid={cid} · {algo.upper()}"
    plt.title(title)
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[viz] wrote {outpath}")


def _read_stats_csv(stats_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_csv)
    rename = {}
    if "cid" in df.columns and "cluster" not in df.columns:
        rename["cid"] = "cluster"
    if "suffix_purity" not in df.columns:
        for c in df.columns:
            if c.startswith("suffix") and "purity" in c:
                rename[c] = "suffix_purity"
    if "pos_purity" not in df.columns:
        for c in df.columns:
            if c.startswith("pos") and "purity" in c:
                rename[c] = "pos_purity"
    if rename:
        df = df.rename(columns=rename)
    if "cluster" not in df.columns:
        raise SystemExit(f"[stats] {stats_csv} has no 'cluster' (cid) column.")
    return df


def suggest_clusters(kdir: Path, stats_csv: Path, kind: str = "semantic",
                     topn: int = 8, min_size: int = 15) -> pd.DataFrame:
    """
    Return a small table of candidate cluster IDs to visualize.
    kind: 'semantic' (high MSI) or 'morphological' (low MSI / high suffix signal).
    Works with CSVs that have 'cluster' or 'cid' as the id column.
    """
    df = pd.read_csv(stats_csv)

    if "cluster" in df.columns:
        id_col = "cluster"
    elif "cid" in df.columns:
        df = df.rename(columns={"cid": "cluster"})
        id_col = "cluster"
    else:
        raise SystemExit(
            f"[suggest] {stats_csv} has columns {list(df.columns)} — "
            "expected an id column named 'cluster' or 'cid'."
        )

    if "msi" not in df.columns:
        raise SystemExit(f"[suggest] {stats_csv} is missing 'msi'. Re-run: mkcli cluster-stats ...")

    if "size" in df.columns:
        df = df[df["size"] >= min_size].copy()

    if kind == "semantic":
        df["_score"] = df["msi"]      
    else:
        df["_score"] = -df["msi"]     

    sort_cols = ["_score"]
    sort_asc  = [False]

    has_pos    = "pos_purity" in df.columns
    has_mxsuf  = "max_suffix_purity" in df.columns

    if kind == "semantic":
        if has_pos:  
            sort_cols += ["pos_purity"];          sort_asc += [False]
        if has_mxsuf: 
            sort_cols += ["max_suffix_purity"];   sort_asc += [True]
    else:  
        if has_mxsuf: 
            sort_cols += ["max_suffix_purity"];   sort_asc += [False]
        if "size" in df.columns:
            sort_cols += ["size"];                sort_asc += [False]

    df = df.sort_values(sort_cols, ascending=sort_asc)

    keep = [c for c in [
        id_col, "size",
        "top_pos", "pos_purity", "pos_entropy", "pos_coverage",
        "max_suffix_purity", "msi", "_score"
    ] if c in df.columns]

    out = df[keep].head(topn).copy()

    if "cid" in pd.read_csv(stats_csv, nrows=0).columns:
        out = out.rename(columns={"cluster": "cid"})

    return out


def viz_top_clusters(kdir: Path, stats_csv: Path, outdir: Path,
                     kind: str = "semantic", topn: int = 2,
                     algo: str = "pca", max_words: int = 40, seed: int = 42):
    sel = suggest_clusters(kdir, stats_csv, kind=kind, topn=topn)
    outdir.mkdir(parents=True, exist_ok=True)
    for _, r in sel.iterrows():
        cid = int(r["cluster"])
        out = outdir / f"{kdir.parent.name}_{kdir.name}_cid{cid}_{kind}_{algo}.png"
        plot_single_cluster(kdir, cid, out, max_words=max_words, algo=algo, seed=seed, show_centroid=False)
        print(f"[viz-top] {kind}: cid={cid} (size={int(r['size'])}, MSI={r['msi']:.3f}) -> {out}")

