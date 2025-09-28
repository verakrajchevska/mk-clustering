from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _read_meta(kdir: Path) -> dict:
    meta = {}
    log = kdir / "run.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def _load_kv(path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=False)


def _centroids(kdir: Path):
    """
    Returns:
      C: (K,D) centroid matrix (row-normalized)
      wordsets: list[set[str]] words per cluster id (0..K-1)
      meta: dict from run.log
    """
    meta = _read_meta(kdir)
    vec_path = meta.get("vec", "")
    if not vec_path:
        raise SystemExit(f"[align] missing 'vec=' in {kdir}/run.log")
    kv = _load_kv(vec_path)

    asg = pd.read_csv(kdir / "assignments.csv") 
    K = int(meta.get("K", asg["cluster"].max() + 1))

    asg = asg[asg["word"].isin(kv.key_to_index)]
    if asg.empty:
        raise SystemExit(f"[align] no words from assignments found in vectors for {kdir}")

    wordsets = [set() for _ in range(K)]
    for w, c in asg[["word", "cluster"]].itertuples(index=False):
        wordsets[int(c)].add(w)

    D = kv.vector_size
    C = np.zeros((K, D), dtype=np.float32)
    for cid in range(K):
        words = list(wordsets[cid])
        if not words:
            continue
        X = kv[words].astype(np.float32, copy=False)
        if meta.get("normed", "0") == "1":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        C[cid] = X.mean(axis=0)

    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C, wordsets, meta


def _hungarian_max(sim):
    """maximize total similarity; sim is (K,K)"""
    if linear_sum_assignment is None:
        K = sim.shape[0]
        used_r, used_c = set(), set()
        pairs = []
        for _ in range(K):
            i, j = divmod(sim.argmax(), sim.shape[1])
            while (i in used_r) or (j in used_c):
                sim[i, j] = -1e9
                i, j = divmod(sim.argmax(), sim.shape[1])
            used_r.add(i); used_c.add(j)
            pairs.append((i, j))
            sim[i, j] = -1e9
        rows, cols = zip(*pairs)
        return np.array(rows), np.array(cols)
    else:
        rows, cols = linear_sum_assignment(-sim)
        return rows, cols


def align_dirs(kdir_a: Path, kdir_b: Path, out_csv: Path | None = None):
    """
    Align clusters between two runs (same K) and write per-cluster stats.
    Return a dict summary.
    """
    Ca, sets_a, meta_a = _centroids(kdir_a)
    Cb, sets_b, meta_b = _centroids(kdir_b)

    if Ca.shape[0] != Cb.shape[0]:
        raise SystemExit("[align] K mismatch between runs")

    S = Ca @ Cb.T  # rows are normalized - cosine

    # hungarian alignment to maximize similarity
    ra, cb = _hungarian_max(S.copy())
    sim_matched = S[ra, cb]

    # jaccard overlap of word sets
    jac = []
    size_a, size_b = [], []
    for i, j in zip(ra, cb):
        A = sets_a[i]; B = sets_b[j]
        inter = len(A & B)
        union = len(A | B) or 1
        jac.append(inter / union)
        size_a.append(len(A)); size_b.append(len(B))

    df = pd.DataFrame({
        "cid_a": ra,
        "cid_b": cb,
        "centroid_cosine": sim_matched,
        "jaccard": jac,
        "size_a": size_a,
        "size_b": size_b,
    }).sort_values("cid_a")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    summary = {
        "mean_centroid_cosine": float(np.mean(sim_matched)),
        "median_centroid_cosine": float(np.median(sim_matched)),
        "mean_jaccard": float(np.mean(jac)),
        "pairs": len(sim_matched),
        "meta_a": meta_a,
        "meta_b": meta_b,
    }
    return summary, df
