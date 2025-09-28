from pathlib import Path
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from mkclustering.ud import build_lemma2upos
from sklearn.metrics import (
    normalized_mutual_info_score, v_measure_score,
    homogeneity_score, completeness_score, silhouette_score
)


def purity(y_true, y_pred):
    df = pd.DataFrame({"t": y_true, "c": y_pred})
    total = len(df)
    s = 0
    for _, g in df.groupby("c"):
        s += g["t"].value_counts().max()
    return s / total

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


def run(paths, cfg_path):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lemma2upos = build_lemma2upos(Path(cfg["paths"]["ud_dir"]))

    max_samp = int(cfg.get("eval", {}).get("silhouette_sample", 5000))
    if max_samp > 10000:
        max_samp = 10000

    rows = []
    kv_cache = {}
    clusters_root = Path(cfg["paths"]["clusters_dir"])

    for kdir in sorted(clusters_root.glob("**/K*/")):
        meta = _read_meta(kdir)
        vec_path = meta.get("vec", "")
        vecslug  = meta.get("vecslug", "")
        method   = meta.get("method", "")
        normed   = meta.get("normed", "0") == "1"

        asg = pd.read_csv(kdir / "assignments.csv")
        df = asg[asg["word"].isin(lemma2upos)]
        if len(df) < 100:
            continue

        y_true = df["word"].map(lemma2upos).to_numpy()
        y_pred = df["cluster"].to_numpy()

        res = {
            "vecslug": vecslug,
            "method": method,
            "K": int(kdir.name[1:]),
            "n": int(len(df)),
            "purity": purity(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "V": v_measure_score(y_true, y_pred),
            "homogeneity": homogeneity_score(y_true, y_pred),
            "completeness": completeness_score(y_true, y_pred),
        }

        try:
            if vec_path:
                if vec_path not in kv_cache:
                    kv_cache[vec_path] = _load_kv(vec_path)
                kv = kv_cache[vec_path]

                dfv = df[df["word"].isin(kv.key_to_index)]
                if len(dfv) >= 200 and dfv["cluster"].nunique() >= 2:
                    if len(dfv) > max_samp:
                        dfv = dfv.sample(
                            max_samp,
                            random_state=int(cfg.get("eval", {}).get("random_seed", 42)),
                        )
                    words = dfv["word"].tolist()
                    X = kv[words].astype(np.float32, copy=False)
                    if normed:
                        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                        metric = "euclidean"
                    else:
                        metric = "cosine"
                    y_s = dfv["cluster"].to_numpy()
                    if np.unique(y_s).size >= 2:
                        res["silhouette"] = float(silhouette_score(X, y_s, metric=metric))
        except Exception:
            res["silhouette"] = np.nan

        rows.append(res)

    out = Path(cfg["paths"]["eval_dir"]) / "metrics" / "ud_upos_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["vecslug", "method", "K"]).to_csv(out, index=False)
    print(f"Wrote {out} with {len(rows)} rows")
