from pathlib import Path
import os, time
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
try:
    from gensim.models.fasttext import load_facebook_vectors as _load_ft_bin
except Exception:
    _load_ft_bin = None

from .parallel_kmeans_impl import ParallelKMeans


def _vec_slug(vec_path: Path) -> str:
    p = Path(vec_path)
    parent = p.parent.name             
    stem = p.name.replace(".vec.gz","").replace(".vec","").replace(".bin","")
    return f"{parent}-{stem}"          


def _find_trained(paths, want_lemma: bool) -> Path:
    cands = sorted(Path(paths.trained).glob("mk_*skipgram.vec"))
    cands = [p for p in cands if ("lemma" in p.name) == want_lemma]
    if not cands:
        kind = "lemma" if want_lemma else "raw"
        raise SystemExit(f"No {kind} trained vectors in {paths.trained}")
    return cands[0]

def _pick_vec_path(paths, cfg, which: str) -> Path:
    if which == "trained-lemma":
        return _find_trained(paths, want_lemma=True)
    if which == "trained":
        return _find_trained(paths, want_lemma=False)
    if which == "baseline":
        base_dir = getattr(paths, "base", None) or Path(cfg["paths"]["baseline_dir"])
        cand = sorted(Path(base_dir).glob("*.vec*"))
        if not cand:
            raise SystemExit(f"No baseline vectors in {base_dir}")
        return cand[0]
    p = Path(which)
    if not p.exists():
        raise SystemExit(f"Vector path does not exist: {p}")
    return p


def load_vecs(vec_path: Path, max_words: int | None = None):
    """
    Loads word2vec-format text (.vec or .vec.gz).
    If we pass a fastText .bin and gensim supports it, it is loaded too.
    Returns: (words: list[str], X: np.ndarray[float32])
    """
    vec_path = Path(vec_path)
    if vec_path.suffix == ".bin":
        if _load_ft_bin is None:
            raise SystemExit("fastText .bin support not available; use a .vec/.vec.gz instead.")
        kv = _load_ft_bin(str(vec_path))
    else:
        kv = KeyedVectors.load_word2vec_format(str(vec_path), binary=False)

    words = list(kv.index_to_key)
    if max_words:
        words = words[:max_words]
    X = kv[words].astype(np.float32, copy=False)
    return words, X



def _auto_restrict_path(vec_path: Path) -> Path:
    """Pick a sensible default toplist if user didn't set one."""
    name = Path(vec_path).name.lower()
    if "lemma" in name:
        return Path("data/clean/top200k.lemma.txt")
    return Path("data/clean/top200k.txt")


def _apply_toplist_cap(words: list[str], X: np.ndarray, restrict_path: Path | None):
    """
    If restrict_path exists, keep only tokens listed there (order-preserving).
    Returns filtered (words, X) and the path used (or None).
    """
    if restrict_path and restrict_path.exists():
        keep = {
            w.strip()
            for w in restrict_path.read_text(encoding="utf-8").splitlines()
            if w.strip()
        }
        if keep:
            mask = [w in keep for w in words]
            if any(mask):
                words = [w for w, m in zip(words, mask) if m]
                X = X[mask]
                return words, X, restrict_path
    return words, X, None


def run(paths, cfg_path, vectors="trained", method="parallel"):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vec = _pick_vec_path(paths, cfg, vectors)

    max_words = int(cfg["cluster"].get("max_words", 0)) or None
    words, X = load_vecs(vec, max_words=max_words)

    restrict_cfg = (cfg["cluster"].get("restrict_path", "") or "").strip()
    restrict_path = Path(restrict_cfg) if restrict_cfg else _auto_restrict_path(vec)
    words, X, used_toplist = _apply_toplist_cap(words, X, restrict_path)

    norm = bool(cfg["cluster"].get("normalize", True))
    if norm:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    slug = _vec_slug(vec)
    k_values = cfg["cluster"]["k_values"]

    seed = int(cfg.get("eval", {}).get("random_seed", 42))
    np.random.seed(seed)

    max_iter = int(cfg["cluster"].get("max_iter", 100))

    for K in k_values:
        if len(words) < K:
            print(f"[cluster] skip K={K}: only {len(words)} vectors after cap")
            continue

        outdir = Path(paths.clusters) / slug / method / f"K{K}"
        outdir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        if method == "serial":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=10, max_iter=max_iter, random_state=seed, tol=1e-4)
            labels = km.fit_predict(X)
            iters = int(km.n_iter_)
            num_cores = None
        elif method == "parallel":
            num_cores = int(cfg["cluster"].get("num_cores", os.cpu_count() or 1))
            pk = ParallelKMeans(n_clusters=K, max_iter=max_iter, num_cores=num_cores)
            pk.fit(X)
            labels = pk.predict(X)
            iters = int(pk.iterations)
        else:
            raise SystemExit(f"Unknown method '{method}'")

        secs = time.time() - t0

        pd.DataFrame({"word": words, "cluster": labels}).to_csv(outdir / "assignments.csv", index=False)
        with open(outdir / "run.log", "w", encoding="utf-8") as f:
            f.write(f"vec={vec}\n")
            f.write(f"vecslug={slug}\n")
            f.write(f"K={K}\n")
            f.write(f"method={method}\n")
            f.write(f"N={len(words)}\n")
            f.write(f"D={X.shape[1]}\n")
            f.write(f"normed={1 if norm else 0}\n")
            f.write(f"iterations={iters}\n")
            if num_cores is not None:
                f.write(f"num_cores={num_cores}\n")
            f.write(f"max_iter={max_iter}\n")
            f.write(f"secs={secs:.2f}\n")
            f.write(f"max_words={max_words or 'all'}\n")
            if used_toplist:
                f.write(f"restrict_path={used_toplist}\n")

        print(f"K={K} done in {secs:.2f}s >> {outdir}")
