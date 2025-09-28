import typer
from pathlib import Path          
from mkclustering.paths import Paths
from mkclustering import corpus, clean, lemma, embeddings, clustering, ud, metrics, probes, bench, viz
import pandas as pd
import yaml

app = typer.Typer(help="Macedonian clustering pipeline CLI")

@app.command()
def fetch(cfg: str = "configs/default.yaml"):
    """Download or validate presence of Leipzig/OSCAR and UD files."""
    paths = Paths(cfg)
    corpus.fetch(paths, cfg)

@app.command()
def clean_text(cfg: str = "configs/default.yaml"):
    """Language-filter, deduplicate, normalize; write data/clean/mk_corpus.txt"""
    paths = Paths(cfg)
    clean.run(paths, cfg)

@app.command()
def lemma_text(cfg: str = "configs/default.yaml"):
    """Lemmatize cleaned text data/clean/mk_corpus.lemma.txt"""
    paths = Paths(cfg)
    lemma.run(paths, cfg)

@app.command()
def download_baseline(cfg: str = "configs/default.yaml"):
    """Fetch pretrained fastText Macedonian vectors into embeddings/baseline/"""
    paths = Paths(cfg)
    embeddings.fetch_baseline(paths)

@app.command()
def train_embeddings(cfg: str = "configs/default.yaml", use_lemma: bool = False):
    """Train fastText on raw or lemmatized corpus."""
    paths = Paths(cfg)
    embeddings.train(paths, cfg, use_lemma=use_lemma)

@app.command()
def cluster(cfg: str = "configs/default.yaml", vectors: str = "trained", method: str = "parallel"):
    """
    Run k-means over chosen vectors:
      vectors: 'trained' | 'baseline' | path to .vec/.bin
      method:  'parallel' | 'serial'
    """
    paths = Paths(cfg)
    clustering.run(paths, cfg, vectors=vectors, method=method)

@app.command()
def eval_intrinsic(cfg: str = "configs/default.yaml"):
    """Compute purity, NMI, V, silhouette using UD labels."""
    paths = Paths(cfg)
    metrics.run(paths, cfg)

@app.command()
def eval_probes(cfg: str = "configs/default.yaml"):
    """Run Top-k neighbor and Odd-One-Out probes."""
    paths = Paths(cfg)
    probes.run(paths, cfg)

@app.command()
def bench_speed(cfg: str = "configs/default.yaml"):
    """Benchmark serial vs parallel k-means runtime/speedup."""
    paths = Paths(cfg)
    bench.run(paths, cfg)

@app.command()
def make_plots(cfg: str = "configs/default.yaml"):
    """Generate PNG figures from CSV metrics into reports/figures/"""
    from mkclustering import plots
    from mkclustering.paths import Paths
    plots.run(Paths(cfg), cfg)

@app.command()
def build_toplist(cfg: str = "configs/default.yaml", use_lemma: bool = False, src: str = ""):
    """Build top-N vocab list from cleaned or lemma corpus."""
    import yaml
    from pathlib import Path
    from mkclustering.corpus import build_top_vocab

    with open(cfg, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)

    if src:
        in_path = Path(src)
    else:
        base = Path(c["paths"]["clean_dir"])
        in_path = base / ("mk_corpus.lemma.txt" if use_lemma else "mk_corpus.txt")

    N = int(c["cluster"].get("max_words", 200000))
    default_out = "data/clean/top200k.lemma.txt" if use_lemma else "data/clean/top200k.txt"
    out = Path(c["cluster"].get("restrict_path", default_out))

    build_top_vocab(in_path, out, top_n=N)
    print(f"[toplist] wrote {out} from {in_path} (top {N})")


@app.command()
def viz_cluster(
    slug: str = typer.Option(..., help="Vector slug, e.g. trained-mk_skipgram"),
    method: str = typer.Option(..., help="serial|parallel"),
    K: int = typer.Option(..., help="Number of clusters"),
    cid: int = typer.Option(..., help="Cluster id to visualize"),
    cfg: str = typer.Option("configs/default.yaml", help="Config path"),
    max_words: int = typer.Option(40, help="Max words to plot"),
    algo: str = typer.Option("pca", help="pca|tsne"),
    show_centroid: bool = typer.Option(False, help="Show centroid marker"),
):
    paths = Paths(cfg)
    kdir = Path(paths.clusters) / slug / method / f"K{K}"
    out = Path("reports/figures") / f"cluster_{slug}_{method}_K{K}_cid{cid}_{algo}.png"
    viz.plot_single_cluster(kdir, cid, out, max_words=max_words, algo=algo, show_centroid=show_centroid)


@app.command()
def align_clusters(
    slug_a: str = typer.Option(..., help="Vec slug A, e.g. trained-mk_skipgram"),
    method_a: str = typer.Option(..., help="serial|parallel for A"),
    slug_b: str = typer.Option(..., help="Vec slug B (usually same model)"),
    method_b: str = typer.Option(..., help="serial|parallel for B"),
    K: int = typer.Option(..., help="Number of clusters"),
    cfg: str = "configs/default.yaml",
):
    """
    Align clusters between two runs (same K) using centroid cosine + Hungarian.
    Writes CSV and prints summary (mean centroid cosine, mean Jaccard).
    """
    from mkclustering.align import align_dirs
    paths = Paths(cfg)
    kdir_a = Path(paths.clusters) / slug_a / method_a / f"K{K}"
    kdir_b = Path(paths.clusters) / slug_b / method_b / f"K{K}"
    out = Path(paths.eval) / "metrics" / f"alignment_{slug_a}_{method_a}__{slug_b}_{method_b}_K{K}.csv"
    summary, df = align_dirs(kdir_a, kdir_b, out_csv=out)
    print(f"[align] wrote {out}")
    print(f"[align] mean centroid cosine: {summary['mean_centroid_cosine']:.4f}")
    print(f"[align] median centroid cosine: {summary['median_centroid_cosine']:.4f}")
    print(f"[align] mean jaccard: {summary['mean_jaccard']:.4f}")


@app.command()
def cluster_stats(
    slug: str = typer.Option(..., help="Vector slug, e.g. trained-mk_skipgram"),
    method: str = typer.Option(..., help="serial|parallel"),
    K: int = typer.Option(..., help="Number of clusters"),
    cfg: str = "configs/default.yaml",
):
    """
    Compute per-cluster morphology/semantics indicators:
      - suffix purity (2/3/4 chars, plus max)
      - POS purity & entropy (from UD)
    Writes a CSV and prints macro averages.
    """
    from mkclustering.clusterstats import per_cluster_stats
    import yaml
    paths = Paths(cfg)
    with open(cfg, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)

    kdir = Path(paths.clusters) / slug / method / f"K{K}"
    df = per_cluster_stats(kdir, Path(c["paths"]["ud_dir"]))
    out = Path(paths.eval) / "metrics" / f"cluster_stats_{slug}_{method}_K{K}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[stats] wrote {out}")

    print(f"[stats] mean max-suffix-purity: {df['max_suffix_purity'].mean():.3f}")
    print(f"[stats] mean pos-purity:        {df['pos_purity'].mean():.3f}")
    print(f"[stats] mean pos-entropy:       {df['pos_entropy'].mean():.3f}")
    print(f"[stats] mean max-suffix-purity: {df['max_suffix_purity'].mean():.3f}")
    print(f"[stats] mean pos-purity:        {df['pos_purity'].mean():.3f}")
    print(f"[stats] mean pos-entropy:       {df['pos_entropy'].mean():.3f} (0–1; higher=mixed)")
    print(f"[stats] mean POS coverage:      {df['pos_coverage'].mean():.3f}")
    print(f"[stats] mean MSI (pos-suf):     {df['msi'].mean():.3f}  (>0 semantic, <0 morphological)")


@app.command()
def make_tables(cfg: str = "configs/default.yaml"):
    """Generate Markdown/CSV/LaTeX summary tables into reports/tables/."""
    from mkclustering import tables
    from mkclustering.paths import Paths
    tables.run(Paths(cfg), cfg)


@app.command()
def suggest_clusters(
    slug: str,
    method: str,
    k: int,
    cfg: str = "configs/default.yaml",
    kind: str = typer.Option("semantic", help="semantic|morphological"),
    topn: int = typer.Option(8, help="how many clusters to list"),
    min_size: int = typer.Option(15, help="minimum words in cluster"),
):
    """Print top-N cluster IDs by MSI (semantic) or -MSI (morphological)."""
    paths = Paths(cfg)
    kdir = Path(paths.clusters) / slug / method / f"K{k}"

    with open(cfg, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    eval_dir = Path(getattr(paths, "eval_dir", c["paths"]["eval_dir"]))

    stats_csv = eval_dir / "metrics" / f"cluster_stats_{slug}_{method}_K{k}.csv"
    if not stats_csv.exists():
        raise SystemExit(
            f"[suggest] Missing {stats_csv}. Run:\n"
            f"  mkcli cluster-stats --slug {slug} --method {method} --k {k}"
        )

    df = viz.suggest_clusters(kdir, stats_csv, kind=kind, topn=topn, min_size=min_size)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))


@app.command()
def viz_top(
    slug: str,
    method: str,
    k: int = typer.Option(..., "--k"),
    kind: str = typer.Option("semantic", help="semantic|morphological"),
    topn: int = 2,
    algo: str = "pca",
    max_words: int = 40,
    cfg: str = "configs/default.yaml",
):
    """Auto-plot the top-N semantic or morphological clusters for a run."""
    paths = Paths(cfg)
    kdir = Path(paths.clusters) / slug / method / f"K{k}"
    stats_csv = Path(paths.eval_dir) / "metrics" / f"cluster_stats_{slug}_{method}_K{k}.csv"
    outdir = Path("reports/figures")
    viz.viz_top_clusters(kdir, stats_csv, outdir, kind=kind, topn=topn, algo=algo, max_words=max_words)


@app.command()
def cluster_sklearn(
    vec_path: str = typer.Option(..., help="Path to .vec or .bin word vectors (fastText/word2vec format)"),
    K: int = typer.Option(..., help="Number of clusters"),
    slug_out: str = typer.Option(..., help="Slug for output, e.g. trained-mk_lemma_skipgram"),
    cfg: str = "configs/default.yaml",
    algorithm: str = typer.Option("lloyd", help="sklearn algorithm: 'lloyd' or 'elkan'"),
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
    n_jobs: int = typer.Option(None, help="Limit BLAS/OpenMP threads (defaults to all cores)"),
    restrict_top: int = typer.Option(None, help="Optional top-N words to keep (overrides config if given)"),
):
    """
    Runs scikit-learn KMeans on L2-normalized vectors and writes outputs in the same
    layout as other runs so downstream eval works:
      clusters/<slug_out>/sklearn/K<K>/{assignments.csv, centroids.npy, cluster_*.txt, meta.yaml}
    """
    import os, time, numpy as np, pandas as pd, yaml
    from pathlib import Path
    from gensim.models import KeyedVectors
    from sklearn.cluster import KMeans
    from mkclustering.paths import Paths

    if n_jobs is not None:
        os.environ["OMP_NUM_THREADS"] = str(n_jobs)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)
        os.environ["NUMEXPR_NUM_THREADS"] = str(n_jobs)

    paths = Paths(cfg)
    vec_path = Path(vec_path)
    binary = vec_path.suffix.lower() == ".bin"

    print(f"[sklearn] loading vectors: {vec_path} (binary={binary})")
    kv = KeyedVectors.load_word2vec_format(str(vec_path), binary=binary)

    topN = restrict_top
    if topN is None:
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f)
            topN = int(c["cluster"].get("max_words", 200000))
        except Exception:
            topN = len(kv.index_to_key)
    words = kv.index_to_key[:topN]
    X = kv.get_normed_vectors()[:topN]  
    if X is None:  
        X = kv.vectors[:topN]
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    print(f"[sklearn] vectors: {X.shape[0]} words x {X.shape[1]} dims (topN={topN})")

    km = KMeans(
        n_clusters=int(K),
        init="k-means++",
        n_init=1,                
        max_iter=int(max_iter),
        tol=float(tol),
        algorithm=algorithm,
        random_state=int(seed),
    )
    t0 = time.time()
    labels = km.fit_predict(X)
    t1 = time.time()
    C = km.cluster_centers_
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    print(f"[sklearn] done K={K} in {t1 - t0:.3f}s  (algorithm={algorithm})")

    kdir = Path(paths.clusters) / slug_out / "sklearn" / f"K{K}"
    kdir.mkdir(parents=True, exist_ok=True)

    asg = pd.DataFrame({"word": words, "cluster": labels.astype(int)})
    asg.to_csv(kdir / "assignments.csv", index=False, encoding="utf-8")
    print(f"[sklearn] wrote {kdir / 'assignments.csv'}")

    for cid in range(int(K)):
        cl_words = asg.loc[asg["cluster"] == cid, "word"].tolist()
        (kdir / f"cluster_{cid}.txt").write_text("\n".join(cl_words), encoding="utf-8")

    np.save(kdir / "centroids.npy", C)

    meta = {
        "tool": "sklearn.KMeans",
        "algorithm": algorithm,
        "seed": seed,
        "max_iter": max_iter,
        "tol": tol,
        "K": int(K),
        "n_jobs_env": n_jobs,
        "vec_path": str(vec_path),
        "n_words": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "runtime_sec": float(t1 - t0),
        "normalized_inputs": True,
        "normalized_centroids": True,
    }
    with open(kdir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)


    runlog = (
        f"vec={vec_path}\n"
        f"vecslug={slug_out}\n"
        f"K={K}\n"
        f"method=sklearn\n"
        f"N={X.shape[0]}\n"
        f"D={X.shape[1]}\n"
        f"normed=1\n"
        f"iterations={km.n_iter_}\n"
        f"num_cores={os.environ.get('OMP_NUM_THREADS','')}\n"
        f"max_iter={max_iter}\n"
        f"secs={t1 - t0:.2f}\n"
        f"max_words={topN}\n"
        f"restrict_path=\n"
    )
    (kdir / "run.log").write_text(runlog, encoding="utf-8")

    print(f"[sklearn] wrote run to {kdir}")


if __name__ == "__main__":
    app()
