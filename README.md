# Accelerating Word Embedding Clustering with Parallel k-Means: A Case Study on the Macedonian Corpus

[![DOI](https://zenodo.org/badge/1065995902.svg)](https://doi.org/10.5281/zenodo.17220854)

This repository provides a fully reproducible pipeline for cleaning a Macedonian web corpus, lemmatizing it, training word embeddings (fastText; raw and lemma variants), and clustering vectors with serial, parallel, and scikit-learn K-Means. It includes intrinsic evaluation, alignment diagnostics for morphology/semantics, visualizations, and runtime benchmarking.

A key contribution is a parallel k-means implementation that accelerates the assignment/update steps across CPU cores while preserving scikit-learn-compatible behavior. The code exposes a simple CLI and drop-in API, making it easy to compare serial vs. parallel clustering at different K, vector types, and hardware settings.

The pipeline is packaged as a small Python module with a Typer CLI: **`mkcli`**.


## Contents

- [Quick start](#quick-start)
- [Setup](#setup)
- [Data: Leipzig + UD](#data-leipzig--ud)
- [Pipeline overview](#pipeline-overview)
- [Commands (CLI)](#commands-cli)
- [Outputs & layout](#outputs--layout)
- [Reproducing paper figures & tables](#reproducing-paper-figures--tables)
- [Tips & troubleshooting](#tips--troubleshooting)
- [Citing / Acknowledgements](#citing--acknowledgements)

---

## Quick start

```bash
# 1) Create & activate a venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install requirements
pip install -U pip
pip install -e ".[all]"

# 3) Put raw Leipzig text here (see below for how to obtain it)
#    data/raw/leipzig_all.txt  (UTF-8 plain text, one sentence/line is fine)

# 4) Optional: download the baseline fastText vectors
mkcli download-baseline

# 5) Clean and lemmatize
mkcli clean-text
mkcli lemma-text

# 6) Train fastText (raw & lemma)
mkcli train-embeddings              # raw
mkcli train-embeddings --use-lemma  # lemma

# 7) Cluster (serial & parallel) using values of K from configs/default.yaml
mkcli cluster --vectors trained --method serial
mkcli cluster --vectors trained --method parallel
mkcli cluster --vectors trained-lemma --method serial
mkcli cluster --vectors trained-lemma --method parallel

# 8) Optional: scikit-learn KMeans 
mkcli cluster-sklearn --vec-path embeddings/trained/mk_skipgram.vec --K 800 --slug-out trained-mk_skipgram --algorithm elkan --n-jobs 4
mkcli cluster-sklearn --vec-path embeddings/trained/mk_lemma_skipgram.vec --K 800 --slug-out trained-mk_lemma_skipgram --algorithm elkan --n-jobs 4

# 9) Intrinsic evaluation + plots + tables
mkcli eval-intrinsic
mkcli bench-speed
mkcli make-plots
mkcli make-tables
```

> **Where is `mkcli`?** It’s the Typer CLI defined in `src/mkclustering/cli.py`. When you install this repo in editable mode (`pip install -e .`), `mkcli` becomes available on your `$PATH`.

---

## Setup

### Python & OS
- Python 3.9–3.11 tested
- Linux/macOS recommended (Windows WSL works)

### Install
```bash
git clone https://github.com/verakrajchevska/mk-clustering.git mk-clustering
cd mk-clustering
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[all]"
```

> The `.[all]` extra installs: `fasttext`, `gensim`, `scikit-learn`, `typer`, `pandas`, `numpy`, `matplotlib`, `classla`, `conllu`, etc.

### Configuration
Edit `configs/default.yaml` if you need to change folders, K grid, threads, etc.
Key bits:
```yaml
paths:
  data_dir: data
  raw_dir: data/raw
  clean_dir: data/clean
  ud_dir: data/ud
  embeddings_dir: embeddings
  baseline_dir: embeddings/baseline
  trained_dir: embeddings/trained
  clusters_dir: clusters
  eval_dir: eval

cluster:
  k_values: [200, 400, 800, 1200, 2000]
  normalize: true
  max_iter: 100
  num_cores: 4          # for parallel variant
  max_words: 200000     # vocabulary cap (top-N)
  restrict_path: ""     # optional path to toplist (auto-chosen if blank)

embeddings:
  models: [skipgram]    # or ["cbow","skipgram"]
  dim: 300
  ws: 5
  epoch: 10
  lr: 0.05
  min_count: 5
  threads: 8

eval:
  random_seed: 42
  silhouette_sample: 5000
```

---

## Data: Leipzig + Universal Dependencies (UD)

### Leipzig Macedonian web text
Download Macedonian web data from the Leipzig Corpora Collection:  
<https://wortschatz.uni-leipzig.de/en/download/mkd>

For reproducibility, concatenate or convert the 1M slice you choose to a single UTF‑8 text file and place it at:
```
data/raw/leipzig_all.txt
```
(One sentence/line is fine; the cleaner will normalize.)

### Universal Dependencies (UD)
Place a Macedonian UD CoNLL‑U file (e.g., `mk_mtb-ud-test.conllu`) into:
```
data/ud/
```
Used for POS-based metrics and MSI.

> If you already have a different UD treebank, any `.conllu` under `data/ud/` will be read.

---

## Pipeline overview

1. **Clean** (deduplicate, Unicode NFKC, strip URLs/emails/punct, lowercase, optional language filter) → `data/clean/mk_corpus.txt`
2. **Lemmatize** with CLASSLA‑Stanza (Macedonian model) → `data/clean/mk_corpus.lemma.txt`
3. **Train fastText** (raw + lemma) → `embeddings/trained/*.bin/.vec`
4. **Cluster** with:
   - serial (scikit‑learn KMeans)
   - parallel (multiprocessing master–worker, cosine‑equivalent on L2‑normalized vectors)
   - optional **sklearn** “baseline” runner
5. **Evaluate**: POS purity, suffix purity (2/3/4 chars and max), MSI (=POS − suffix), silhouette, NMI, V, homogeneity, completeness
6. **Benchmark** runtime and compute speedup
7. **Visualize** clusters (PCA/t‑SNE) and plot metrics vs. K
8. **Align** serial ↔ parallel clusterings at fixed K with Hungarian matching (centroid cosine, Jaccard overlap)

---

## Commands (CLI)

### Clean & Lemmatize
```bash
mkcli clean-text
mkcli lemma-text
```
- `clean-text` reads `data/raw/leipzig_all.txt` and writes `data/clean/mk_corpus.txt`.
- `lemma-text` runs CLASSLA (tokenize, POS, lemma) and writes `data/clean/mk_corpus.lemma.txt`.

### Baseline vectors
```bash
mkcli download-baseline
```
Downloads `cc.mk.300.vec.gz` to `embeddings/baseline/`.

### Train embeddings
```bash
mkcli train-embeddings              # raw
mkcli train-embeddings --use-lemma  # lemma
```
Writes `embeddings/trained/mk_*{cbow,skipgram}.{bin,vec}`.

### Build top‑N vocabulary (optional)
```bash
mkcli build-toplist --use-lemma      # or omit for raw
```
Writes `data/clean/top200k{.lemma}.txt` and is auto‑picked during clustering.

### Cluster
```bash
# raw, serial
mkcli cluster --vectors trained --method serial
# raw, parallel
mkcli cluster --vectors trained --method parallel
# lemma, serial
mkcli cluster --vectors trained-lemma --method serial
# lemma, parallel
mkcli cluster --vectors trained-lemma --method parallel
```
Outputs per (model,method,K) under `clusters/<slug>/<method>/K<...>/` with `assignments.csv` and `run.log`.

### scikit‑learn comparison
```bash
mkcli cluster-sklearn --vec-path embeddings/trained/mk_skipgram.vec \
  --K 800 --slug-out trained-mk_skipgram --algorithm elkan --n-jobs 4
```
Writes `assignments.csv`, `centroids.npy`, `cluster_*.txt`, `meta.yaml`, and `run.log` under `clusters/<slug>/sklearn/K<...>/`.

### Intrinsic metrics
```bash
mkcli eval-intrinsic
```
Creates `eval/metrics/ud_upos_metrics.csv` (purity, NMI, V, homogeneity, completeness, silhouette).

### Cluster morphology/semantics stats
```bash
mkcli cluster-stats --slug trained-mk_skipgram --method parallel --k 800
```
Creates `eval/metrics/cluster_stats_<slug>_<method>_K<k>.csv` (suffix purities, POS stats, MSI).

### Alignment (serial ↔ parallel)
```bash
mkcli align-clusters \
  --slug-a trained-mk_skipgram --method-a serial \
  --slug-b trained-mk_skipgram --method-b parallel --k 800
```
Creates `eval/metrics/alignment_<...>.csv` with per‑cluster matched centroid cosine and Jaccard; prints macro means.

### Benchmarks, plots & tables
```bash
mkcli bench-speed
mkcli make-plots     # figures in reports/figures
mkcli make-tables    # CSV/MD/LaTeX in reports/tables
```

### Visualize clusters
```bash
mkcli viz-cluster --slug trained-mk_lemma_skipgram --method parallel --k 200 --cid 15 --max-words 40 --show-centroid
# or auto‑pick top semantic/morphological clusters and plot
mkcli viz-top trained-mk_skipgram parallel --k 200 --kind morphological --topn 2
```

---

## Outputs & layout

```
data/
  raw/leipzig_all.txt
  clean/mk_corpus.txt
  clean/mk_corpus.lemma.txt
  clean/top200k.txt
  clean/top200k.lemma.txt
  ud/*.conllu

embeddings/
  baseline/cc.mk.300.vec.gz
  trained/mk_{,lemma_}{cbow,skipgram}.{bin,vec}

clusters/<slug>/{serial,parallel,sklearn}/K{200,400,800,1200,2000}/
  assignments.csv
  run.log
  [sklearn only] centroids.npy, cluster_*.txt, meta.yaml

eval/metrics/
  ud_upos_metrics.csv
  cluster_stats_*_K*.csv
  alignment_*_K*.csv
  benchmarks.csv, benchmarks_pivot.csv

reports/
  figures/*.png
  tables/*.csv|*.md|*.tex
```

---

## Helper scripts (optional)

These small utilities are handy when you want to quickly pick clusters to visualize, find aligned cluster IDs between runs, or list the most centroid-similar words inside a cluster. They assume you’ve already generated the relevant `clusters/**/K*/assignments.csv` and `eval/metrics/*.csv`.

> Run them from the repository root (they use relative paths). Activate your venv first: `source .venv/bin/activate`.

### 1) `choosing_cid.py`
Reads `eval/metrics/cluster_stats_*_K200.csv` for **raw** and **lemma** (parallel, K=200 by default) and prints:
- the most **morphological** raw cluster (lowest MSI),
- the most **semantic** lemma cluster (highest MSI),
plus ready-to-copy `mkcli viz-cluster` commands.

Example:
```bash
python choosing_cid.py
# RAW / parallel / K=200 == most morphological
cid=103, MSI=-0.421
# LEMMA / parallel / K=200 == most semantic
cid=196, MSI=0.287
```

### 2) `clusters_aligned_match.py`
Given an alignment CSV (e.g., produced by `mkcli align-clusters` for raw K=200 serial↔parallel), it prints which cluster in run B matches a specific cid in run A, along with centroid cosine.

Example:
```bash
# First create the alignment CSV:
mkcli align-clusters \
  --slug-a trained-mk_skipgram --method-a serial \
  --slug-b trained-mk_skipgram --method-b parallel --k 200

# Then query a cid:
python clusters_aligned_match.py
serial cid 147 ↔ parallel cid 103  (centroid cosine = 0.9514)

```

### 3) `list_top_words_in_K.py`
Lists words in `cid` (default in the script) sorted by cosine to the cluster centroid, helpful for sanity-checking cluster coherence or labeling.

Example:
```bash
python list_top_words_in_K.py | head -n 50
# повлекување 0.921
# спојување 0.911
# вметнување 0.904
# ...
```

## Reproducing paper figures & tables

- **MSI vs K**: produced by `plots_msi.py` → `reports/figures/msi_vs_K.png`
- **Runtime vs K / Speedup**: `mkcli bench-speed` then `mkcli make-plots` → `runtime_vs_K.png`, `speedup_vs_K.png`
- **Cluster‑stats summary**: `mkcli make-tables` → `reports/tables/cluster_stats_summary.{csv,md,tex}`
- **Alignment table**: `mkcli make-tables` → `reports/tables/alignment_summary.{csv,md,tex}`
- **sklearn vs parallel**: run `cluster-sklearn` for K in {200,800,2000}, then `cluster-stats`, then assemble a compact table from the generated CSVs (or adapt `tables.py` to filter `method in {parallel,sklearn}` and `K in {200,800,2000}`).

---

## Tips & troubleshooting

- **Leipzig download**: the site serves links via a form. Download the “1M” Macedonian web sample manually and place the unpacked text under `data/raw/leipzig_all.txt`.
- **Toplist cap**: to keep memory bounded and make runs comparable, the pipeline caps to the top‑N tokens. Edit `cluster.max_words` or provide your own list via `cluster.restrict_path`.
- **Normalization**: vectors are L2‑normalized before clustering. Euclidean distance on the unit sphere is equivalent to cosine similarity.
- **Parallel cores**: set `cluster.num_cores` in the config for the parallel variant. For sklearn baselines, limit BLAS/OpenMP threads with `--n-jobs` (the command sets `OMP_NUM_THREADS`, etc.).
- **Reproducibility**: seeds are fixed where applicable; `run.log`/`meta.yaml` capture the parameters, iterations, and wall‑clock times.
- **Disk footprint**: large artifacts (full embeddings, cluster word lists) can be heavy. See `.gitignore` (provided) for sensible defaults that keep `assignments.csv`, `run.log`, and `meta.yaml` only.

---

## Citing / Acknowledgements

If you use this pipeline, please cite the accompanying paper and the tools we rely on:

- **fastText** for training word embeddings  
- **CLASSLA‑Stanza** for Macedonian lemmatization  
- **scikit‑learn** and **gensim** for clustering & vector I/O  
- **Universal Dependencies** for POS supervision

This work uses the **Leipzig Corpora Collection** Macedonian web data for training/evaluation.

---

## License

The code in this repository is released under the [MIT License](https://opensource.org/license/mit).
The datasets and pretrained models referenced or downloaded by the pipeline
(Leipzig Corpora, fastText vectors, Universal Dependencies, and
CLASSLA/Stanza models) are **not** covered by this license and remain under
their respective licenses.
