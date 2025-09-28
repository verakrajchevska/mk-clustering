from pathlib import Path
import subprocess, shlex, os, sys, json
from mkclustering.paths import Paths
import gensim

BASELINE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mk.300.vec.gz" 

def fetch_baseline(paths: Paths):
    paths.base.mkdir(parents=True, exist_ok=True)
    out = paths.base / "cc.mk.300.vec.gz"
    if not out.exists():
        import requests
        r = requests.get(BASELINE_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        print(f"Downloaded baseline to {out}")
    else:
        print(f"Baseline already present at {out}")

def train(paths: Paths, cfg_path: str, use_lemma: bool = False):
    import yaml, fasttext
    from pathlib import Path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]; e = cfg["embeddings"]

    inp = Path(p["clean_dir"]) / ("mk_corpus.lemma.txt" if use_lemma else "mk_corpus.txt")
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    Path(p["trained_dir"]).mkdir(parents=True, exist_ok=True)

    for model in e["models"]:
        out_prefix = Path(p["trained_dir"]) / f"mk_{'lemma_' if use_lemma else ''}{model}"
        ft = fasttext.train_unsupervised(
            input=str(inp),
            model=("cbow" if model == "cbow" else "skipgram"),
            dim=int(e["dim"]),
            epoch=int(e["epoch"]),
            lr=float(e["lr"]),
            ws=int(e["ws"]),
            minCount=int(e["min_count"]),
            thread=int(e["threads"]),
        )
        ft.save_model(str(out_prefix) + ".bin")

        vocab = ft.get_words()
        dim = int(e["dim"])
        with open(str(out_prefix) + ".vec", "w", encoding="utf-8") as f:
            f.write(f"{len(vocab)} {dim}\n")
            for w in vocab:
                vec = ft.get_word_vector(w)
                f.write(w + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
        print(f"[fastText] wrote {out_prefix}.bin and {out_prefix}.vec")

