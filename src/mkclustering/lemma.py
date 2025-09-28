from pathlib import Path
import sys, time, yaml

def run(paths, cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = Path(cfg["paths"]["clean_dir"]) / "mk_corpus.txt"
    out = Path(cfg["paths"]["clean_dir"]) / "mk_corpus.lemma.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}. Run `mkcli clean-text` first.")

    import classla
    try:
        classla.download("mk")
    except Exception as e:
        print(f"[lemma] classla.download warning: {e}", file=sys.stderr)

    nlp = classla.Pipeline(
        "mk",
        processors="tokenize,pos,lemma",
        use_gpu=bool(cfg.get("lemma", {}).get("use_gpu", False)),
        tokenize_pretokenized=False,
        pos_use_lexicon=False,
        verbose=False,
    )

    t0 = time.time()
    n_lines = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = nlp(line)
            lemmas = []
            for sent in doc.sentences:
                lemmas.extend([w.lemma for w in sent.words if w.lemma])
            if lemmas:
                fout.write(" ".join(lemmas) + "\n")
            n_lines += 1
            if n_lines % 1000 == 0:
                print(f"[classla] {n_lines} lines...", file=sys.stderr)

    print(f"[lemma] Wrote {out} using CLASSLA in {time.time()-t0:.1f}s")
