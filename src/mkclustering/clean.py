from pathlib import Path
import hashlib, yaml
from mkclustering.corpus import normalize as mk_normalize, is_macedonian

def run(paths, cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    corpus_cfg = cfg.get("corpus", {}) or {}
    raw_text = corpus_cfg.get("raw_text") or str(Path(paths.raw) / "leipzig_all.txt")
    raw_file = Path(raw_text)
    out_file = Path(cfg["paths"]["clean_dir"]) / "mk_corpus.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        raise SystemExit(f"Missing input: {raw_file}. Put Leipzig text there first.")

    min_len      = int(corpus_cfg.get("min_line_len", 20))
    lowercase    = bool(corpus_cfg.get("lowercase", True))
    sample_n     = int(corpus_cfg.get("sample_lines", 0)) or None
    lang_filter  = bool(corpus_cfg.get("lang_filter", False))  

    seen, kept, total = set(), 0, 0

    with raw_file.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            if sample_n and total > sample_n:
                break

            line = line.strip()
            if not line:
                continue

            line = mk_normalize(line, lowercase=lowercase)
            if len(line) < min_len:
                continue

            if lang_filter and not is_macedonian(line):
                continue

            h = hashlib.md5(line.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            fout.write(line + "\n")
            kept += 1
            if kept % 100000 == 0:
                print(f"[clean] kept {kept} / seen {total}")

    print(f"[clean] wrote {out_file} | kept {kept} lines (from {total}, uniques={len(seen)})")
