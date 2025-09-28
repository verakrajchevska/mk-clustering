from __future__ import annotations
from pathlib import Path
import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


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


def _load_pos_map(ud_dir: Path) -> dict[str, str]:
    """
    Build a word to UPOS lookup from all *.conllu under ud_dir.
    We map BOTH surface form and lemma (lowercased).
    """
    from collections import defaultdict, Counter
    pos_counts = defaultdict(Counter)
    for conllu in Path(ud_dir).glob("**/*.conllu"):
        for line in conllu.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:
                continue
            form = cols[1].lower()
            lemma = cols[2].lower()
            upos = cols[3]
            pos_counts[form].update([upos])
            pos_counts[lemma].update([upos])
    pos_map = {w: cnt.most_common(1)[0][0] for w, cnt in pos_counts.items()}
    return pos_map



def per_cluster_stats(kdir: Path, ud_dir: Path) -> pd.DataFrame:
    meta = _read_meta(kdir)
    asg = pd.read_csv(kdir / "assignments.csv")  
    kv = _load_kv(meta["vec"])
    pos_map = _load_pos_map(ud_dir)

    rows = []
    for cid, g in asg.groupby("cluster"):
        words_all = g["word"].tolist()
        words = [w for w in words_all if w in kv.key_to_index]

        def _suffix_purity(words, L):
            cand = [w[-L:] for w in words if len(w) >= L]
            if not cand:
                return "", 0.0
            from collections import Counter
            suf, n = Counter(cand).most_common(1)[0]
            return suf, n / len(cand)

        suf2, p2 = _suffix_purity(words, 2)
        suf3, p3 = _suffix_purity(words, 3)
        suf4, p4 = _suffix_purity(words, 4)
        max_suf = max(p2, p3, p4)

        tags = [pos_map.get(w.lower(), None) for w in words]
        tags = [t for t in tags if t is not None]
        covered = len(tags)
        if covered:
            from collections import Counter
            cnt = Counter(tags)
            top_pos, top_n = cnt.most_common(1)[0]
            pos_purity = top_n / covered
            import numpy as np, math
            p = np.array([c / covered for c in cnt.values()], dtype=float)
            H = float(-(p * np.log(p + 1e-12)).sum())
            pos_entropy = H / math.log(17.0)  # 17 UPOS tags
        else:
            top_pos, pos_purity, pos_entropy = "", 0.0, 0.0

        rows.append({
            "cid": int(cid),
            "size": len(words_all),
            "suffix2": suf2, "suffix2_purity": p2,
            "suffix3": suf3, "suffix3_purity": p3,
            "suffix4": suf4, "suffix4_purity": p4,
            "max_suffix_purity": max_suf,
            "top_pos": top_pos,
            "pos_purity": pos_purity,
            "pos_entropy": pos_entropy,
            "pos_coverage": covered / max(1, len(words_all)),
            "msi": pos_purity - max_suf, 
        })
    return pd.DataFrame(rows).sort_values("cid")