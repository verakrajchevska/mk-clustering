from pathlib import Path
from collections import Counter
import conllu

def build_lemma2upos(ud_dir: Path) -> dict[str,str]:
    lemma2upos_counts = {}
    for p in ud_dir.glob("*.conllu"):
        with open(p, "r", encoding="utf-8") as f:
            for sent in conllu.parse(f.read()):
                for tok in sent:
                    if isinstance(tok["id"], int):
                        lemma = tok.get("lemma")
                        upos = tok.get("upostag")
                        if not lemma or not upos: 
                            continue
                        lemma2upos_counts.setdefault(lemma, Counter())[upos] += 1
    return {lem: cnt.most_common(1)[0][0] for lem, cnt in lemma2upos_counts.items()}
