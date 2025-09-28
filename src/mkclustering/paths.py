from pathlib import Path
import yaml

class Paths:
    def __init__(self, cfg="configs/default.yaml"):
        with open(cfg, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        p = self.cfg["paths"]
        self.root = Path(".").resolve()
        self.data = Path(p["data_dir"])
        self.ud = Path(p["ud_dir"])
        self.raw = Path(p["raw_dir"])
        self.clean = Path(p["clean_dir"])
        self.emb = Path(p["embeddings_dir"])
        self.base = Path(p["baseline_dir"])
        self.trained = Path(p["trained_dir"])
        self.clusters = Path(p["clusters_dir"])
        self.eval = Path(p["eval_dir"])

        for d in [self.data,self.ud,self.raw,self.clean,self.emb,self.base,self.trained,self.clusters,self.eval]:
            d.mkdir(parents=True, exist_ok=True)
