from pathlib import Path
import pandas as pd

def _parse_log(log: Path) -> dict:
    meta = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

    def _i(k, default=0):
        try:
            return int(meta.get(k, default))
        except Exception:
            return default

    def _f(k, default=float("nan")):
        try:
            return float(meta.get(k, default))
        except Exception:
            return default

    def _b(k):
        return 1 if str(meta.get(k, "0")).lower() in {"1", "true", "yes"} else 0

    return {
        "vec":       meta.get("vec", ""),
        "vecslug":   meta.get("vecslug", ""),
        "method":    meta.get("method", ""),
        "K":         _i("K"),
        "N":         _i("N"),
        "D":         _i("D"),
        "normed":    _b("normed"),
        "iterations":_i("iterations"),
        "num_cores": _i("num_cores"),
        "max_iter":  _i("max_iter"),
        "max_words": meta.get("max_words", ""),
        "secs":      _f("secs"),
        "log_path":  str(log),
    }

def run(paths, cfg_path):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logs = list(Path(cfg["paths"]["clusters_dir"]).glob("**/K*/run.log"))
    rows = [_parse_log(p) for p in logs]
    df = pd.DataFrame(rows)

    out_dir = Path(cfg["paths"]["eval_dir"]) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.dropna(subset=["secs"])
    df = (df.sort_values(["vecslug","K","method","secs"])
            .groupby(["vecslug","K","method"], as_index=False).first())

    long_out = out_dir / "benchmarks.csv"
    df.to_csv(long_out, index=False)

    pivots = []
    for slug, g in df.groupby("vecslug"):
        pv = g.pivot_table(index="K", columns="method", values="secs", aggfunc="min")
        pv = pv.reset_index()
        if {"serial","parallel"} <= set(pv.columns):
            pv["speedup"] = pv["serial"] / pv["parallel"]
        pv.insert(0, "vecslug", slug)
        pivots.append(pv)
    if pivots:
        bench_pivot = pd.concat(pivots, ignore_index=True)
        bench_pivot.to_csv(out_dir / "benchmarks_pivot.csv", index=False)

    print(f"Wrote {long_out} ({len(df)} rows)")
