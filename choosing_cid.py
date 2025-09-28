import pandas as pd
from pathlib import Path

def load_stats(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]

    cid_col = None
    for cand in ["cluster", "cid", "cluster_id", "c"]:
        if cand in df.columns:
            cid_col = cand
            break
    if cid_col is None:
        raise SystemExit(f"No cluster id column found.\nColumns: {list(df.columns)}")

    if cid_col != "cluster":
        df["cluster"] = df[cid_col].astype(int)

    if "msi" not in df.columns:
        if "pos_purity" not in df.columns:
            raise SystemExit("No 'msi' and no 'pos_purity' column found to compute it.")
        suf_col = None
        for cand in ["max_suffix_purity", "suffix_purity_max", "suffix_purity"]:
            if cand in df.columns:
                suf_col = cand
                break
        if suf_col is None:
            for c in df.columns:
                if "suffix" in c and "purity" in c:
                    suf_col = c
                    break
        if suf_col is None:
            raise SystemExit("No 'msi' and couldn't find a suffix-purity column to compute it.")
        df["msi"] = df["pos_purity"] - df[suf_col]

    return df

raw_path   = Path("eval/metrics/cluster_stats_trained-mk_skipgram_parallel_K200.csv")
lemma_path = Path("eval/metrics/cluster_stats_trained-mk_lemma_skipgram_parallel_K200.csv")

df_raw   = load_stats(raw_path)
df_lemma = load_stats(lemma_path)

cid_raw_morph = int(df_raw.sort_values("msi", ascending=True).iloc[0]["cluster"])
msi_raw_morph = float(df_raw.sort_values("msi", ascending=True).iloc[0]["msi"])

cid_lem_sem   = int(df_lemma.sort_values("msi", ascending=False).iloc[0]["cluster"])
msi_lem_sem   = float(df_lemma.sort_values("msi", ascending=False).iloc[0]["msi"])

print("RAW / parallel / K=200 == most morphological")
print(f"  cid={cid_raw_morph}, MSI={msi_raw_morph:.3f}")
print("LEMMA / parallel / K=200 == most semantic")
print(f"  cid={cid_lem_sem}, MSI={msi_lem_sem:.3f}")

print("\nNext we can visualize the clusters with 'mkcli viz-cluster'")