import pandas as pd
df = pd.read_csv("eval/metrics/alignment_trained-mk_skipgram_serial__trained-mk_skipgram_parallel_K200.csv")
CA = [c for c in df.columns if c.lower().startswith("cid") and c.lower().endswith("_a")][0]
CB = [c for c in df.columns if c.lower().startswith("cid") and c.lower().endswith("_b")][0]
COS = [c for c in df.columns if "cos" in c.lower()][0]

cid = 147
match = df.loc[df[CA] == cid].iloc[0]
print("serial cid", int(cid)," ↔ parallel cid", int(match[CB]), " (centroid cosine =", float(match[COS]), ")")
