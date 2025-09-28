import pandas as pd
from gensim.models import KeyedVectors
import numpy as np, sys

K = 200; cid = 103  # modify as needed
print(f"K={K}, cid={cid}")
asg = pd.read_csv(f"clusters/trained-mk_skipgram/parallel/K{K}/assignments.csv")
vec = "embeddings/trained/mk_skipgram.vec"  # or the path from run.log
kv = KeyedVectors.load_word2vec_format(vec, binary=False)
cl_words = [w for w in asg[asg.cluster==cid].word if w in kv]
cent = np.mean(kv[cl_words], axis=0)
cent /= np.linalg.norm(cent) + 1e-12
sims = [(w, float(np.dot(kv[w]/(np.linalg.norm(kv[w])+1e-12), cent))) for w in cl_words]
for w, s in sorted(sims, key=lambda x: -x[1])[:50]:
    print(w, round(s,3))
