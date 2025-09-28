.PHONY: setup fetch clean lemma baseline train cluster eval probes bench all

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

fetch: ; mkcli fetch
clean: ; mkcli clean-text
lemma: ; mkcli lemma-text
baseline: ; mkcli download-baseline
train: ; mkcli train-embeddings && mkcli train-embeddings --use-lemma
cluster: ; mkcli cluster --vectors trained --method serial
eval: ; mkcli eval-intrinsic
probes: ; mkcli eval-probes
bench: ; mkcli bench-speed

all: fetch clean lemma baseline train cluster eval probes
