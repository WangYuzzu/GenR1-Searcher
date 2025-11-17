## Data Construction Pipeline

1. Dataset Download
hotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
2Wiki: https://www.dropbox.com/s/npidmtadreo6df2/data.zip
or use `from datasets import load_dataset` to download

2. Stage 1, 2, 3
stage 1: hop < 3
stage 2: llm + retrieve -> Right && llm -> Wrong
stage 3: llm + retrieve Diff llm + generation