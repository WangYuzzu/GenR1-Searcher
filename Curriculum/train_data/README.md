## Data Construction Pipeline

**1.Download Dataset && Model**

hotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

2Wiki: https://www.dropbox.com/s/npidmtadreo6df2/data.zip

or use `from datasets import load_dataset` to download

Model is Qwen/Qwen2.5-7B-Instruct

**2. Stage 1, 2, 3**

stage 1: hop < 3(The number of hops in hotpotQA is 2. 2Wiki determines the number of hops by the length of the evidences.)

stage 2: llm + retrieve -> Right && llm -> Wrong

stage 3: llm + retrieve Diff llm + generation