## Data Preparation

1. Download raw KILT data: http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json (about 37GB), And convert it to jsonl format
and save as kilt_knowledgesource.jsonl

2. Split Into 100-Word Chunks: 
```bash
python split_kilt_to_100.py
```
Then you get wiki_kilt_100_really.tsv

3. Download E5 retriever and parallel embedding(maybe take some time, depends on the gpu num):
```bash
bash split_and_embed.sh
```
Then you get n wiki_split_XX.pkl

4. Index:
```bash
python build_corpus_index.py
```
Then you get flat_enwiki.bin