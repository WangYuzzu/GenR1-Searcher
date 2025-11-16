python split_and_embed.py \
  --tsv /root/autodl-tmp/GenR1-Searcher/wiki_kilt_100_really.tsv \
  --output_dir /root/autodl-tmp/emb_fix0204 \
  --num_splits 7  \
  --gpus 0,1,2,3,4,5,6 \
  --model_path /YOUR/E5/PATH \

# --num_splits: n <= GPU available number