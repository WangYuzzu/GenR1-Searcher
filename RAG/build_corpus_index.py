#!/usr/bin/env python3
import os, glob, pickle
import numpy as np
import faiss

def main():
    split_dir = "/root/autodl-tmp/GenR1-Searcher/wiki_corpus_index_bulid/emb"
    pattern   = os.path.join(split_dir, "wiki_split_*.pkl")
    out_index = os.path.join(split_dir, "flat_enwiki.bin")  # 全精度 Flat 索引

    # 1) 读取 embeddings
    split_paths = sorted(glob.glob(pattern))
    arrays = []
    for path in split_paths:
        with open(path, "rb") as f:
            arr = pickle.load(f)
        if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
            arr = arr.cpu().numpy()
        # 转为 float32 ndarray
        arr = np.ascontiguousarray(np.asarray(arr, dtype="float32"))
        arrays.append(arr)

    # 2) 重建全量 embeddings
    num = len(arrays)
    total = sum(a.shape[0] for a in arrays)
    dim   = arrays[0].shape[1]
    corpus = np.zeros((total, dim), dtype="float32")
    for i, a in enumerate(arrays):
        idxs = i + np.arange(a.shape[0]) * num
        corpus[idxs] = a
    corpus = np.ascontiguousarray(corpus)

    # 3) index
    print(f"Building IndexFlatIP on {total} vectors, dim={dim} …")
    index = faiss.IndexFlatIP(dim)
    index.add(corpus)
    print("Add done, ntotal =", index.ntotal)

    # 4) 保存索引
    faiss.write_index(index, out_index)
    print("Index saved to", out_index)

if __name__ == "__main__":
    main()
