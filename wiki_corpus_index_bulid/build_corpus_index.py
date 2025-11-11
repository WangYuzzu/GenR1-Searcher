#!/usr/bin/env python3
import os, glob, pickle
import numpy as np
import faiss

def main():
    split_dir = "/root/autodl-tmp/R1-Searcher/wiki_corpus_index_bulid/emb"
    pattern   = os.path.join(split_dir, "wiki_split_*.pkl")
    out_index = os.path.join(split_dir, "sq4_enwiki.bin")

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

    # 3) 构建 Scalar Quantizer 索引，4 bit 量化
    print(f"Training ScalarQuantizer Q4 on {total} vectors, dim={dim} …")
    sq = faiss.IndexScalarQuantizer(
        dim,
        faiss.ScalarQuantizer.QT_4bit,           # 4 bit 量化，每维4比特
        faiss.METRIC_INNER_PRODUCT
    )
    sq.train(corpus)                            # 先训练量化器
    print("Train done. Adding vectors …")
    sq.add(corpus)                              # 再添加所有向量
    print("Add done, ntotal =", sq.ntotal)

    # 4) 保存量化索引
    faiss.write_index(sq, out_index)
    print("Index saved to", out_index)

if __name__ == "__main__":
    main()
