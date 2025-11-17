import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
from flask import Flask, request, jsonify
import faiss
import numpy as np
import torch
import sys
from transformers import AutoTokenizer, AutoModel


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for E5 embeddings."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch, seq_len, 1]
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)                                      # [batch, hidden]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [batch, 1]
    return summed / counts


def encode_e5_queries(texts):
    """Encode a list of query texts with E5 (global model/tokenizer/device)."""
    # E5 query 前缀
    batch_texts = ["query: " + t for t in texts]

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=300,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        embeddings = mean_pool(outputs.last_hidden_state, enc["attention_mask"])

    embeddings = embeddings.cpu().numpy().astype("float32")
    embeddings = np.ascontiguousarray(embeddings)
    return embeddings


# 创建 Flask 应用
app = Flask(__name__)


@app.route("/queries", methods=["POST"])
def query():
    # 从请求中获取查询向量
    data = request.json
    print(f"接受到data: {data}")
    queries = data["queries"]
    print(f"解析queries: {queries}")
    k = data.get("k", 3)

    # 1) 编码 query
    query_embeddings = encode_e5_queries(queries)

    all_answers = []
    print("正在检索...")
    D, I = index.search(query_embeddings, k=k)
    for idx in I:
        answers_for_query = [corpus[i] for i in idx[:k]]
        all_answers.append(answers_for_query)

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    data_type = sys.argv[1]
    port = sys.argv[2]

    # 加载 E5 模型
    model_name_or_path = "intfloat/e5-large-v2"
    print(f"[INFO] Loading E5 model from: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print(f"[INFO] 当前使用的 GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")
        print("[INFO] 当前运行在 CPU 上")

    print(f"[DEBUG] 模型 device: {next(model.parameters()).device}")
    print("模型已加载完毕")

    # 加载语料库
    assert data_type == "hotpotqa"
    file_path = "/root/autodl-tmp/GenR1-Searcher/RAG/wiki_kilt_100_really.tsv"
    corpus = load_corpus(file_path)
    print(f"语料库已加载完毕-{len(corpus)}")

    # 加载建好的索引（注意要和 E5 向量对应的那个 flat_enwiki.bin）
    index_path = "/root/autodl-tmp/GenR1-Searcher/RAG/emb/flat_enwiki.bin"
    index = faiss.read_index(index_path)

    print(f"[INFO] 索引已加载: {index_path}, ntotal={index.ntotal}")
    print("可以开始查询")

    app.run(host="0.0.0.0", port=port, debug=False)
