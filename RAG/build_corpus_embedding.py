import os
import argparse
import pickle
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def load_corpus(file_path: str) -> List[str]:
    """Load corpus from a TSV file and return a list of texts.

    Each line is expected to have at least two columns separated by a tab.
    This function keeps the second column (title + paragraph).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
        c_len = len(corpus)
        new_corpus: List[str] = []
        line_num = 0
        for line in corpus:
            line_num += 1
            if line_num % 10000 == 0:
                print(f"Percent: {line_num}/{c_len}")

            title_text = line.split('\t')[1].strip('')
            new_corpus.append(title_text)
    return new_corpus


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for sentence embeddings (standard for E5 models)."""
    # last_hidden_state: [batch, seq_len, hidden]
    # attention_mask:    [batch, seq_len]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch, seq_len, 1]
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)                                      # [batch, hidden]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [batch, 1]
    return summed / counts


def encode_corpus_with_e5(
    texts: List[str],
    model_name_or_path: str,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 300,
) -> np.ndarray:
    """Encode a list of texts using an E5 model.

    E5 expects inputs prefixed with "passage:" for corpus / documents.
    """
    print(f"Loading E5 model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            # E5 corpus prefix
            batch_texts = ["passage: " + t for t in batch_texts]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            embeddings = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

            if (start // batch_size) % 10 == 0:
                print(f"Encoded {end}/{len(texts)} paragraphs")

    corpus_embeddings = np.vstack(all_embeddings)
    return corpus_embeddings.astype("float32")


def process_corpus(file_path: str, save_path: str, gpu_id: int):
    # Load corpus
    print("Start load corpus")
    corpus = load_corpus(file_path)
    print(f"Load {len(corpus)} from {file_path}.")

    # Print a few samples
    for sample in corpus[:2]:
        print(sample)

    # Decide device; CUDA_VISIBLE_DEVICES is set by caller, so we can just use 'cuda'
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Choose model path: prefer env var set by split_and_embed.py, fallback to default E5
    model_name_or_path = "intfloat/e5-large-v2"

    print("Start encode with E5")
    corpus_embeddings = encode_corpus_with_e5(
        corpus,
        model_name_or_path=model_name_or_path,
        device=device,
        batch_size=256,
        max_length=300,
    )

    print("Shape of the corpus embeddings:", corpus_embeddings.shape)
    print("Data type of the embeddings:", corpus_embeddings.dtype)

    print("Start save")
    with open(save_path, "ab") as f:
        pickle.dump(corpus_embeddings, f)
    print("Save over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process corpus and save embeddings (E5 version).')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output pickle file.')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID to use (default: -1 for CPU).')

    args = parser.parse_args()
    # 保持和原来的行为一致：虽然 split_and_embed.py 会设置，这里再同步一份
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    process_corpus(args.file_path, args.save_path, args.gpu_id)
