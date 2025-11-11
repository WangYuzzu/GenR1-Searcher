#!/usr/bin/env python3
"""
split_and_embed.py

1. 把大的 wiki_kilt_100_really.tsv 按行均匀切成 n 份
2. 并行调用 build_corpus_embedding.py，对每份做向量化
3. 每份结果落到 output_dir/wiki_split_XX.pkl

用法示例：
    python split_and_embed.py \
        --tsv  /root/wangyu/R1-Searcher-main/wiki_kilt_100_really.tsv \
        --output_dir  emb \
        --num_splits  8 \
        --gpus        0,1,2,3,4,5,6,7 \
        --model_path  /opt/aps/workdir/model/bge-large-en-v1.5
"""

import argparse, os, math, subprocess, pathlib, sys, textwrap
from multiprocessing import Pool

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EMBED_SCRIPT = SCRIPT_DIR / "build_corpus_embedding.py"

def split_tsv(tsv_path: str, out_prefix: str, num_splits: int):
    """按行把 TSV 均匀切成 num_splits 份；返回切好的文件列表"""
    out_files = [open(f"{out_prefix}{i:02d}.tsv", "w", encoding="utf-8")
                 for i in range(1, num_splits + 1)]
    with open(tsv_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            out_files[idx % num_splits].write(line)
    for f in out_files:
        f.close()
    return [f"{out_prefix}{i:02d}.tsv" for i in range(1, num_splits + 1)]

def run_embed(args):
    """subprocess 调用 build_corpus_embedding.py"""
    file_path, pickle_path, gpu_id, model_path = args
    cmd = [
        sys.executable, str(EMBED_SCRIPT),
        "--file_path", file_path,
        "--save_path", pickle_path,
        "--gpu_id",   str(gpu_id),
    ]
    # 把模型路径通过环境变量写进去（脚本内部用 FlagModel(path)）
    env = os.environ.copy()
    env["BGE_MODEL_PATH"] = model_path
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Step-by-step:
          1. split large TSV → wiki_split_XX.tsv
          2. run build_corpus_embedding.py on each split (parallel)

        Note: build_corpus_embedding.py 会从环境变量 BGE_MODEL_PATH 读取模型目录，
        也可在脚本内写死。"""))
    ap.add_argument("--tsv", required=True, help="大 TSV 文件路径")
    ap.add_argument("--output_dir", required=True, help="向量 pickle 存放目录")
    ap.add_argument("--num_splits", type=int, default=8, help="切几份")
    ap.add_argument("--gpus", default="0", help="逗号分隔的 GPU 编号")
    ap.add_argument("--model_path", required=True, help="bge-large-en-v1.5 本地路径")
    args = ap.parse_args()

    tsv_path = pathlib.Path(args.tsv).expanduser().resolve()
    out_dir  = pathlib.Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Split
    split_prefix = out_dir / "wiki_split_"
    split_files = split_tsv(str(tsv_path), str(split_prefix), args.num_splits)
    print(f"✓  Split into {len(split_files)} files")

    # 2. Parallel embed
    gpu_list = [int(i) for i in args.gpus.split(",")]
    if len(gpu_list) < len(split_files):
        print("⚠ GPU 数小于分片数，将轮流复用 GPU。")

    jobs = []
    for i, split_file in enumerate(split_files):
        gpu_id = gpu_list[i % len(gpu_list)]
        pickle_path = out_dir / (pathlib.Path(split_file).stem + ".pkl")
        jobs.append((split_file, str(pickle_path), gpu_id, args.model_path))

    with Pool(len(gpu_list)) as pool:
        pool.map(run_embed, jobs)

    print("✓  All splits finished.")

if __name__ == "__main__":
    main()
