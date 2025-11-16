import os
import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from openai import OpenAI
import sys
import os
import re
from datasets import load_dataset
import http.client
import json
import copy
from tqdm import tqdm
import multiprocessing
from time import sleep
import requests
from collections import defaultdict
import random
import requests
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_file", type=str, default="None")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--search_port", type=str, default="5003")
    parser.add_argument("--gendoc_port", type=str, default="5004")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prompt_type", type=str, default="None")
    return parser.parse_args()


def process_text(examples, tokenizer, type=None):
#     sys_prompt = """You are a helpful assistant.
# Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
# The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
#
# During the thinking process, you have access to two external tools if necessary:
#
# **Tool 1: Search System**
# - Format: "<|begin_of_query|> search query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_query|>"
# - Function: Retrieves relevant information from external knowledge bases
# - Response format: "<|begin_of_documents|> ...search results... </|end_of_documents|>"
#
# **Tool 2: Document Generation System**
# - Format: "<|begin_of_generation|> generation query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_generation|>"
# - Function: A 7B parameter language model that generates documents tailored to your query
# - Response format: "<|begin_of_documents|> ...generated document... </|end_of_documents|>"
#
# **Note**: For multi-hop problems, it is recommended to decompose them into multiple simple sub-problems before using the tool. Each sub-problem should contain only a single query goal."""

    sys_prompt = """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".

During the thinking process, you have access to the external tools if necessary:

**Tool : Search System**
- Format: "<|begin_of_query|> search query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_query|>"
- Function: Retrieves relevant information from external knowledge bases
- Response format: "<|begin_of_documents|> ...search results... </|end_of_documents|>"

**Note**: For multi-hop problems, it is recommended to decompose them into multiple simple sub-problems before using the tool. Each sub-problem should contain only a single query goal."""
    question = examples["question"]
    messages_chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages_chat,
        tokenize=False,
        add_generation_prompt=True
    )
    examples["chat_prompt"] = chat_prompt + "<think>"
    # 保留idx字段
    if "idx" in examples:
        examples["idx"] = examples["idx"]
    return examples


def main():
    print("=Begin=" * 10)
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    temp = args.temp
    search_port = args.search_port
    gendoc_port = args.gendoc_port
    type = args.prompt_type
    model_path = args.model_path
    gpu_memory_rate = args.gpu_memory_rate

    # 读取数据
    data_ori_all = []
    with open(args.src_file, "r") as f:
        data_ori_all = []
        for i, line in enumerate(f):
            if args.start_sample <= i < args.end_sample:
                obj_ori = json.loads(line)
                data_ori_all.append(obj_ori)
            if i >= args.end_sample - 1:
                break

    print("All Data Length: ", len(data_ori_all))
    chunk_size = 100
    chunk_num = len(data_ori_all) // chunk_size
    if len(data_ori_all) % chunk_size != 0:
        chunk_num += 1

    # 初始化模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_rate, trust_remote_code=True)

    for h in range(chunk_num):
        print("==" * 80)
        print("Begin Chunk: ", h, "All: ", chunk_num)
        data_ori = data_ori_all[h * chunk_size:(h + 1) * chunk_size]
        data = []

        for i in range(len(data_ori)):
            for j in range(1):
                data.append(data_ori[i])

        data_keys = ["idx", "question", "answer", "gen_text_store"]
        ds = Dataset.from_dict({key: [d[key] for d in data] for key in data_keys})
        print(len(ds))
        ds = ds.map(
            process_text,
            num_proc=16,
            fn_kwargs={"tokenizer": tokenizer, "type": type},
        )
        print(ds)

        # 更新停止标记，包含两个工具的结束标记
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</|end_of_query|>", "</|end_of_generation|>", "</answer>"]
        sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, stop=stop_tokens)

        finished_all_list = []
        continued_answer = copy.deepcopy(data)

        for k in range(16):
            if len(ds) == 0:
                print("请确定是不是真的ok了")
                print(len(ds))
                break

            outputs = llm.generate(ds['chat_prompt'], sampling_params)

            finished_texts = []
            continued_texts = []

            # 分别收集两种查询
            search_query_list = []
            gendoc_query_list = []
            search_indices = []
            gendoc_indices = []

            for i, output in enumerate(outputs):
                prompt = output.prompt
                idx = continued_answer[i]["idx"]
                answer = continued_answer[i]["answer"]
                question = continued_answer[i]["question"]
                gen_text_store = continued_answer[i].get("gen_text_store", "")
                stop_reason = output.outputs[0].stop_reason
                generated_text = output.outputs[0].text

                # 统计工具使用次数
                search_count = continued_answer[i].get("search_count", 0)
                gendoc_count = continued_answer[i].get("gendoc_count", 0)

                if k == 9:  # 调用次数太多了，直接停掉
                    original_data = {
                        "idx": idx,
                        "question": question,
                        "answer": answer,
                        "generated_text": generated_text,
                        "stop_reason_final": "many_tool_calls",
                        "pred_ans": "I don't know.",
                        "search_count": search_count,
                        "gendoc_count": gendoc_count
                    }
                    finished_texts.append(original_data)
                    continue

                # 处理完成的回答
                if "<answer>" in generated_text and stop_reason == "</answer>":
                    original_data = {
                        "idx": idx,
                        "question": question,
                        "answer": answer,
                        "pred_ans": generated_text.split("<answer>")[-1].split("</answer>")[0],
                        "stop_reason_final": "finished",
                        "gen_text_store": gen_text_store + generated_text + "</answer>",
                        "search_count": search_count,
                        "gendoc_count": gendoc_count
                    }
                    finished_texts.append(original_data)

                # 处理搜索请求
                elif "<|begin_of_query|>" in generated_text and stop_reason == "</|end_of_query|>":
                    query = generated_text.split("<|begin_of_query|>")[-1].split("</|end_of_query|>")[0]
                    query = query.replace('"', "").replace("'", "").replace("\t", " ").replace("...", "").strip()
                    if query:
                        search_query_list.append(query)
                        search_indices.append(len(continued_texts))
                        search_count += 1

                        original_data = {
                            "idx": idx,
                            "chat_prompt": prompt + generated_text.strip(),
                            "answer": answer,
                            "question": question,
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "search_count": search_count,
                            "gendoc_count": gendoc_count,
                            "tool_type": "search"
                        }
                        continued_texts.append(original_data)
                    else:
                        original_data = {
                            "idx": idx,
                            "question": question,
                            "answer": answer,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text": generated_text,
                            "stop_reason_final": "query_inst_error",
                            "pred_ans": "I don't know.",
                            "search_count": search_count,
                            "gendoc_count": gendoc_count
                        }
                        finished_texts.append(original_data)

                # 处理文档生成请求
                elif "<|begin_of_generation|>" in generated_text and stop_reason == "</|end_of_generation|>":
                    gen_query = generated_text.split("<|begin_of_generation|>")[-1].split("</|end_of_generation|>")[0]
                    gen_query = gen_query.replace('"', "").replace("'", "").strip()
                    if gen_query:
                        gendoc_query_list.append(gen_query)
                        gendoc_indices.append(len(continued_texts))
                        gendoc_count += 1

                        original_data = {
                            "idx": idx,
                            "chat_prompt": prompt + generated_text.strip(),
                            "answer": answer,
                            "question": question,
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "search_count": search_count,
                            "gendoc_count": gendoc_count,
                            "tool_type": "gendoc"
                        }
                        continued_texts.append(original_data)
                    else:
                        original_data = {
                            "idx": idx,
                            "question": question,
                            "answer": answer,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text": generated_text,
                            "stop_reason_final": "generation_inst_error",
                            "pred_ans": "I don't know.",
                            "search_count": search_count,
                            "gendoc_count": gendoc_count
                        }
                        finished_texts.append(original_data)

                else:
                    original_data = {
                        "idx": idx,
                        "question": question,
                        "answer": answer,
                        "stop_reason_final": "shot_down",
                        "pred_ans": "I don't know.",
                        "search_count": search_count,
                        "gendoc_count": gendoc_count
                    }
                    finished_texts.append(original_data)

            print("Search queries:", search_query_list)
            print("Generation queries:", gendoc_query_list)
            print("==" * 80)

            # 处理搜索请求
            if len(search_query_list) > 0:
                url_wiki = f"http://0.0.0.0:{search_port}/queries"
                topk = 5
                try:
                    response = requests.post(url_wiki, json={"queries": search_query_list, "k": topk})
                    if response.status_code == 200:
                        result = response.json()
                        search_answers = result["answers"]

                        # 将搜索结果插入到对应的continued_texts中
                        search_idx = 0
                        for j, cont_text in enumerate(continued_texts):
                            if cont_text.get("tool_type") == "search" and search_idx < len(search_answers):
                                retrieve_docs = search_answers[search_idx]
                                if len(retrieve_docs) > 0:
                                    doc_content_list = []
                                    for m in range(len(retrieve_docs)):
                                        doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[m])
                                        doc_content_list.append(f"({m + 1}){doc_now}\n")
                                    doc_content = ''.join(doc_content_list)
                                else:
                                    doc_content = "None"

                                cont_text[
                                    "chat_prompt"] += "</|end_of_query|>\n\n<|begin_of_documents|>\n" + doc_content + "</|end_of_documents|>\n\n"
                                cont_text[
                                    "gen_text_store"] += "</|end_of_query|>\n\n<|begin_of_documents|>\n" + doc_content + "</|end_of_documents|>\n\n"
                                search_idx += 1
                except Exception as e:
                    print(f"搜索服务错误: {e}")

            # 处理文档生成请求
            if len(gendoc_query_list) > 0:
                url_gendoc = f"http://0.0.0.0:{gendoc_port}/generate_docs"
                topk = 1
                try:
                    response = requests.post(url_gendoc, json={"queries": gendoc_query_list, "k": topk})
                    if response.status_code == 200:
                        result = response.json()
                        generated_docs = result["documents"]

                        # 将生成的文档插入到对应的continued_texts中
                        gendoc_idx = 0
                        for j, cont_text in enumerate(continued_texts):
                            if cont_text.get("tool_type") == "gendoc" and gendoc_idx < len(generated_docs):
                                generated_doc = generated_docs[gendoc_idx]
                                cont_text[
                                    "chat_prompt"] += "</|end_of_generation|>\n\n<|begin_of_documents|>\n" + generated_doc + "</|end_of_documents|>\n\n"
                                cont_text[
                                    "gen_text_store"] += "</|end_of_generation|>\n\n<|begin_of_documents|>\n" + generated_doc + "</|end_of_documents|>\n\n"
                                gendoc_idx += 1
                except Exception as e:
                    print(f"文档生成服务错误: {e}")

            # 清理tool_type标记
            for cont_text in continued_texts:
                if "tool_type" in cont_text:
                    del cont_text["tool_type"]

            finished_all_list.extend(finished_texts)

            if len(continued_texts) == 0:
                if len(finished_texts) > 0:
                    output_file = args.src_file.replace(".jsonl",
                                                        f"-{model_path.split('/')[-2]}{model_path.split('/')[-1]}_two_tools_temp{args.temp}_type{type}.jsonl")
                    with open(output_file, "a") as f:
                        for text in finished_texts:
                            f.write(json.dumps(text) + "\n")
                break
            else:
                data_keys_again = continued_texts[0].keys()
                ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                continued_answer = copy.deepcopy(continued_texts)

            print("==" * 80)
            print(
                f"Epoch: {k}, New_Finished: {len(finished_texts)}, All_Finished: {len(finished_all_list)}, Continued: {len(continued_texts)}")
            print(f"Begin Writing Epoch: {k}")
            print("==" * 80)

            if len(finished_texts) > 0:
                output_file = args.src_file.replace(".jsonl",
                                                    f"-{model_path.split('/')[-2]}{model_path.split('/')[-1]}_two_tools_temp{args.temp}_type{type}.jsonl")
                with open(output_file, "a") as f:
                    for text in finished_texts:
                        f.write(json.dumps(text) + "\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()