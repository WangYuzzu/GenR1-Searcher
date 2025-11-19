import os
import json
import torch
import requests
import re
import copy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import argparse
import time
import multiprocessing as mp
from typing import List, Dict, Any
from functools import partial


class TestDataset:
    """测试数据集类"""

    def __init__(self, jsonl_file):
        self.data = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_rl_model(checkpoint_path, base_model_path=None, device='cuda', gpu_memory_rate=0.9, tensor_parallel_size=1):
    """
    加载RL训练的模型

    Args:
        checkpoint_path: checkpoint路径
        base_model_path: 基础模型路径
        device: 设备
        gpu_memory_rate: GPU内存使用率
        tensor_parallel_size: 张量并行大小

    Returns:
        model, tokenizer
    """
    print(f"正在加载模型从: {checkpoint_path}")
    print(f"GPU配置: {tensor_parallel_size}张卡并行，内存利用率{gpu_memory_rate}")

    # 检查checkpoint目录内容
    if os.path.isdir(checkpoint_path):
        files = os.listdir(checkpoint_path)
        print(f"Checkpoint目录包含: {files}")

        # 检查是否有.pt或.bin文件
        weight_files = [f for f in files if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.safetensors')]
        if weight_files:
            print(f"找到权重文件: {weight_files}")

    # 检查是否是RL训练的checkpoint子目录（如global_step300）
    if not os.path.exists(os.path.join(checkpoint_path, "config.json")):
        print(f"{checkpoint_path} 没有config.json，看起来是RL训练的权重目录")

        # 必须有基础模型路径
        if not base_model_path:
            print(f"检测到RL权重文件，但没有指定基础模型路径")
            print("请使用 --base_model_path 参数指定基础模型路径，例如：")
            print("  --base_model_path /root/autodl-tmp/Qwen-2.5-3B-Instruct")
            print("  --base_model_path Qwen/Qwen2.5-7B-Instruct")
            raise ValueError("RL权重需要指定基础模型路径")

        if not os.path.exists(base_model_path):
            print(f"基础模型路径不存在: {base_model_path}")
            raise ValueError(f"基础模型路径不存在: {base_model_path}")

        print(f"使用基础模型: {base_model_path}")

        # 加载tokenizer从基础模型
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 创建临时目录来存放合并后的模型
        import tempfile
        import shutil

        temp_model_dir = tempfile.mkdtemp(prefix="merged_model_")
        print(f"创建临时模型目录: {temp_model_dir}")

        try:
            # 复制基础模型文件到临时目录
            print("复制基础模型文件...")
            for item in os.listdir(base_model_path):
                src = os.path.join(base_model_path, item)
                dst = os.path.join(temp_model_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src) and item not in ['.git', '__pycache__']:
                    shutil.copytree(src, dst)

            # 复制RL权重文件到临时目录，覆盖基础模型权重
            print("复制RL权重文件...")
            for item in os.listdir(checkpoint_path):
                if item.endswith('.pt') or item.endswith('.bin') or item.endswith('.safetensors'):
                    src = os.path.join(checkpoint_path, item)
                    # 重命名actor权重文件为标准名称
                    if 'actor' in item.lower():
                        # 尝试找到基础模型的权重文件名
                        base_weight_files = [f for f in os.listdir(temp_model_dir)
                                             if f.endswith('.bin') or f.endswith('.safetensors')]
                        if base_weight_files:
                            dst_name = base_weight_files[0]  # 使用相同的文件名
                        else:
                            dst_name = 'pytorch_model.bin'  # 默认名称
                        dst = os.path.join(temp_model_dir, dst_name)
                    else:
                        dst = os.path.join(temp_model_dir, item)

                    print(f"  复制 {item} -> {os.path.basename(dst)}")
                    shutil.copy2(src, dst)

            # 使用合并后的模型目录加载
            print(f"加载合并后的模型...")
            llm = LLM(
                model=temp_model_dir,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_rate,
                trust_remote_code=True
            )

            # 保存临时目录路径，以便后续清理
            llm._temp_model_dir = temp_model_dir

        except Exception as e:
            # 如果失败，清理临时目录并回退到基础模型
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            print(f"合并模型失败: {e}")
            print(f"回退到基础模型: {base_model_path}")
            print("警告：将使用基础模型而非RL训练的权重")

            llm = LLM(
                model=base_model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_rate,
                trust_remote_code=True
            )

    else:
        # 标准加载流程（checkpoint包含完整模型）
        print("检测到完整模型目录，使用标准加载流程")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=checkpoint_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_rate,
            trust_remote_code=True
        )

    print("模型加载完成!")
    return llm, tokenizer


def create_chat_prompt(question: str, tokenizer) -> str:
    """
    创建聊天prompt

    Args:
        question: 问题
        tokenizer: tokenizer

    Returns:
        格式化的prompt
    """
    messages_chat = [
        {
            "role": "system",
            "content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".

During the thinking process, you have access to two external tools if necessary:

**Tool 1: Search System**
- Format: "<|begin_of_query|> search query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_query|>"
- Function: Retrieves relevant information from external knowledge bases
- Response format: "<|begin_of_documents|> ...search results... </|end_of_documents|>"

**Tool 2: Document Generation System** 
- Format: "<|begin_of_generation|> generation query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_generation|>"
- Function: A 7B parameter language model that generates documents tailored to your query
- Response format: "<|begin_of_documents|> ...generated document... </|end_of_documents|>"

**Note**: Each query must involve only a single concept or topic."""
        },
        {"role": "user", "content": question}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages_chat,
        tokenize=False,
        add_generation_prompt=True
    )

    return chat_prompt + "<think>"


def call_retrieval_service(queries: List[str], url: str = "http://0.0.0.0:5003/queries", k: int = 3,
                           timeout: int = 300) -> List[List[str]]:
    """
    调用检索服务

    Args:
        queries: 查询列表
        url: 检索服务URL
        k: 返回top-k结果
        timeout: 超时时间

    Returns:
        检索结果列表
    """
    try:
        response = requests.post(url, json={"queries": queries, "k": k}, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            return result["answers"]
        else:
            print(f"检索服务响应异常: {response.status_code}")
            return [[] for _ in queries]
    except Exception as e:
        print(f"检索请求失败: {e}")
        return [[] for _ in queries]


def call_generation_service(queries: List[str], url: str = "http://101.42.41.82:5004/generate_docs", k: int = 1,
                            timeout: int = 300) -> List[str]:
    """
    调用文档生成服务

    Args:
        queries: 查询列表
        url: 生成服务URL
        k: 返回top-k结果
        timeout: 超时时间

    Returns:
        生成文档列表
    """
    try:
        response = requests.post(url, json={"queries": queries, "k": k}, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            return result["documents"]
        else:
            print(f"文档生成服务响应异常: {response.status_code}")
            return ["No document generated." for _ in queries]
    except Exception as e:
        print(f"文档生成请求失败: {e}")
        return ["No document generated." for _ in queries]


def format_retrieval_results(docs: List[str]) -> str:
    """
    格式化检索结果

    Args:
        docs: 文档列表

    Returns:
        格式化的文档字符串
    """
    if not docs:
        return "None"

    doc_content_list = []
    for j, doc in enumerate(docs):
        doc_now = re.sub(r'^\d+\s+', '', doc)
        doc_content_list.append(f"({j + 1}){doc_now}\n")

    return ''.join(doc_content_list)


def generate_answer_with_tools(llm, question: str, tokenizer,
                               retrieval_url: str = "http://0.0.0.0:5003/queries",
                               generation_url: str = "http://101.42.41.82:5004/generate_docs",
                               max_rounds: int = 10,
                               temperature: float = 0.0) -> Dict[str, Any]:
    """
    使用工具生成答案

    Args:
        llm: vLLM模型
        question: 问题
        tokenizer: tokenizer
        retrieval_url: 检索服务URL
        generation_url: 生成服务URL
        max_rounds: 最大轮数
        temperature: 生成温度

    Returns:
        包含答案和统计信息的字典
    """
    # 创建初始prompt
    current_prompt = create_chat_prompt(question, tokenizer)

    # 统计信息
    retrieve_count = 0
    generate_count = 0
    round_count = 0
    full_generation = ""

    # 停止标记 - 只在工具调用时停止，让模型自然生成完整内容
    stop_tokens = ["</|end_of_query|>", "</|end_of_generation|>"]

    for round_num in range(max_rounds):
        round_count += 1

        # 设置采样参数 - 评估时使用确定性生成
        sampling_params = SamplingParams(
            temperature=temperature,  # 评估时应该是0.0
            top_p=1.0,  # 评估时不使用top_p采样
            max_tokens=512,
            stop=stop_tokens
        )

        # 生成
        outputs = llm.generate([current_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        stop_reason = outputs[0].outputs[0].stop_reason

        full_generation += generated_text

        print(f"第{round_num + 1}轮生成: {generated_text[:100]}...")
        print(f"停止原因: {stop_reason}")

        # 检查是否需要检索
        if "<|begin_of_query|>" in generated_text and stop_reason == "</|end_of_query|>":
            query = generated_text.split("<|begin_of_query|>")[-1].split("</|end_of_query|>")[0]
            query = query.replace('"', "").strip()
            query = " ".join(query.split())

            if query:
                print(f"检索查询: {query}")
                retrieve_count += 1

                # 调用检索服务
                retrieval_results = call_retrieval_service([query], retrieval_url, k=3)
                doc_content = format_retrieval_results(retrieval_results[0] if retrieval_results else [])

                # 更新prompt
                current_prompt = (current_prompt + generated_text + "</|end_of_query|>\n\n" +
                                  "<|begin_of_documents|>\n" + doc_content + "</|end_of_documents|>\n\n")
                continue
            else:
                print("检索查询为空")
                break

        # 检查是否需要生成文档
        elif "<|begin_of_generation|>" in generated_text and stop_reason == "</|end_of_generation|>":
            gen_query = generated_text.split("<|begin_of_generation|>")[-1].split("</|end_of_generation|>")[0]
            gen_query = gen_query.strip()

            if gen_query:
                print(f"生成查询: {gen_query}")
                generate_count += 1

                # 调用生成服务
                generation_results = call_generation_service([gen_query], generation_url, k=1)
                generated_doc = generation_results[0] if generation_results else "No document generated."

                # 更新prompt
                current_prompt = (current_prompt + generated_text + "</|end_of_generation|>\n\n" +
                                  "<|begin_of_documents|>\n" + generated_doc + "</|end_of_documents|>\n\n")
                continue
            else:
                print("生成查询为空")
                break

        # 其他情况：生成结束或包含完整答案
        else:
            print(f"生成结束，原因: {stop_reason}")
            current_prompt = current_prompt + generated_text
            full_generation += generated_text

            # 检查是否包含完整的答案
            if "<answer>" in full_generation and "</answer>" in full_generation:
                # 提取完整答案
                answer_part = full_generation.split("<answer>")[-1]
                final_answer = answer_part.split("</answer>")[0].strip()

                return {
                    "question": question,
                    "final_answer": final_answer,
                    "full_generation": full_generation,
                    "retrieve_count": retrieve_count,
                    "generate_count": generate_count,
                    "round_count": round_count,
                    "status": "completed"
                }

            # 如果只有</think>但没有答案，继续生成答案部分
            elif "</think>" in full_generation and "<answer>" not in full_generation:
                print("思考完成，继续生成答案...")
                answer_prompt = current_prompt + "\n\n<answer>"

                # 生成最终答案 - 确定性采样
                final_sampling_params = SamplingParams(
                    temperature=temperature,  # 保持一致
                    top_p=1.0,
                    max_tokens=256,
                    stop=["</answer>"]
                )

                final_outputs = llm.generate([answer_prompt], final_sampling_params)
                final_answer = final_outputs[0].outputs[0].text.strip()
                full_generation += "\n\n<answer>" + final_answer + "</answer>"

                return {
                    "question": question,
                    "final_answer": final_answer,
                    "full_generation": full_generation,
                    "retrieve_count": retrieve_count,
                    "generate_count": generate_count,
                    "round_count": round_count,
                    "status": "completed"
                }

            # 其他情况：不完整的生成
            else:
                final_answer = "Generation incomplete or malformed."
                return {
                    "question": question,
                    "final_answer": final_answer,
                    "full_generation": full_generation,
                    "retrieve_count": retrieve_count,
                    "generate_count": generate_count,
                    "round_count": round_count,
                    "status": "incomplete"
                }

    # 超过最大轮数
    print("超过最大轮数限制")
    return {
        "question": question,
        "final_answer": "Max rounds exceeded.",
        "full_generation": full_generation,
        "retrieve_count": retrieve_count,
        "generate_count": generate_count,
        "round_count": round_count,
        "status": "max_rounds_exceeded"
    }


def normalize_answer(text: str) -> str:
    """
    标准化答案文本，用于计算EM
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def calculate_cover_em(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    计算Cover EM得分

    Args:
        predictions: 预测答案列表
        references: 参考答案列表

    Returns:
        评估结果字典
    """
    assert len(predictions) == len(references), "预测和参考答案数量不一致"

    cover_matches = 0
    exact_matches = 0
    detailed_results = []

    for pred, ref in zip(predictions, references):
        norm_pred = normalize_answer(pred)
        norm_ref = normalize_answer(ref)

        # Cover EM: 检查覆盖关系
        is_cover_match = (norm_pred in norm_ref) or (norm_ref in norm_pred)

        # Exact EM: 完全匹配
        is_exact_match = norm_pred == norm_ref

        if is_cover_match:
            cover_matches += 1
        if is_exact_match:
            exact_matches += 1

        detailed_results.append({
            'prediction': pred,
            'reference': ref,
            'normalized_prediction': norm_pred,
            'normalized_reference': norm_ref,
            'cover_match': is_cover_match,
            'exact_match': is_exact_match
        })

    total = len(predictions)
    cover_em_score = cover_matches / total
    exact_em_score = exact_matches / total

    return {
        'cover_em_score': cover_em_score,
        'exact_em_score': exact_em_score,
        'total_questions': total,
        'cover_matches': cover_matches,
        'exact_matches': exact_matches,
        'detailed_results': detailed_results
    }


def worker_evaluate(worker_id: int, data_chunk: List[Dict], args, results_queue):
    """
    单个worker进程的评估函数

    Args:
        worker_id: worker编号
        data_chunk: 分配给这个worker的数据块
        args: 参数
        results_queue: 结果队列
    """
    try:
        # 设置当前进程使用的GPU
        gpu_id = worker_id % torch.cuda.device_count()  # 循环分配GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Worker {worker_id} 启动，使用GPU {gpu_id}，处理 {len(data_chunk)} 个样本")

        # 加载模型（每个进程独立加载）
        llm, tokenizer = load_rl_model(
            args.checkpoint_path,
            base_model_path=args.base_model_path,
            gpu_memory_rate=args.gpu_memory_rate,
            tensor_parallel_size=1  # 每个进程使用1张卡
        )

        worker_results = []

        # 处理分配给这个worker的数据
        for i, item in enumerate(data_chunk):
            question = item['question']
            reference = item['answer']

            if i % 10 == 0:  # 每10个样本打印一次进度
                print(f"Worker {worker_id}: 处理进度 {i + 1}/{len(data_chunk)}")

            # 生成答案
            result = generate_answer_with_tools(
                llm, question, tokenizer,
                retrieval_url=args.retrieval_url,
                generation_url=args.generation_url,
                temperature=args.temperature
            )

            worker_results.append({
                'worker_id': worker_id,
                'question_id': f"{worker_id}_{i}",
                'question': question,
                'reference': reference,
                'prediction': result['final_answer'],
                'full_generation': result['full_generation'],
                'retrieve_count': result['retrieve_count'],
                'generate_count': result['generate_count'],
                'round_count': result['round_count'],
                'status': result['status']
            })

        # 将结果放入队列
        results_queue.put(worker_results)
        print(f"Worker {worker_id} 完成，处理了 {len(data_chunk)} 个样本")

    except Exception as e:
        print(f"Worker {worker_id} 出错: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put([])  # 放入空结果避免主进程等待


def evaluate_model_parallel(test_file: str, args, num_workers: int = 4) -> Dict[str, Any]:
    """
    并行评估模型

    Args:
        test_file: 测试文件路径
        args: 参数对象
        num_workers: 并行worker数量

    Returns:
        评估结果
    """
    print(f"开始并行评估，使用 {num_workers} 个worker进程")
    print(f"测试文件: {test_file}")

    # 加载测试数据
    dataset = TestDataset(test_file)
    if args.max_samples > 0:
        dataset.data = dataset.data[:args.max_samples]

    total_samples = len(dataset)
    print(f"总共 {total_samples} 个测试样本")

    # 将数据分块给不同的worker
    chunk_size = (total_samples + num_workers - 1) // num_workers
    data_chunks = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        if start_idx < total_samples:
            chunk = dataset.data[start_idx:end_idx]
            data_chunks.append(chunk)
            print(f"Worker {i}: 样本 {start_idx}-{end_idx - 1} ({len(chunk)} 个)")

    # 创建结果队列
    results_queue = mp.Queue()

    # 启动worker进程
    processes = []
    for i, chunk in enumerate(data_chunks):
        p = mp.Process(
            target=worker_evaluate,
            args=(i, chunk, args, results_queue)
        )
        p.start()
        processes.append(p)

    # 收集所有结果
    all_worker_results = []
    for i in range(len(data_chunks)):
        worker_results = results_queue.get()
        all_worker_results.extend(worker_results)
        print(f"收到worker结果，当前总数: {len(all_worker_results)}")

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"所有worker完成！总共处理 {len(all_worker_results)} 个样本")

    # 整理结果
    predictions = []
    references = []
    detailed_results = []

    # 统计信息
    total_retrieve_count = 0
    total_generate_count = 0
    status_counts = defaultdict(int)

    for result in all_worker_results:
        predictions.append(result['prediction'])
        references.append(result['reference'])
        detailed_results.append(result)

        total_retrieve_count += result['retrieve_count']
        total_generate_count += result['generate_count']
        status_counts[result['status']] += 1

    # 计算Cover EM等指标
    print("\n正在计算评估指标...")
    em_results = calculate_cover_em(predictions, references)

    # 汇总结果
    evaluation_results = {
        'metrics': em_results,
        'statistics': {
            'total_samples': len(all_worker_results),
            'avg_retrieve_count': total_retrieve_count / len(all_worker_results) if all_worker_results else 0,
            'avg_generate_count': total_generate_count / len(all_worker_results) if all_worker_results else 0,
            'total_retrieve_count': total_retrieve_count,
            'total_generate_count': total_generate_count,
            'status_distribution': dict(status_counts),
            'num_workers_used': len(data_chunks)
        },
        'detailed_results': detailed_results
    }

    return evaluation_results


def evaluate_model_single(llm, tokenizer, test_file: str,
                          retrieval_url: str = "http://0.0.0.0:5003/queries",
                          generation_url: str = "http://101.42.41.82:5004/generate_docs",
                          max_samples: int = -1,
                          temperature: float = 0.0) -> Dict[str, Any]:
    """
    单进程评估模型

    Args:
        llm: vLLM模型
        tokenizer: tokenizer
        test_file: 测试文件路径
        retrieval_url: 检索服务URL
        generation_url: 生成服务URL
        max_samples: 最大样本数，-1表示全部
        temperature: 生成温度

    Returns:
        评估结果
    """
    print(f"开始单进程评估，测试文件: {test_file}")

    # 加载测试数据
    dataset = TestDataset(test_file)
    if max_samples > 0:
        dataset.data = dataset.data[:max_samples]

    print(f"总共 {len(dataset)} 个测试样本")

    predictions = []
    references = []
    detailed_results = []

    # 统计信息
    total_retrieve_count = 0
    total_generate_count = 0
    status_counts = defaultdict(int)

    print("正在生成答案...")
    for i, item in enumerate(tqdm(dataset.data)):
        question = item['question']
        reference = item['answer']

        if i < 3:  # 只为前3个显示详细信息
            print(f"\n{'=' * 60}")
            print(f"处理第 {i + 1}/{len(dataset)} 个问题")
            print(f"问题: {question}")
            print(f"参考答案: {reference}")

        # 生成答案
        result = generate_answer_with_tools(
            llm, question, tokenizer,
            retrieval_url=retrieval_url,
            generation_url=generation_url,
            temperature=temperature
        )

        prediction = result['final_answer']
        predictions.append(prediction)
        references.append(reference)

        # 统计信息
        total_retrieve_count += result['retrieve_count']
        total_generate_count += result['generate_count']
        status_counts[result['status']] += 1

        detailed_results.append({
            'question_id': i,
            'question': question,
            'reference': reference,
            'prediction': prediction,
            'full_generation': result['full_generation'],
            'retrieve_count': result['retrieve_count'],
            'generate_count': result['generate_count'],
            'round_count': result['round_count'],
            'status': result['status']
        })

        if i < 3:  # 只为前3个显示详细信息
            print(f"预测答案: {prediction}")
            print(f"检索次数: {result['retrieve_count']}, 生成次数: {result['generate_count']}")
            print(f"状态: {result['status']}")
            print(f"完整生成过程:\n{result['full_generation']}")

    # 计算Cover EM等指标
    print("\n正在计算评估指标...")
    em_results = calculate_cover_em(predictions, references)

    # 汇总结果
    evaluation_results = {
        'metrics': em_results,
        'statistics': {
            'total_samples': len(dataset),
            'avg_retrieve_count': total_retrieve_count / len(dataset),
            'avg_generate_count': total_generate_count / len(dataset),
            'total_retrieve_count': total_retrieve_count,
            'total_generate_count': total_generate_count,
            'status_distribution': dict(status_counts)
        },
        'detailed_results': detailed_results
    }

    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description='评估RL训练的LLM模型（支持工具调用）')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--test_file', type=str, required=True,
                        help='测试文件路径(jsonl格式)')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                        help='结果输出文件')
    parser.add_argument('--retrieval_url', type=str, default='http://0.0.0.0:5003/queries',
                        help='检索服务URL')
    parser.add_argument('--generation_url', type=str, default='http://101.42.41.82:5004/generate_docs',
                        help='文档生成服务URL')
    parser.add_argument('--gpu_memory_rate', type=float, default=0.9,
                        help='GPU内存使用率')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='最大测试样本数，-1表示全部')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='并行worker数量（1=单进程，>1=多进程并行）')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='张量并行大小（单进程模式下使用的GPU数量）')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='生成温度（评估时建议使用0.0）')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='基础模型路径（当checkpoint只包含权重时使用）')

    args = parser.parse_args()

    # 检查CUDA
    if not torch.cuda.is_available():
        print("CUDA不可用，可能影响性能")

    available_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {available_gpus}")

    if args.num_workers > available_gpus:
        print(f"警告：worker数量({args.num_workers})超过可用GPU数量({available_gpus})")
        print(f"建议设置 --num_workers {available_gpus}")

    print(f"开始评估...")
    print(f"模型路径: {args.checkpoint_path}")
    print(f"测试文件: {args.test_file}")
    print(f"检索服务: {args.retrieval_url}")
    print(f"生成服务: {args.generation_url}")

    if args.num_workers > 1:
        print(f"并行模式: {args.num_workers} 个worker进程，每个使用1张GPU")
        print(f"预计加速比: ~{args.num_workers}x")
    else:
        print(f"单进程模式: {args.tensor_parallel_size}张卡并行，内存利用率{args.gpu_memory_rate}")

    print(f"生成温度: {args.temperature} {'确定性生成' if args.temperature == 0.0 else '非确定性生成'}")

    llm = None  # 初始化llm变量

    try:
        # 根据模式选择评估方法
        if args.num_workers > 1:
            # 多进程并行模式
            print(f"\n启动并行评估模式...")

            results = evaluate_model_parallel(
                args.test_file,
                args,
                num_workers=args.num_workers
            )
        else:
            # 单进程模式
            print(f"\n启动单进程评估模式...")

            # 加载模型
            llm, tokenizer = load_rl_model(
                args.checkpoint_path,
                base_model_path=args.base_model_path,
                gpu_memory_rate=args.gpu_memory_rate,
                tensor_parallel_size=args.tensor_parallel_size
            )

            # 评估模型
            results = evaluate_model_single(
                llm, tokenizer, args.test_file,
                retrieval_url=args.retrieval_url,
                generation_url=args.generation_url,
                max_samples=args.max_samples,
                temperature=args.temperature
            )

        # 打印结果
        print(f"\n{'=' * 80}")
        print("评估结果:")
        print(f"{'=' * 80}")
        print(f"Cover EM得分: {results['metrics']['cover_em_score']:.4f}")
        print(f"Exact EM得分: {results['metrics']['exact_em_score']:.4f}")
        print(f"总问题数: {results['metrics']['total_questions']}")
        print(f"Cover匹配数: {results['metrics']['cover_matches']}")
        print(f"Exact匹配数: {results['metrics']['exact_matches']}")

        print(f"\n统计信息:")
        print(f"平均检索次数: {results['statistics']['avg_retrieve_count']:.2f}")
        print(f"平均生成次数: {results['statistics']['avg_generate_count']:.2f}")
        print(f"总检索次数: {results['statistics']['total_retrieve_count']}")
        print(f"总生成次数: {results['statistics']['total_generate_count']}")

        if 'num_workers_used' in results['statistics']:
            print(f"使用worker数量: {results['statistics']['num_workers_used']}")

        print(f"\n状态分布:")
        for status, count in results['statistics']['status_distribution'].items():
            print(f"  {status}: {count} ({count / results['statistics']['total_samples'] * 100:.1f}%)")

        # 保存详细结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n详细结果已保存到: {args.output_file}")

    finally:
        # 清理临时目录
        cleanup_temp_dirs(llm)


def cleanup_temp_dirs(llm=None):
    """清理临时模型目录"""
    if llm and hasattr(llm, '_temp_model_dir'):
        import shutil
        temp_dir = llm._temp_model_dir
        if os.path.exists(temp_dir):
            print(f"清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()