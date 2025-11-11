# gendoc_server.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import argparse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import logging
import time
from datetime import datetime
import requests

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenDocModel:
    def __init__(self, model_path: str, wiki_service_url: str = "http://101.42.41.82:5004"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wiki_service_url = wiki_service_url  # 添加wiki服务URL

        logger.info(f"🚀 加载GenDoc模型: {model_path} 到 {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        logger.info(f"✅ GenDoc模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB")

        # 生成参数
        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    # def _retrieve_wiki_documents(self, queries: List[str], k: int = 1) -> List[List[str]]:
    #     """从wiki语料库中检索相关文档"""
    #     try:
    #         response = requests.post(
    #             f"{self.wiki_service_url}/queries",
    #             json={"queries": queries, "k": k},
    #             timeout=120
    #         )
    #
    #         if response.status_code == 200:
    #             result = response.json()
    #             return result.get("answers", [[] for _ in queries])
    #         else:
    #             logger.error(f"Wiki检索失败，状态码: {response.status_code}")
    #             return [[] for _ in queries]
    #
    #     except Exception as e:
    #         logger.error(f"Wiki检索服务调用失败: {e}")
    #         return [[] for _ in queries]
    def _retrieve_wiki_documents(self, queries: List[str], k: int = 1) -> List[List[str]]:
        """从wiki语料库中检索相关文档"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'GenDocServer/1.0'  # 添加 User-Agent
            }

            response = requests.post(
                f"{self.wiki_service_url}/queries",
                json={"queries": queries, "k": k},
                headers=headers,  # 添加请求头
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("answers", [[] for _ in queries])
            else:
                logger.error(f"Wiki检索失败，状态码: {response.status_code}")
                return [[] for _ in queries]

        except Exception as e:
            logger.error(f"Wiki检索服务调用失败: {e}")
            return [[] for _ in queries]

    def generate_documents(self, queries: List[str], k: int = 3) -> List[str]:
        """全并行生成文档 - 所有query×k个文档一次性生成"""
        total_docs = len(queries) * k
        logger.info(f"📝 尝试并行生成 {len(queries)} 个查询 × {k} 个文档 = {total_docs} 个文档")

        try:
            # 尝试全并行生成
            return self._generate_all_parallel(queries, k)

        except torch.cuda.OutOfMemoryError:
            logger.warning("⚠️ 全并行生成显存不足，回退到分批处理")
            torch.cuda.empty_cache()
            return self._generate_batch_fallback(queries, k)

        except Exception as e:
            logger.error(f"❌ 全并行生成失败: {e}，回退到分批处理")
            return self._generate_batch_fallback(queries, k)

    def _generate_all_parallel(self, queries: List[str], k: int) -> List[str]:
        """尝试全并行生成所有文档"""

        # 先检索每个query的参考文档
        logger.info(f"🔍 开始检索 {len(queries)} 个查询的参考文档...")
        reference_docs = self._retrieve_wiki_documents(queries, k=1)  # 每个query检索1个参考文档

        # 1. 构建所有prompts (N×k个)
        all_prompts = []
        query_indices = []  # 记录每个prompt对应的query索引

        for query_idx, (query, ref_docs) in enumerate(zip(queries, reference_docs)):
            # 获取该query的参考文档（如果有的话）
            reference_doc = ref_docs[0] if ref_docs else None
            prompt = self._build_prompt(query, reference_doc)

            for doc_idx in range(k):
                all_prompts.append(prompt)
                query_indices.append(query_idx)

        logger.info(f"🚀 构建了 {len(all_prompts)} 个prompts，开始批量生成...")

        # 2. 检查显存是否足够
        if not self._check_memory_sufficient(len(all_prompts)):
            raise torch.cuda.OutOfMemoryError("预估显存不足")

        # 3. 批量tokenize所有prompts
        inputs = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)

        # 4. 批量生成所有文档
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **self.generation_config
            )

        # 5. 批量解码
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        logger.info(f"✅ 成功生成 {len(generated_texts)} 个文档")

        # 6. 重新组织结果 - 按query分组
        results = self._reorganize_results(queries, generated_texts, query_indices, k)

        # ✅ 打印每个 query 的文档总词数
        for query_idx, merged_doc in enumerate(results):
            total_words = len(merged_doc.split())
            logger.info(f"📊 Query {query_idx + 1} 总词数: {total_words}")

        return results

    def _check_memory_sufficient(self, num_prompts: int) -> bool:
        """检查显存是否足够进行全并行生成"""
        if not torch.cuda.is_available():
            return False

        # 获取当前显存状态
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory
        free_gb = free_memory / (1024 ** 3)

        # 估算需要的额外显存 (很粗略的估计)
        # 假设每个prompt在生成时需要额外的显存
        estimated_need_gb = num_prompts * 0.2  # 每个prompt大约需要200MB

        logger.info(f"🔍 显存检查: 可用={free_gb:.1f}GB, 预估需要={estimated_need_gb:.1f}GB")

        return free_gb > estimated_need_gb * 1.2  # 留20%安全边距

    def _reorganize_results(self, queries: List[str], generated_texts: List[str],
                            query_indices: List[int], k: int) -> List[str]:
        """将生成的文档按query重新组织"""
        results = [[] for _ in queries]

        # 将每个生成的文档分配到对应的query
        for i, (text, query_idx) in enumerate(zip(generated_texts, query_indices)):
            if text.strip():
                doc_num = (i % k) + 1  # 文档编号 1, 2, 3, ...
                formatted_doc = f"Document {doc_num}: {text.strip()}"
                results[query_idx].append(formatted_doc)

        # 合并每个query的所有文档
        final_results = []
        for query_idx, query_docs in enumerate(results):
            if query_docs:
                merged_doc = "\n\n".join(query_docs)
            else:
                merged_doc = f"No relevant document could be generated for query: {queries[query_idx]}"
            final_results.append(merged_doc)

        return final_results

    def _generate_batch_fallback(self, queries: List[str], k: int) -> List[str]:
        """回退方案：分批处理"""
        results = []

        # 动态计算安全的batch size
        batch_size = self._calculate_safe_batch_size(len(queries), k)
        logger.info(f"🔄 使用回退方案，batch_size={batch_size}")

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]

            try:
                # 尝试对这个batch进行并行处理
                batch_results = self._generate_all_parallel(batch_queries, k)
                results.extend(batch_results)

            except (torch.cuda.OutOfMemoryError, Exception):
                # 如果还是失败，回退到最保守的逐个处理
                logger.warning(f"⚠️ batch {i // batch_size + 1} 仍然失败，使用逐个处理")
                torch.cuda.empty_cache()

                for query in batch_queries:
                    query_result = self._generate_single_query_documents(query, k)
                    results.append(query_result)

            # 每个batch后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _generate_single_query_documents(self, query: str, k: int) -> str:
        """最保守的方案：为单个query顺序生成k个文档"""
        # 先检索参考文档
        reference_docs = self._retrieve_wiki_documents([query], k=1)
        reference_doc = reference_docs[0][0] if reference_docs and reference_docs[0] else None

        prompt = self._build_prompt(query, reference_doc)
        docs = []

        for i in range(k):
            try:
                doc = self._generate_single_document(prompt)
                if doc.strip():
                    docs.append(f"Document {i + 1}: {doc.strip()}")
            except Exception as e:
                logger.error(f"❌ 文档 {i + 1} 生成失败: {e}")

        return "\n\n".join(docs) if docs else f"Failed to generate documents for query: {query}"

    def _calculate_safe_batch_size(self, num_queries: int, k: int) -> int:
        """根据可用显存计算安全的batch size"""
        if not torch.cuda.is_available():
            return 1

        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_gb = free_memory / (1024 ** 3)

        # 更激进的batch size计算
        if free_gb > 10:
            return min(8, num_queries)  # 显存充足：最多8个queries
        elif free_gb > 6:
            return min(4, num_queries)  # 显存较好：最多4个queries
        elif free_gb > 3:
            return min(2, num_queries)  # 显存紧张：最多2个queries
        else:
            return 1  # 显存不足：逐个处理

    def _build_prompt(self, query: str, reference_doc: str = None) -> str:
        """构建生成文档的提示词"""
        system_content = "You are a document generator. Generate a background document from Wikipedia to answer the given question."

        if reference_doc:
            system_content += f"\nHere are some references that may be relevant:\n{reference_doc}"
        else:
            system_content += " Output only the document content."

        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": query
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def _generate_single_document(self, prompt: str) -> str:
        """生成单个文档"""
        try:
            # Tokenize输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **self.generation_config
                )

            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"❌ 单文档生成失败: {e}")
            return ""

    def get_memory_usage(self):
        """获取显存使用情况"""
        if torch.cuda.is_available():
            return {
                "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024 ** 3:.1f}GB"
            }
        return {"status": "CUDA not available"}


class GenDocServer:
    def __init__(self, model_path: str, wiki_service_url: str = "http://101.42.41.82:5004",
                 host: str = "0.0.0.0", port: int = 5004):
        self.host = host
        self.port = port

        # 直接初始化模型（不使用Ray）
        logger.info("🚀 初始化GenDoc模型...")
        self.gendoc_model = GenDocModel(model_path, wiki_service_url)

        # 预热模型
        logger.info("🔥 预热GenDoc模型...")
        test_result = self.gendoc_model.generate_documents(["test query"], k=1)
        logger.info(f"✅ 模型预热完成: {test_result[0][:100]}...")

        # 创建FastAPI应用
        self.app = FastAPI(title="GenDoc Server")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/generate_docs")
        async def generate_docs(request: Request):
            try:
                data = await request.json()
                queries = data.get("queries", [])
                k = data.get("k", 3)  # 默认生成3个文档
                print(f'生成{k}个文档')

                if not queries:
                    return JSONResponse({"error": "No queries provided"}, status_code=400)

                logger.info(f"📥 收到 {len(queries)} 个生成请求")
                start_time = time.time()

                # 直接调用模型生成文档（不使用Ray）
                documents = self.gendoc_model.generate_documents(queries, k)

                end_time = time.time()
                logger.info(f"📤 生成完成，耗时: {end_time - start_time:.2f}秒")
                print(f'生成文档如下: {documents}')

                return JSONResponse({
                    "documents": documents,
                    "count": len(documents),
                    "generation_time": f"{end_time - start_time:.2f}s"
                })

            except Exception as e:
                logger.error(f"❌ 生成请求处理失败: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/health")
        async def health_check():
            try:
                memory_info = self.gendoc_model.get_memory_usage()
                return JSONResponse({
                    "status": "healthy",
                    "memory": memory_info,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                return JSONResponse({"status": "unhealthy", "error": str(e)})

        @self.app.get("/")
        async def root():
            return JSONResponse({
                "message": "GenDoc Server is running",
                "endpoints": {
                    "generate_docs": "POST /generate_docs - 生成文档",
                    "health": "GET /health - 健康检查"
                }
            })

    def run(self):
        logger.info(f"🌐 启动GenDoc服务器 http://{self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def main():
    parser = argparse.ArgumentParser(description="GenDoc Server")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen-2.5-7B-Instruct")
    parser.add_argument("--wiki_service_url", type=str, default="http://101.42.41.82:5004",
                        help="Wiki检索服务的URL")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5004)

    args = parser.parse_args()

    try:
        # 直接初始化并启动服务器（不使用Ray）
        server = GenDocServer(args.model_path, args.wiki_service_url, args.host, args.port)
        server.run()

    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
