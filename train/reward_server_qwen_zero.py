import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
import string
from collections import Counter

logger = init_logger(__name__)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )


def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：""《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()


class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = datasets.load_dataset("json", data_files=args.data_path)["train"].to_list()
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        print(len(self.eval_data_dict))
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
        self.all_batches = []  # 存储所有批次数据
        self.cnt = 0
        self.stage = args.stage

    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_qa(self, query):
        # 新格式解析
        if "<|im_start|>user" in query and "<|im_start|>assistant" in query:
            # 提取问题部分
            user_start = query.find("<|im_start|>user") + len("<|im_start|>user")
            user_end = query.find("<|im_end|>", user_start)
            question = query[user_start:user_end].strip()

            # 提取解决方案部分
            assistant_start = query.find("<|im_start|>assistant") + len("<|im_start|>assistant")
            assistant_end = query.find("<|im_end|>", assistant_start)
            if assistant_end == -1:  # 如果没有结束标记，取到最后
                solution = query[assistant_start:].strip()
            else:
                solution = query[assistant_start:assistant_end].strip()

        else:
            # 保留旧格式兼容性
            remove_prefix = " ".join(query.split("\n\nUser:")[1:])
            question = remove_prefix.split("\nAssistant: <think>")[0].strip()
            solution = query.split("\nAssistant: <think>")[-1].strip()

        return question, solution

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict[query]

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_reward(self, queries):
        assert self.stage in [1, 2, 3], "The stage can only be 1, 2, or 3."

        if self.stage == 1:
            format_reward = 0.5
            answer_reward = 0
        else:       # 2, 3 stage
            format_reward = 0
            answer_reward = 1

        # 第一步：预处理所有queries
        for i in range(len(queries)):
            queries[i] = (
                    strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                    + self.tokenizer.eos_token
            )

        logger.info(f"queries[0]: {queries[0]}")

        # 第二步：解析每个query并计算分数
        batch_data = []  # 当前批次的数据
        scores = []  # 返回的分数列表

        for i, query in enumerate(queries):
            self.cnt += 1

            # 解析问题和解决方案
            question, solution = self.get_qa(query)

            # 初始化分数和完成状态
            score = 0.0
            finished = "0"

            # 第三步：检查是否完成（有answer标签）
            has_answer_tags = "<answer>" in solution and "</answer>" in solution
            if not has_answer_tags:
                finished = "0"
            else:
                finished = "1"

                # 答案奖励
                pred_answer = self.get_query_pred(solution)
                ground_truth = self.get_query_answer(question)
                cover_flag = cover_exact_match_score_1(pred_answer, ground_truth)
                print('---'*10)
                print(f'模型答案: {pred_answer}\n 标准答案: {ground_truth}')
                print(f'cover_flag: {cover_flag}')
                if cover_flag:
                    score += answer_reward

                # 第四步：计算各种格式标签奖励（只有完成的才能加分）
                count_begin_query = solution.count("<|begin_of_query|>")
                count_end_query = solution.count("</|end_of_query|>")

                count_begin_gen = solution.count("<|begin_of_generation|>")
                count_end_gen = solution.count("</|end_of_generation|>")

                count_begin_think = solution.count("<think>")
                count_end_think = solution.count("</think>")

                count_begin_answer = solution.count("<answer>")
                count_end_answer = solution.count("</answer>")

                # 各种标签奖励
                if count_begin_query == count_end_query >= 1:
                    score += format_reward

                if count_begin_gen == count_end_gen >= 1:
                    score += format_reward

                if count_begin_think == count_end_think >= 1:
                    score += format_reward

                if count_begin_answer == count_end_answer >= 1:
                    score += format_reward

            # 保存当前样本数据
            sample_data = {
                "question": question,
                "solution": solution,
                "score": score,
                "finished": finished
            }

            batch_data.append(sample_data)
            scores.append(score)

        # 第五步：保存批次数据
        self._save_batch_data(batch_data)

        return scores

    def _save_batch_data(self, batch_data):
        """保存批次数据到JSON文件"""
        # 添加当前批次到总列表
        self.all_batches.append(batch_data)

        # 保存到JSON文件
        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.all_batches, f, ensure_ascii=False, indent=2)

        # 输出简单统计
        total_samples = len(batch_data)
        finished_count = sum(1 for sample in batch_data if sample['finished'] == "1")
        avg_score = sum(sample['score'] for sample in batch_data) / total_samples if total_samples > 0 else 0.0

        logger.info(f"Batch {len(self.all_batches)} saved:")
        logger.info(f"  - Samples: {total_samples}")
        logger.info(f"  - Finished: {finished_count}/{total_samples}")
        logger.info(f"  - Avg score: {avg_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSON log file")
    parser.add_argument("--stage", type=int, default=-1, help="Stage number")

    args = parser.parse_args()

    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

# example:
# python reward_server_qwen_zero.py --data_path /path/to/data.json --reward_pretrain /path/to/model --log_file /path/to/results.json --port 1278 --host 127.0.0.1