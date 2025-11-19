from datasets import load_dataset
import random
import json

# 1. 加载 HotpotQA fullwiki 版本
dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

# 2. 随机抽样 523 条
train = dataset["train"]
indices = random.sample(range(len(train)), 523)
subset = [train[i] for i in indices]

# 3. 保存到当前目录
output_path = "stage1.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print("已保存到:", output_path)
