import json
import re
from collections import Counter, defaultdict


def normalize_answer(s):
    """标准化答案用于EM计算"""
    # 转小写
    s = s.lower()
    # 去除标点
    s = re.sub(r'[^\w\s]', '', s)
    # 去除多余空格
    s = ' '.join(s.split())
    return s.strip()


def calculate_exact_match(pred, gold):
    """计算是否完全匹配（严格）"""
    return normalize_answer(pred) == normalize_answer(gold)


def calculate_contain_match(pred, gold):
    """计算是否包含匹配（宽松）"""
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    # 检查标准答案是否在预测答案中
    return norm_gold in norm_pred if norm_gold else False


def analyze_results(file_path):
    """分析评估结果"""
    results = []

    # 读取结果
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # 基础统计
    total = len(results)
    print(f"总样本数: {total}")
    print("=" * 50)

    # 计算两种匹配分数
    em_correct = 0
    contain_correct = 0

    for item in results:
        pred_ans = item.get('pred_ans', '').strip()
        gold_ans = item.get('answer', '').strip()

        if calculate_exact_match(pred_ans, gold_ans):
            em_correct += 1
            contain_correct += 1  # 严格匹配的肯定也是包含匹配
        elif calculate_contain_match(pred_ans, gold_ans):
            contain_correct += 1

    em_score = em_correct / total * 100 if total > 0 else 0
    contain_score = contain_correct / total * 100 if total > 0 else 0

    print(f"严格匹配 (Exact Match): {em_correct}/{total} = {em_score:.2f}%")
    print(f"宽松匹配 (Contain Match): {contain_correct}/{total} = {contain_score:.2f}%")
    print(f"仅包含不完全匹配: {contain_correct - em_correct} 个")
    print("=" * 50)

    # 统计完成状态
    status_counter = Counter([item.get('stop_reason_final', 'unknown') for item in results])
    print("完成状态分布:")
    for status, count in status_counter.most_common():
        print(f"  {status}: {count} ({count / total * 100:.1f}%)")
    print("=" * 50)

    # 统计工具使用情况
    search_counts = []
    gendoc_counts = []

    for item in results:
        search_counts.append(item.get('search_count', 0))
        gendoc_counts.append(item.get('gendoc_count', 0))

    # 搜索工具统计
    print("搜索工具使用统计:")
    print(f"  平均使用次数: {sum(search_counts) / len(search_counts):.2f}")
    print(f"  最大使用次数: {max(search_counts)}")
    print(f"  未使用搜索的样本数: {search_counts.count(0)}")

    # 文档生成工具统计
    print("\n文档生成工具使用统计:")
    print(f"  平均使用次数: {sum(gendoc_counts) / len(gendoc_counts):.2f}")
    print(f"  最大使用次数: {max(gendoc_counts)}")
    print(f"  未使用生成的样本数: {gendoc_counts.count(0)}")
    print("=" * 50)

    # 展示一些例子
    print("\n随机展示5个例子:")
    import random
    sample_items = random.sample(results, min(5, len(results)))

    for i, item in enumerate(sample_items, 1):
        pred_ans = item.get('pred_ans', '')
        gold_ans = item.get('answer', '')
        is_exact = calculate_exact_match(pred_ans, gold_ans)
        is_contain = calculate_contain_match(pred_ans, gold_ans)

        print(f"\n例子 {i}:")
        print(f"  问题: {item.get('question', '')[:100]}...")
        print(f"  标准答案: {gold_ans}")
        print(f"  预测答案: {pred_ans}")
        print(f"  严格匹配: {'✓' if is_exact else '✗'}")
        print(f"  包含匹配: {'✓' if is_contain else '✗'}")
        print(f"  搜索次数: {item.get('search_count', 0)}, 生成次数: {item.get('gendoc_count', 0)}")

    # 返回两种分数
    return em_score, contain_score


# 主函数
if __name__ == "__main__":
    file_path = "/root/autodl-tmp/GenR1-Searcher/data/eval_set/musique_500-ckptstep113_two_tools_temp0.0_typeNone.jsonl"

    print(f"分析文件: {file_path}\n")
    em_score, contain_score = analyze_results(file_path)

    print(f"\n最终分数:")
    print(f"  严格匹配 (EM): {em_score:.2f}%")
    print(f"  宽松匹配 (Contains): {contain_score:.2f}%")