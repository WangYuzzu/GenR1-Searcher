import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse


def count_tag_pairs(solution):
    """统计solution中各种标签对的数量"""
    tag_counts = {}

    # 统计 <think> </think> 配对
    think_start = solution.count("<think>")
    think_end = solution.count("</think>")
    tag_counts['think_pairs'] = min(think_start, think_end)

    # 统计 <|begin_of_query|> </|end_of_query|> 配对
    query_start = solution.count("<|begin_of_query|>")
    query_end = solution.count("</|end_of_query|>")
    tag_counts['query_pairs'] = min(query_start, query_end)

    # 统计 <|begin_of_generation|> </|end_of_generation|> 配对
    gen_start = solution.count("<|begin_of_generation|>")
    gen_end = solution.count("</|end_of_generation|>")
    tag_counts['generation_pairs'] = min(gen_start, gen_end)

    return tag_counts


def analyze_reward_data(json_file, samples_per_step=144):
    """分析奖励数据"""

    # 读取数据
    print(f"Reading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        all_batches = json.load(f)

    # 展平数据（因为每个batch只有1个样本）
    all_samples = []
    for batch in all_batches:
        all_samples.extend(batch)

    print(f"Total samples: {len(all_samples)}")

    # 按step重新组织数据
    steps_data = []
    for i in range(0, len(all_samples), samples_per_step):
        step_samples = all_samples[i:i + samples_per_step]
        steps_data.append(step_samples)

    print(f"Total steps: {len(steps_data)}")

    # 统计每个step的指标
    step_stats = []

    for step_idx, step_samples in enumerate(steps_data):
        step_num = step_idx + 1

        # 初始化统计变量
        total_think_pairs = 0
        total_query_pairs = 0
        total_generation_pairs = 0
        total_score = 0
        valid_samples = 0

        for sample in step_samples:
            if 'solution' in sample and 'score' in sample:
                # 统计标签对
                tag_counts = count_tag_pairs(sample['solution'])
                total_think_pairs += tag_counts['think_pairs']
                total_query_pairs += tag_counts['query_pairs']
                total_generation_pairs += tag_counts['generation_pairs']

                # 统计分数
                total_score += sample['score']
                valid_samples += 1

        # 计算平均值
        if valid_samples > 0:
            avg_think_pairs = total_think_pairs / valid_samples
            avg_query_pairs = total_query_pairs / valid_samples
            avg_generation_pairs = total_generation_pairs / valid_samples
            avg_score = total_score / valid_samples
        else:
            avg_think_pairs = avg_query_pairs = avg_generation_pairs = avg_score = 0

        step_stats.append({
            'step': step_num,
            'avg_think_pairs': avg_think_pairs,
            'avg_query_pairs': avg_query_pairs,
            'avg_generation_pairs': avg_generation_pairs,
            'avg_score': avg_score,
            'valid_samples': valid_samples
        })

        # 打印每个step的统计
        print(f"Step {step_num:3d}: "
              f"Think={avg_think_pairs:.2f}, "
              f"Query={avg_query_pairs:.2f}, "
              f"Gen={avg_generation_pairs:.2f}, "
              f"Score={avg_score:.2f}, "
              f"Samples={valid_samples}")

    return step_stats


def plot_statistics(step_stats, save_path=None):
    """绘制统计折线图"""

    # 提取数据
    steps = [stat['step'] for stat in step_stats]
    think_pairs = [stat['avg_think_pairs'] for stat in step_stats]
    query_pairs = [stat['avg_query_pairs'] for stat in step_stats]
    generation_pairs = [stat['avg_generation_pairs'] for stat in step_stats]
    scores = [stat['avg_score'] for stat in step_stats]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练过程中各指标变化趋势', fontsize=16, fontweight='bold')

    # 1. Think标签对数量
    ax1.plot(steps, think_pairs, 'b-', marker='o', linewidth=2, markersize=4)
    ax1.set_title('Think标签对平均数量', fontsize=12)
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('平均数量')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 2. Query标签对数量
    ax2.plot(steps, query_pairs, 'g-', marker='s', linewidth=2, markersize=4)
    ax2.set_title('Query标签对平均数量', fontsize=12)
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('平均数量')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 3. Generation标签对数量
    ax3.plot(steps, generation_pairs, 'r-', marker='^', linewidth=2, markersize=4)
    ax3.set_title('Generation标签对平均数量', fontsize=12)
    ax3.set_xlabel('训练步数')
    ax3.set_ylabel('平均数量')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # 4. 平均分数
    ax4.plot(steps, scores, 'purple', marker='D', linewidth=2, markersize=4)
    ax4.set_title('平均奖励分数', fontsize=12)
    ax4.set_xlabel('训练步数')
    ax4.set_ylabel('平均分数')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


def plot_combined_statistics(step_stats, save_path=None):
    """绘制综合统计图"""

    # 提取数据
    steps = [stat['step'] for stat in step_stats]
    think_pairs = [stat['avg_think_pairs'] for stat in step_stats]
    query_pairs = [stat['avg_query_pairs'] for stat in step_stats]
    generation_pairs = [stat['avg_generation_pairs'] for stat in step_stats]
    scores = [stat['avg_score'] for stat in step_stats]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 左侧y轴：标签对数量
    ax1.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('标签对平均数量', fontsize=12)

    line1 = ax1.plot(steps, think_pairs, 'b-', marker='o', linewidth=2, label='Think标签对')
    line2 = ax1.plot(steps, query_pairs, 'g-', marker='s', linewidth=2, label='Query标签对')
    line3 = ax1.plot(steps, generation_pairs, 'r-', marker='^', linewidth=2, label='Generation标签对')

    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 右侧y轴：分数
    ax2 = ax1.twinx()
    ax2.set_ylabel('平均奖励分数', fontsize=12)
    line4 = ax2.plot(steps, scores, 'purple', marker='D', linewidth=2, label='奖励分数')
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=0)

    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('训练过程中各指标变化趋势（综合视图）', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        combined_path = save_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"综合图表已保存到: {combined_path}")

    plt.show()


def save_statistics_to_csv(step_stats, csv_path):
    """保存统计数据到CSV文件"""
    import csv

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['step', 'avg_think_pairs', 'avg_query_pairs', 'avg_generation_pairs', 'avg_score',
                      'valid_samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for stat in step_stats:
            writer.writerow(stat)

    print(f"统计数据已保存到: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='分析奖励数据统计')
    parser.add_argument('--json_file', type=str, required=True, help='JSON日志文件路径')
    parser.add_argument('--samples_per_step', type=int, default=144, help='每步的样本数量')
    parser.add_argument('--save_plot', type=str, default='reward_statistics.png', help='保存图表的路径')
    parser.add_argument('--save_csv', type=str, default='reward_statistics.csv', help='保存CSV的路径')

    args = parser.parse_args()

    # 分析数据
    step_stats = analyze_reward_data(args.json_file, args.samples_per_step)

    # 绘制图表
    plot_statistics(step_stats, args.save_plot)
    plot_combined_statistics(step_stats, args.save_plot)

    # 保存CSV
    save_statistics_to_csv(step_stats, args.save_csv)

    # 打印总结
    print(f"\n=== 分析总结 ===")
    print(f"总步数: {len(step_stats)}")
    if step_stats:
        print(f"最终平均Think标签对: {step_stats[-1]['avg_think_pairs']:.2f}")
        print(f"最终平均Query标签对: {step_stats[-1]['avg_query_pairs']:.2f}")
        print(f"最终平均Generation标签对: {step_stats[-1]['avg_generation_pairs']:.2f}")
        print(f"最终平均分数: {step_stats[-1]['avg_score']:.2f}")


if __name__ == "__main__":
    main()

# 使用示例:
# python reward_analysis.py --json_file results.json --samples_per_step 144 --save_plot reward_trends.png --save_csv reward_data.csv