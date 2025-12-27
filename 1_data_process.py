# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 配置参数 ===
BASELINE_SIZE = 1000
OURS_SIZE = 1000
OUTPUT_DIR = "./data"
# 这是你刚刚手动下载的文件名
LOCAL_FILE = "alpaca_gpt4_data_zh.json" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(">>> [1/4] 尝试读取本地数据集...")

# 检查文件是否存在
if not os.path.exists(LOCAL_FILE):
    print(f"错误：在当前目录下没找到 '{LOCAL_FILE}'！")
    print("请确认你已经手动下载了该文件，并把它放在了代码旁边。")
    exit()

try:
    # 直接用 Pandas 读取本地 JSON，不走 HuggingFace 的网络接口
    df = pd.read_json(LOCAL_FILE)
    print(f"本地读取成功！原始数据量: {len(df)} 条")
except Exception as e:
    print(f"读取失败: {e}")
    exit()

print(">>> [2/4] 执行 AlpaGasus 筛选策略...")

def calculate_quality_score(row):
    # 确保转为字符串，防止部分数据为空
    instruction = str(row.get('instruction', '')) + str(row.get('input', ''))
    output = str(row.get('output', ''))
    
    score = 0
    length = len(output)

    # 规则1: 长度
    if length < 10: return -100
    if length > 400: score += 3
    elif length > 100: score += 1

    # 规则2: 结构
    if "1." in output and "2." in output: score += 2
    if "```" in output: score += 4
    if "|" in output and "-" in output: score += 2

    # 规则3: 关键词
    keywords = ["步骤", "如何", "代码", "解释", "分析", "为什么"]
    for kw in keywords:
        if kw in instruction:
            score += 1
            break 
    return score

df['quality_score'] = df.apply(calculate_quality_score, axis=1)
print("    打分完成。")

print(">>> [3/4] 生成数据集...")

# 1. Baseline: 随机抽取
df_baseline = df.sample(n=BASELINE_SIZE, random_state=42)

# 2. Ours: 选取分数最高的
df_ours = df.sort_values(by='quality_score', ascending=False).head(OURS_SIZE)

print(f"    Baseline 平均分: {df_baseline['quality_score'].mean():.2f}")
print(f"    Ours 平均分:     {df_ours['quality_score'].mean():.2f}")

# 保存
baseline_path = os.path.join(OUTPUT_DIR, "train_baseline.json")
ours_path = os.path.join(OUTPUT_DIR, "train_ours.json")
df_baseline.to_json(baseline_path, orient="records", lines=True, force_ascii=False)
df_ours.to_json(ours_path, orient="records", lines=True, force_ascii=False)

print(f"  JSON文件已保存到 {OUTPUT_DIR} 文件夹")

print(">>> [4/4] 绘制对比图...")
plt.figure(figsize=(10, 6))
sns.histplot(df_baseline['quality_score'], color='gray', label='Baseline (Random)', kde=True, element="step", alpha=0.3)
sns.histplot(df_ours['quality_score'], color='blue', label='Ours (Selected)', kde=True, element="step", alpha=0.5)
plt.title("Data Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.legend()
plt.grid(axis='y', alpha=0.3)

img_path = os.path.join(OUTPUT_DIR, "data_distribution.png")
plt.savefig(img_path)
print(f" 图表已保存: {img_path}")

print("\n=== Step 1 完成！ ===")