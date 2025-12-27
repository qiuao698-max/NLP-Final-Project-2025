# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import os

# 设置绘图风格
plt.style.use('ggplot')
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun'] 
plt.rcParams['axes.unicode_minus'] = False

LOG_FILE = "./data/loss_log.json"
OUTPUT_IMG = "./data/loss_curve.png"

def plot_loss():
    print(f"[INFO] 正在读取日志文件: {LOG_FILE}...")
    
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] 找不到文件: {LOG_FILE}")
        return

    with open(LOG_FILE, "r") as f:
        data = json.load(f)
        loss_ours_plus = data.get("ours_plus", [])

    loss_baseline = data.get("baseline", [])
    loss_ours = data.get("ours", [])

    if not loss_baseline or not loss_ours:
        print("[ERROR] 数据为空，请检查 Step 2 是否完整运行。")
        return

    # === 开始绘图 ===
    plt.figure(figsize=(10, 6), dpi=120)
    
    # 绘制 Baseline (灰色虚线)
    plt.plot(loss_baseline, label="Baseline (随机筛选)", color="gray", linestyle="--", alpha=0.6, linewidth=2)
    
    # 绘制 Ours (红色实线 - 你的算法)
    plt.plot(loss_ours, label="Ours (AlpaGasus 策略)", color="#E24A33", linewidth=2.5)

    plt.title("模型训练 Loss 对比图", fontsize=14)
    plt.xlabel("训练步数 (Steps)", fontsize=12)
    plt.ylabel("Loss 值 (越低越好)", fontsize=12)
    if loss_ours_plus:
        plt.plot(loss_ours_plus, label="Ours+ (Rank=32 + NEFTune)", color="green", linewidth=2.5, linestyle="-.")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS]以此证明你的实验成功！图片已保存为: {OUTPUT_IMG}")
    print("快去打开这张图看看吧！")

if __name__ == "__main__":
    plot_loss()