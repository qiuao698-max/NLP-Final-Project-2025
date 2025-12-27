# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 设置路径
BASE_MODEL_PATH = "./Qwen2.5-0.5B-Instruct"    # 原版底座
ADAPTER_PATH = "./saved_models/ours_final"     # 你微调出来的“大脑”

print(f"[INFO] 正在加载基础模型: {BASE_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True, device_map="auto")

print(f"[INFO] 正在挂载你的微调参数: {ADAPTER_PATH} ...")
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("加载成功！现在你可以跟它聊天了。")
except Exception as e:
    print(f"加载失败，请确认你运行了 2_train_save.py。错误: {e}")
    exit()

def chat(query):
    # 构造对话格式
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, 
        temperature=0.7,    # 0.7 比较有创造力
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

print("\n" + "="*40)
print("聊天机器人已启动 (输入 'exit' 退出)")
print("="*40)

while True:
    q = input("\n请提问 (User): ")
    if q.lower() in ['exit', 'quit']:
        break
    
    print("Thinking...", end="", flush=True)
    ans = chat(q)
    print(f"\rAI 回答: {ans}")