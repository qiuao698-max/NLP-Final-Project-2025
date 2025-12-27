# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

MODEL_PATH = "./Qwen2.5-0.5B-Instruct" 
MAX_STEPS = 120          # 【改进1】步数翻倍
LEARNING_RATE = 2e-4   

def train_improved(data_file, output_sub_dir):
    print(f"\n>>> [改进实验] 开始训练 Ours+ (High Rank) 模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True)
    
    # 【改进2】LoRA 配置增强
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=32,            # <--- 原来是 8，现在改成 32 (脑容量变大)
        lora_alpha=64,   # <--- 配合 r=32，alpha 通常设为 2*r
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 全模块微调
    )
    model = get_peft_model(model, peft_config)
    print("    模型加载成功！LoRA Rank 已提升至 32 (参数量增加)。")

    # 打印一下可训练参数量，让你写报告用
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=data_file, split="train")
    
    def process_func(example):
        instruction = example.get('instruction', '') + example.get('input', '')
        response = example.get('output', '')
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False) + tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(input_ids) > 128: input_ids = input_ids[:128]
        return {"input_ids": input_ids, "labels": [-100] * len(tokenizer.encode(prompt, add_special_tokens=False)) + input_ids[len(tokenizer.encode(prompt, add_special_tokens=False)):]}

    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_sub_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=2,
        max_steps=MAX_STEPS,          
        learning_rate=LEARNING_RATE,
        save_steps=120,          
        neftune_noise_alpha=5.0,     
        fp16=True,
        logging_steps=5,
        use_cpu=False if torch.cuda.is_available() else True, 
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
    trainer.train()
    
    # 保存训练好的模型
    save_path = "./saved_models/ours_plus"
    print(f"正在保存改进版模型到 {save_path} ...")
    model.save_pretrained(save_path)
    
    loss_history = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
    return loss_history

if __name__ == "__main__":
    # 使用筛选后的高质量数据进行训练
    loss_improved = train_improved("./data/train_ours.json", "./results/ours_plus")
    
    # 把这个新数据保存下来，准备画图
    import json
    # 读取旧数据
    if os.path.exists("./data/loss_log.json"):
        with open("./data/loss_log.json", "r") as f:
            old_data = json.load(f)
    else:
        old_data = {}
    
    # 加入新数据
    old_data["ours_plus"] = loss_improved
    
    # 存回去
    with open("./data/loss_log.json", "w") as f:
        json.dump(old_data, f)
    print("\n改进实验完成！数据已更新，快去画图吧！")