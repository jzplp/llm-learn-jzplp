"""
用 LoRA 微调 Qwen3-0.6B
用法:
    python train_lora.py  # 全量训练集 + 全量验证集
    python train_lora.py --limit 100 --val_limit 50  # 训练集只用前100条，验证集前50条
    python train_lora.py --resume Qwen3-0.6B-lora/checkpoint-200  # 从指定 checkpoint 继续训练

暂停：训练中按 Ctrl+C；恢复时保持 --limit / --val_limit 等参数与上次一致
"""
import argparse

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# 模型目录与数据预处理函数复用 prepare_data.py
from prepare_data import MODEL_DIR, prepareData

# LoRA 权重输出目录（只保存适配器，不包含基座模型）
OUTPUT_DIR = "Qwen3-0.6B-lora"


def build_model():
    """加载模型，并用 peft 给它挂上 LoRA 适配器。"""
    # CPU 训练统一使用 fp32（原始权重是 bf16，这里会转换成 fp32）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
    )
    # 训练时关闭 KV cache：它只用于生成加速，训练阶段用不到
    model.config.use_cache = False

    # LoRA 配置：冻结原模型参数，只在指定线性层旁加可训练的低秩矩阵
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        r=8,  # LoRA 秩，越大可训练参数越多、表达能力越强
        lora_alpha=16,  # 缩放系数，实际缩放为 alpha / r，通常取 2*r
        bias="none",  # 不训练 bias
        # 作用在 Qwen3 注意力层的 q/k/v/o 投影矩阵上
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    # get_peft_model 会冻结基座参数，只保留 LoRA 参数可训练
    model = get_peft_model(model, lora_config)
    # 打印可训练参数量及占比，便于确认 LoRA 是否挂载成功
    model.print_trainable_parameters()
    return model


def main():
    # 命令行参数：训练集/验证集条数限制
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="训练集只取前 N 条（0=全部）")
    parser.add_argument("--val_limit", type=int, default=0, help="验证集只取前 N 条（0=全部）")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定 checkpoint 目录继续训练，例如 Qwen3-0.6B-lora/checkpoint-200",
    )
    args = parser.parse_args()

    # 复用 prepare_data.py：返回训练集、验证集，已套好 Qwen3 模板，prompt 部分的 labels 为 -100
    train_dataset, val_dataset, _ = prepareData(args.limit, args.val_limit)
    print(f"训练样本数: {len(train_dataset)}，验证样本数: {len(val_dataset)}")

    # 加载模型并挂 LoRA
    model = build_model()

    # 按 3 个 epoch 的总优化步数计算 3% 预热步数
    steps_per_epoch = max(len(train_dataset) // 16, 1)
    warmup_steps = int(steps_per_epoch * 3 * 0.03)

    # TrainingArguments 只是训练配置容器，由 Trainer 读取后执行
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # checkpoint 和最终权重保存目录
        per_device_train_batch_size=1,  # batch size，与 prepare_data 一致
        gradient_accumulation_steps=16,  # 累积 16 步再更新，等效增大 batch
        num_train_epochs=3,  # 训练轮数
        learning_rate=2e-4,  # LoRA 参数学习率，通常比全参微调大
        lr_scheduler_type="cosine",  # 余弦退火学习率
        warmup_steps=warmup_steps,  # 前 3% 步数线性预热
        logging_steps=10,  # 每 10 步打印一次 loss
        save_strategy="steps",  # 按步数保存 checkpoint
        save_steps=10,  # 每 10 步保存一次
        save_total_limit=2,  # 最多保留 2 个 checkpoint
        eval_strategy="steps",  # 按步数计算 val loss
        eval_steps=200,  # 每 200 步验证一次
        per_device_eval_batch_size=1,  # 验证 batch size
        dataloader_pin_memory=False,  # CPU 训练没有加速器，关闭 pin memory
        optim="adamw_torch",  # 使用 PyTorch 原生 AdamW
        report_to="none",  # 不上报 wandb 等平台
        seed=42,  # 随机种子，保证可复现
    )

    # Trainer 负责：按批读取数据、优化器、学习率调度、日志、保存等训练流程
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 开始训练；args.resume 为 None 时从头开始
    trainer.train(resume_from_checkpoint=args.resume)

    # 只保存 LoRA 适配器（几十 MB），不保存整个模型和分词器
    trainer.save_model(OUTPUT_DIR)
    # 保存训练状态（含 log_history）到 trainer_state.json
    trainer.save_state()

    print(f"LoRA 适配器已保存到: {OUTPUT_DIR}")
    print(f"训练日志已保存到: {OUTPUT_DIR}/trainer_state.json")

if __name__ == "__main__":
    main()
