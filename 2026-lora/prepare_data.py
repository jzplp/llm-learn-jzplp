"""
读取 CMRC2018 训练数据，套入 Qwen3 对话模板，生成 torch 训练输入数据

用法一（命令行预览）:
    python prepare_data.py               # 读取全部数据并预览训练集第 1 条
    python prepare_data.py --split val   # 读取全部数据并预览验证集第 1 条

用法二（被其他脚本 import）:
    from prepare_data import prepareData
    train_dataset, val_dataset, tokenizer = prepareData()
    train_dataset, val_dataset, tokenizer = prepareData(limit=5, val_limit=10)

每条样本含 input_ids / labels，无 padding；打乱由 Trainer 负责
"""
import argparse
import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 训练数据文件
TRAIN_JSON = "squad-style-data/cmrc2018_train.json"
# 验证数据文件
VAL_JSON = "squad-style-data/cmrc2018_trial.json"
# 模型目录
MODEL_DIR = "Qwen3-0.6B-train"

def load_examples(path):
    """读取 SQuAD 格式 json，返回 (上下文, 问题, 答案) 的列表。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                answers = qa.get("answers") or []
                if answers:
                    examples.append(
                        (para["context"], qa["question"], answers[0]["text"])
                    )
    return examples

def build_prompt(question, context):
    return [
        {
            "role": "user",
            "content": (
                "请阅读下面的材料，回答问题。"
                "答案必须来自材料原文，直接输出答案文本，不要输出其他任何内容。\n\n"
                f"材料：{context}\n\n问题：{question}\n答案："
            ),
        }
    ]

def encode(example, tokenizer):
    """把一条问答转成训练用的 {input_ids, labels}。"""
    context, question, answer = example

    # 第 1 步：组合字符串模板（材料 + 问题）
    user_messages = build_prompt(question, context)

    # 第 2 步：用模板生成非答案部分（user + assistant 起始符）并直接分词
    #   tokenize=True               直接输出 token id 列表
    #   add_generation_prompt=True  追加 assistant 起始符，即模型开始生成答案的位置
    #   enable_thinking=False       关闭思考模式
    #   add_special_tokens=False    模板里已含特殊符号，不能让 tokenizer 再加 BOS/EOS
    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        add_special_tokens=False,
    )["input_ids"]

    # 第 3 步：答案直接分词，末尾补 eos（<|im_end|>，id=151645），让模型学会停止
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    # 第 4 步：拼接成完整输入，前段是问题（不算 loss），后段是答案（算 loss）
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    # 第 5 步：转 tensor
    #   dtype=torch.long   token id 必须是 int64
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class QADataset(Dataset):
    """包装样本列表，提供 PyTorch Dataset 接口，可直接交给 Trainer。"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_dataset(path, tokenizer, limit=0):
    """读取指定 json 并预处理成 QADataset，训练集/验证集共用。"""
    examples = load_examples(path)
    if limit > 0:
        examples = examples[:limit]
    return QADataset([encode(ex, tokenizer) for ex in examples])


def prepareData(limit=0, val_limit=0):
    """读取训练集和验证集并完成预处理，返回 (Dataset, Dataset, tokenizer)。

    train_dataset: 训练集，每条含 input_ids / labels
    val_dataset:   验证集，同样含 input_ids / labels
    tokenizer:     两边共用同一份分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    train_dataset = build_dataset(TRAIN_JSON, tokenizer, limit)
    val_dataset = build_dataset(VAL_JSON, tokenizer, val_limit)
    return train_dataset, val_dataset, tokenizer


def main():
    """命令行脚本入口：读取全部数据，预览指定数据集的第 1 条样本。"""
    parser = argparse.ArgumentParser(description="预览 CMRC2018 预处理结果")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="预览训练集还是验证集（默认 train）",
    )
    args = parser.parse_args()

    train_dataset, val_dataset, tokenizer = prepareData()
    dataset = train_dataset if args.split == "train" else val_dataset

    print(f"训练集：{len(train_dataset)} 条，验证集：{len(val_dataset)} 条")

    sample = dataset[0]
    print("\n=== 第 1 条样本 ===")
    for k, v in sample.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    prompt_len = int((sample["labels"] == -100).sum())
    total_len = sample["input_ids"].shape[0]
    print(f"prompt tokens = {prompt_len}, answer tokens = {total_len - prompt_len}")
    print("input_ids 解码（完整）:")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
    print("labels 解码（仅答案）:")
    answer_ids = sample["labels"][sample["labels"] != -100].tolist()
    print(tokenizer.decode(answer_ids, skip_special_tokens=False))

# 命令行调用入口
if __name__ == "__main__":
    main()
