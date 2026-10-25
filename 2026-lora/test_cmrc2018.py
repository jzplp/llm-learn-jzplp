import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# trial 数据集
INPUT_FILE = "squad-style-data/cmrc2018_trial.json"
# 预测结果输出文件
OUTPUT_FILE = "squad-style-data/cmrc2018_trial_result.json"
# 模型文件路径
MODEL_NAME = "../Qwen3-0.6B"
# 记录起点，用于统计总耗时
START_TIME = time.perf_counter()

# 读取squad格式数据集，输出数组中元素为{id, question, context}
def load_samples(path):
    dataset = json.load(open(path, encoding="utf-8"))
    samples = []
    for article in dataset["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                samples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": para["context"],
                    }
                )
    return samples

# 构造单轮对话消息，要求模型只输出材料中的原文答案
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

# 对单条问题做一次生成答案字符串并返回
def generate_answer(model, tokenizer, messages):
    # 套用 Qwen3 的对话模板 enable_thinking=False 关闭思考模式
    text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # no_grad：推理阶段关闭梯度记录
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # 最多生成512个token
            do_sample=False, # 贪心解码
        )
    # 只解码新生成的部分（去掉输入的 prompt）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    # 取第一行非空文本作为答案
    return response or ""

# ---- 主流程 ----
# 加载问题数据
samples = load_samples(INPUT_FILE)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
# 推理模式，关闭 dropout
model.eval()

# 逐条提问，收集 {问题id: 答案}
results = {}
# enumerate 在遍历时同时给出下标和元素
for i, sample in enumerate(samples):
    # 本条问题的计时起点
    q_start = time.perf_counter()
    # 生成答案
    answer = generate_answer(model, tokenizer, build_prompt(sample["question"], sample["context"]))
    results[sample["id"]] = answer
    # 本条问题耗时
    q_time = time.perf_counter() - q_start
    print(f"[{i + 1}/{len(samples)}] {sample['id']}: {answer}  ({q_time:.1f}s)")

    # 每 50 条中间落盘一次，避免中途出错丢失结果
    if (i + 1) % 50 == 0:
        json.dump(results, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 最终写入完整结果
json.dump(results, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

total_time = time.perf_counter() - START_TIME
print(f"完成，共 {len(results)} 条，结果保存至 {OUTPUT_FILE}")
print(f"总耗时 {total_time:.1f}s")
