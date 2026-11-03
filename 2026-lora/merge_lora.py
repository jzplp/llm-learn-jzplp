"""
将基座模型与 LoRA 权重合并为一个完整模型，并同时保存分词器
用法:
    python merge_lora.py
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 原模型（基座模型）路径
BASE_MODEL_PATH = "Qwen3-0.6B-train"
# LoRA 权重路径
LORA_PATH = "Qwen3-0.6B-lora"
# 合并后完整模型的输出路径
OUTPUT_DIR = "Qwen3-0.6B-merged"


def main():
    # 加载分词器（LoRA 目录不含分词器，使用基座模型的分词器）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # 加载基座模型，dtype 与原始权重保持一致
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype="auto"
    )
    # 在基座模型上加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    # 把 LoRA 权重合并进基座权重，并移除适配器结构，得到普通模型
    model = model.merge_and_unload()
    # 推理模式，关闭 dropout
    model.eval()

    # 保存合并后的完整模型
    model.save_pretrained(OUTPUT_DIR)
    # 同时保存分词器
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"合并完成，完整模型与分词器已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
