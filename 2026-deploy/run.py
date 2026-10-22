import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig, quantize_, PerGroup
import gc

# 模型权重目录（含 config.json、model.safetensors、tokenizer 文件等）
MODEL_DIR = "./Qwen3-0.6B-int8"

# 加载分词器：负责文本<->token id 转换
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# quant_config = Int8WeightOnlyConfig(granularity=PerGroup(128))
# 2. 将 torchao 配置包装进 Hugging Face 的 TorchAoConfig
# quantization_config = TorchAoConfig(quant_type=quant_config)

# 加载模型：Auto 根据 config.json 自动识别架构（Qwen3ForCausalLM）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",   # 自动分配到可用设备（GPU，无则 CPU）
    # quantization_config=quantization_config
)


# quant_config = Int8WeightOnlyConfig(granularity=PerGroup(32))
# quantize_(model, quant_config)
# gc.collect()

for p in model.parameters():
  p.data = p.data.clone()

# 对话历史列表
messages = []

while True:
    # 读取用户输入，输入 exit 退出循环
    user_input = input("\n你：")
    # 去除字符串首尾空白
    if user_input.strip() == "exit":
        break

    # 把用户消息加入历史对话中
    messages.append({"role": "user", "content": user_input})

    # 将历史对话按 Qwen3 模板转为纯文本
    # add_generation_prompt 在消息末尾附加模型回复的开始标记
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )

    # 将文本编码为输入TokenId的列表(PyTorch tensor 格式)
    model_inputs = tokenizer(text, return_tensors="pt")


    # 其余采样参数取模型默认配置
    # max_new_tokens=512 最多生成 512 个新 token
    outputs = model.generate(**model_inputs, max_new_tokens=512)

    # 解码出新增部分（去掉输入部分），跳过特殊 token
    response = tokenizer.decode(
        outputs[0][len(model_inputs["input_ids"][0]):], skip_special_tokens=True
    )
    print(response);
    # 把模型回复加入历史，用于下一轮对话的上下文
    messages.append({"role": "assistant", "content": response})
