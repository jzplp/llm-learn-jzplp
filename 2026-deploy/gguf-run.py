from llama_cpp import Llama

MODEL_PATH = "./Qwen3-0.6B-q2k-model.gguf"  # 模型文件路径

# 加载模型
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # 上下文大小
    n_threads=8,  # 指定 CPU 线程数
    verbose=False,  # 不打印详细日志
    use_mmap=False # 关闭mmap
)

# 保存对话历史
messages = []

while True:
    user_input = input("你: ").strip()
    if user_input.strip() == "exit":
        break

    # 将用户输入加入历史
    messages.append({"role": "user", "content": user_input})

    # 调用模型拿到回答
    outputs = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
    )
    response = outputs["choices"][0]["message"]["content"].strip()

    # 将助手回复加入历史
    messages.append({"role": "assistant", "content": response})
    print(f"模型: {response}\n")
