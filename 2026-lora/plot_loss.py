r"""根据训练日志 trainer_state.json 绘制训练集 loss 与验证集 loss 曲线。

    # 画曲线，图片默认输出到 test1\Qwen3-0.6B-lora\loss_curve.png
    python plot_loss.py test1\Qwen3-0.6B-lora\trainer_state.json

    # 指定图片输出路径
    python plot_loss.py test1\Qwen3-0.6B-lora\trainer_state.json -o test1\loss.png

    # 指定标题前缀，标题变为 "test1 Training / Validation Loss"
    python plot_loss.py test1\Qwen3-0.6B-lora\trainer_state.json -t test1
"""

import argparse
import json

import matplotlib.pyplot as plt

# 解析命令行参数：必填的 trainer_state.json 路径，以及可选的图片输出路径
def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 trainer_state.json 绘制训练集 loss 与验证集 loss 曲线"
    )
    # 位置参数：训练日志文件路径，必须提供
    parser.add_argument(
        "trainer_state",
        help="trainer_state.json 路径",
    )
    # 可选参数：不传时按输入路径自动生成 loss_curve.png
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="图片输出路径，默认与输入同名的 loss_curve.png",
    )
    # 可选参数：标题前缀，例如传入 test1，最终标题为 "test1 Training / Validation Loss"
    parser.add_argument(
        "-t",
        "--title-prefix",
        default="",
        help="图表标题前缀，例如 test1",
    )
    return parser.parse_args()

# 从 log_history 中拆出训练 loss 和验证 loss 两条曲线
def load_losses(path):
    # trainer_state.json 的 log_history 是一个列表，训练记录带 loss 字段，
    # 验证记录带 eval_loss 字段，按字段区分后各取 step 和 loss
    log_history = json.load(open(path, encoding="utf-8"))["log_history"]
    # 训练集 loss
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    # 验证集 loss
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    return train_steps, train_loss, eval_steps, eval_loss

# 未指定 -o 时，由输入路径推导输出图片路径
# 例：test1\Qwen3-0.6B-lora\trainer_state.json -> test1\Qwen3-0.6B-lora\loss_curve.png
def default_output_path(input_path):
    return input_path.replace(".json", "").replace("trainer_state", "loss_curve") + ".png"

def main():
    # 解析命令行参数
    args = parse_args()
    # 确定图片保存路径：优先用 -o，否则自动生成
    output = args.output or default_output_path(args.trainer_state)

    # 读取训练日志，拿到两条曲线的数据
    train_steps, train_loss, eval_steps, eval_loss = load_losses(args.trainer_state)

    # 画图：训练 loss 用连续折线，验证 loss 用带数据点的折线以便区分
    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_loss, label="train loss", linewidth=1.5)
    plt.plot(eval_steps, eval_loss, marker="o", markersize=4, label="eval loss")

    # 设置坐标轴标题、网格和图例
    plt.xlabel("step")
    plt.ylabel("loss")
    # 图表标题，拼上可选的前缀
    title = f"{args.title_prefix} Training / Validation Loss".strip()
    plt.title(title)
    # 绘制网格，alpha 控制网格线透明度
    plt.grid(alpha=0.3)
    # 显示图例（根据 label 字段生成）
    plt.legend()
    # 自动调整边距，避免标签被裁剪
    plt.tight_layout()
    # 保存图片
    plt.savefig(output, dpi=150)
    print(f"图片已保存至 {output}")

if __name__ == "__main__":
    main()
