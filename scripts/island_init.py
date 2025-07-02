from transformers import AutoConfig, AutoModelForCausalLM
import os
import torch

def modify_qwen_model_and_save(
    model_name: str = "Qwen/Qwen1.5-0.5B", # 0.6B的模型在HuggingFace上通常命名为0.5B
    target_num_layers: int = 6,
    target_num_key_value_heads: int = 2, # 对应您的4KV
    output_dir: str = "my_qwen_0.6b_modified"
):
    """
    加载 Qwen 模型的配置，修改其网络结构参数，然后创建一个新模型并保存到本地。

    Args:
        model_name (str): Hugging Face 上原始模型的名称。
        target_num_layers (int): 目标层数。
        target_num_key_value_heads (int): 目标 KV 头数量。
        output_dir (str): 保存修改后模型和配置的本地目录。
    """
    try:
        print(f"正在加载原始模型 {model_name} 的配置...")
        # 1. 加载原始模型的配置
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        print(f"原始模型配置：")
        print(f"  层数 (num_hidden_layers): {config.num_hidden_layers}")
        print(f"  Q 头 (num_attention_heads): {config.num_attention_heads}")
        print(f"  KV 头 (num_key_value_heads): {config.num_key_value_heads}")

        # 计算新的 Q 头数量 (保持 Q/KV 比例不变，或根据 8Q4KV 的要求调整)
        # 原始Q是16，KV是8，比例是2:1
        # 目标KV是4，所以目标Q应该是 4 * (16/8) = 8
        target_num_attention_heads = target_num_key_value_heads * (
            config.num_attention_heads // config.num_key_value_heads
        )

        print(f"\n正在修改配置参数...")
        # 2. 修改配置参数
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.num_attention_heads = 12  # 768/12=64
        config.num_key_value_heads = 6   # 比例可自定
        config.max_position_embeddings = 1024
        config.intermediate_size = 3072  # 768*4

        print(f"修改后模型配置：")
        print(f"  新层数 (num_hidden_layers): {config.num_hidden_layers}")
        print(f"  新 Q 头 (num_attention_heads): {config.num_attention_heads}")
        print(f"  新 KV 头 (num_key_value_heads): {config.num_key_value_heads}")

        # 3. 创建新的模型实例（不加载预训练权重，只根据新配置初始化结构）
        print(f"\n正在根据新配置创建新的模型结构...")
        # 注意：这里我们只根据配置创建模型结构，权重是随机初始化的
        new_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("新模型结构创建成功。")
        print(f"新模型参数总数: {new_model.num_parameters() / 1e6:.2f}M")

        # 4. 保存新的模型和配置到本地
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n正在将新模型和配置保存到本地目录: {output_dir}...")
        new_model.save_pretrained(output_dir)
        config.save_pretrained(output_dir) # 确保配置也单独保存

        print(f"模型和配置已成功保存到 {output_dir}。")

        print("\n您现在可以像加载Hugging Face官方模型一样加载您的新模型：")
        print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{model_name}', trust_remote_code=True)") # 分词器可以继续使用原始模型的
        print("注意：新模型由于是随机初始化的，需要进行训练才能有实际用途。")

    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查模型名称是否正确，并确保网络连接正常。")

if __name__ == "__main__":
    # 执行函数
    modify_qwen_model_and_save(
        model_name="/data/cuizhengliang/llm_hub/llm_hub/Qwen/Qwen3-0.6B",
        target_num_layers=12,
        target_num_key_value_heads=4,
        output_dir="ckpt/island"
    )
