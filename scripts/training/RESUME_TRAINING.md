# 断点续训功能使用说明

## 概述

训练脚本现在支持断点续训功能，允许从之前保存的检查点继续训练，而不需要从头开始。这对于长时间训练任务特别有用，可以在训练中断后恢复进度。

## 功能特性

- **自动检查点验证**: 自动验证检查点路径的有效性
- **灵活的输出目录管理**: 智能处理输出目录，避免覆盖现有结果
- **完整的训练状态恢复**: 恢复模型权重、优化器状态、学习率调度器等
- **YAML配置支持**: 通过配置文件轻松控制断点续训

## 使用方法

### 1. 通过YAML配置文件

在配置文件中添加 `resume_from_checkpoint` 参数：

```yaml
# 其他训练参数...
max_steps: 40_000
save_steps: 10_000

# 断点续训配置
resume_from_checkpoint: "./output/run-0/checkpoint-10000"
```

如果不需要断点续训，设置为 `null` 或注释掉该行：

```yaml
# 开始新的训练
resume_from_checkpoint: null
```

### 2. 通过命令行参数

```bash
python scripts/training/train.py \
    --config scripts/training/configs/your-config.yaml \
    --resume-from-checkpoint "./output/run-0/checkpoint-10000"
```

## 检查点路径要求

有效的检查点目录必须包含：
- `config.json`: 模型配置文件
- `pytorch_model.bin` 或 `*.safetensors`: 模型权重文件
- 其他训练状态文件（优化器状态等）

## 输出目录行为

- **指定检查点时**: 
  - 如果使用默认输出目录 (`./output/`)，将自动使用检查点的父目录
  - 如果指定了自定义输出目录，将使用指定的目录
- **无检查点时**: 自动创建新的运行目录 (`run-0`, `run-1`, 等)

## 示例配置

参考 `scripts/training/configs/resume-example.yaml` 获取完整的配置示例。

## 注意事项

1. **检查点兼容性**: 确保检查点与当前配置兼容（模型类型、词汇表大小等）
2. **路径正确性**: 检查点路径必须是绝对路径或相对于工作目录的正确路径
3. **存储空间**: 确保有足够的存储空间继续训练
4. **训练参数**: 某些训练参数（如学习率）可能需要根据恢复的训练步数进行调整

## 兼容性问题解决

### PyTorch 版本兼容性

如果遇到以下错误：
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6
```

**解决方案：**

1. **升级 PyTorch（推荐）**：
   ```bash
   conda install pytorch>=2.6 -c pytorch
   # 或
   pip install torch>=2.6
   ```

2. **使用 safetensors 格式**：
   ```python
   # 转换现有检查点
   from safetensors.torch import save_file
   import torch

   state_dict = torch.load('pytorch_model.bin')
   save_file(state_dict, 'model.safetensors')
   ```

3. **使用兼容性检查工具**：
   ```bash
   python scripts/training/check_compatibility.py [checkpoint_path]
   ```

### 自动错误处理

训练脚本现在包含自动错误处理：
- 检测 PyTorch 版本兼容性问题
- 自动回退到新训练模式（如果检查点加载失败）
- 提供详细的错误信息和解决建议

## 错误处理

如果检查点路径无效，训练将：
1. 记录警告信息
2. 自动回退到新训练模式
3. 创建新的输出目录

## 日志信息

训练过程中会输出相关日志：
- 检查点验证结果
- 恢复训练的确认信息
- 输出目录信息
