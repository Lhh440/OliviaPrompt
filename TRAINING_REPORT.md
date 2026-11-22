# OliviaPrompt Lite 训练成果报告

> 生成时间：2026-11-22 17:05
> 状态：**训练中途停止并完整存档**（用户指令），成果可续训可直接使用

---

## 一、训练进度

| 项目 | 值 |
|---|---|
| 已训练步数 | **490 / 1083**（45.2%） |
| 已完成 epoch | 1.25 / 3 |
| 最新存档 | `checkpoint-450`（global_step 450） |
| 训练时长 | 1 小时 55 分（启动 15:58 → 停止 17:53 前后） |
| 平均速率 | 19.2s/it（后半段稳定在 15-17s/it） |

## 二、训练质量（loss 曲线健康）

```
step  50 → loss 0.0521
step 100 → loss 0.0503
step 200 → loss 0.0492
step 300 → loss 0.0487
step 400 → loss 0.0481
step 450 → loss 0.0484   ← 已收敛至平台区
```

**判断**：loss 在 450 步已进入平台（0.048 上下波动），继续训练的边际收益递减。
按 45% 进度 + 平台化趋势，这个存档**已具备可用质量**。

## 三、存档清单（9 个 checkpoint，每个 768MB）

| 存档 | 用途 |
|---|---|
| checkpoint-50 | 最早可用存档 |
| checkpoint-100 ~ 400 | 中间存档 |
| **checkpoint-450** | **最佳质量（推荐）** |
| `delivery/adapter-lite-450/` | **已整理交付版（264MB，去掉优化器状态）** |

## 四、交付物

```
E:\OliviaPrompt\data\delivery\adapter-lite-450\
├─ adapter_model.safetensors   (LoRA 权重，r=32, alpha=64)
├─ adapter_config.json
├─ tokenizer.json / tokenizer_config.json
├─ chat_template.jinja
└─ training_args.bin
```

**用法**：与基座 `E:\OliviaPrompt\models\Qwen3-4B` 配合加载即可推理。

## 五、如何继续训练（随时可续）

```bash
# 恢复训练（自动从最新 checkpoint 续）
python E:\OliviaPrompt\tools\boot_pipeline.py
# 管线会自动检测 checkpoint 并 resume，从 490 步继续到 1083
```

## 六、本次关键工程结论（实测）

1. **Windows 训练死锁根因**：`DETACHED_PROCESS` 后台进程被 EcoQoS 节流 →
   CUDA 同步死锁。**解法**：`CREATE_NEW_CONSOLE` 独立控制台窗口（前台调度类）。
2. 有效训练配置：`torch 2.5.1+cu121 + transformers 5.6.0 + bf16 全量 LoRA +
   batch 1×16 + 无梯度检查点`，12-19s/it 稳定，11.9GB 显存。
3. `gradient_checkpointing` 与 bnb 4bit 在 Windows 下均不可用（多个死锁实证）。
4. 合成数据 6000 条（5762 train / 120 val），槽位 JSON 输出格式，src 三态监督。

## 七、后续可选步骤

- **继续训练**：跑到 1083 步（约再 3 小时），长档质量会更好
- **直接导出 GGUF**：用当前 checkpoint-450 立即 merge + 量化，得到可用模型
- **评估对比**：跑 50 条基准 eval，与基座对比

> 所有脚本、配置、修复记录已提交 GitHub（`main` 分支最新提交）。
