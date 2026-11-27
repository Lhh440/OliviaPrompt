# OliviaPrompt Lite 训练成果报告（最终版）

> 生成时间：2026-11-28 05:45
> 状态：**训练 + 导出 + 评测全部完成**

---

## 一、训练结果

| 项目 | 值 |
|---|---|
| 总步数 | **1083 / 1083**（3 epoch，Qwen3-4B + LoRA r32 α64） |
| 总用时 | 2h56m（450→1083 续训段）+ 前期 1h55m |
| 最终 loss | **0.0268**（450 步平台期 0.0484 → 续训下降 45%） |
| 平均速率 | 15-16s/it（RTX 4070 12GB，bf16 全量 LoRA，batch 1×16） |
| 数据 | 6000 条合成 SFT（5762 train / 120 val），三档语域，src 三态监督 |

## 二、50 条基准评测：微调版 vs 原始基座（few-shot）

```
metric                               lite         base
json_ok_rate                       100.0%        60.0%
slot_acc                            98.0%        52.2%
weighted_acc                        97.8%        55.2%
consistency_error_rate               0.0%         6.7%
```

微调模型槽位准确率 98%、输出 100% 合法 JSON、零自洽错误 ——
产品承诺的「模型只填槽位，格式由代码保证」在 4B 上成立。

## 三、交付物

| 文件 | 说明 |
|---|---|
| `models/olivia-lite-qwen3-4b-q4_k_m.gguf` | **成品模型（2.5GB）**，sha256 见 models/models.json |
| `models/qwen3-4b-base-q4_k_m.gguf` | 基座对照（2.5GB） |
| `training/eval/report_lite.json` / `report_base.json` | 逐 case 评测明细 |
| `training/eval/comparison.txt` | 对比汇总 |
| `E:\OliviaPrompt\data\training-out\lite-4b\` | LoRA adapter（可续训） |

## 四、当夜工程记录（全部已修复并提交）

1. v6 监督进程在启动训练器后自毁（00:05），训练器独立存活跑完 —— 事后用
   `post_pipeline.py / post_pipeline2.py / post_pipeline3.py` 三段式接管线
2. llama.cpp 二进制包缺 Python `conversion` 包 → 从匹配 commit（e412c6f46）
   源码补齐（gh-proxy.com 镜像下载）
3. pip `gguf` 过旧（缺 ATTN_QKV_GATE）→ 用源码自带 gguf-py 覆盖安装
4. `sh()` 不接受 cwd（量化必炸）→ 修复
5. llama-server 健康检查等旧版响应格式 → 兼容 `{"status":"ok"}`

## 五、GitHub

提交在本地 `main`（推送受代理影响，恢复后执行 `git push origin main` 即可）。

> 模型权重不入仓库；分发走 `scripts/start.bat`（按 models.json 从 HF 下载）。
