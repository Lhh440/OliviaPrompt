# OliviaPrompt M4/M5 自动交付报告

生成时间：2026-11-28 06:01:24

## 训练结果
- 1083/1083 步完成（3 epoch），train_runtime 2:56:37（含 450→1083 续训段）
- 最终 train_loss **0.0268**（450 步平台期时为 0.0484，续训显著下降）

## 产物
- `models/olivia-lite-qwen3-4b-q4_k_m.gguf`（2.5GB，sha256 见 models/models.json，status=trained）
- `models/qwen3-4b-base-q4_k_m.gguf`（2.5GB，基座对照）
- `training/eval/report_lite.json` / `report_base.json` / `comparison.txt`
- LoRA adapter：`E:\OliviaPrompt\data\training-out\lite-4b`

## 微调版 vs 原始基座（50 条基准）
```
metric                               lite         base   delta
------------------------------------------------------------------
json_ok_rate                       100.0%        60.0%   = -40.0%
slot_acc                            98.0%        52.2%   = -45.7%
weighted_acc                        97.8%        55.2%   = -42.6%
consistency_error_rate               0.0%         6.7%   = +6.7%
```

## GitHub 推送
提交在本地 D:/OliviaPrompt（main，最新 b468684）。代理 127.0.0.1:7890 恢复后执行：
```bash
git push origin main
```
推送监视器已挂（后台自动重试）。

## 状态
✅ 训练完成 | ✅ GGUF 导出 | ✅ eval 完成 | ✅ 已提交 | 机器保持运行（不关机）
