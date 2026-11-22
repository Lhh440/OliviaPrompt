# OliviaPrompt M4/M5 自动交付说明

本机将在 **60 秒后重启**（训练环境修复必需——多次训练死锁确诊为 GPU 驱动状态被污染，只有重启能清除）。

## 重启后会发生什么（全自动，无需任何操作）

登录 Windows 后，`olivia_pipeline.bat`（启动文件夹）会自动运行完整管线：

1. **全新训练** lite-4B（1083 步，约 3.5-4.5 小时，内置看门狗：死锁自动恢复，最多 50 步损失）
2. merge LoRA → f16 GGUF → **Q4_K_M 量化**
3. 基座 Qwen3-4B 转 GGUF（用于对比）
4. llama-server 起服务 → **50 条基准 eval**（微调版 + 基座 few-shot 对比）
5. sha256 写入 `models/models.json`
6. git 提交 + 推送
7. 生成 `FINAL_REPORT.md`（仓库根目录 + 公共桌面）
8. **自动关机（120 秒缓冲）**

## 监控与产物

- 管线日志：`E:\OliviaPrompt\tools\boot_pipeline.log`
- 训练日志：`E:\OliviaPrompt\tools\train_lite.log`
- eval 报告：`D:\OliviaPrompt\training\eval\report_lite.json` / `report_base.json` / `comparison.txt`

## 提醒

- 重启后 3.5 小时内**不要玩大型游戏**（显存满）；训练结束后随便玩
- 若最终自动关机时你正在游戏：`shutdown /a` 取消当次关机（产物不受影响）
- 若推送失败（凭据问题）：提交已在本地，手动 `git push origin main` 即可
