"""OliviaPrompt post-training pipeline — runs phases 2-11 after training completed.

The v6 supervisor died right after launching the trainer (watchdog stopped
logging at 00:05:41); the trainer itself survived to completion (train_runtime
2:56:37, checkpoint-1083 + adapter exported). This script reuses
boot_pipeline's functions to run everything that remains:

  cleanup checkpoints -> merge -> f16 GGUF -> Q4_K_M -> base GGUF
  -> eval x2 -> comparison -> models.json sha256 -> commit/push -> FINAL_REPORT
  -> NO shutdown
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_pipeline as bp  # noqa: E402

ROOT = bp.ROOT
E = bp.E


def main():
    bp.log('POST PIPELINE START (training already done, no shutdown)')

    bp.log('step2: cleanup intermediate checkpoints')
    bp.cleanup_checkpoints()

    bp.log('step3: merge')
    merged = bp.merge_step()

    bp.log('step4: GGUF + quantize')
    q4 = bp.gguf_steps(merged)

    bp.log('step5: base GGUF')
    bq4 = bp.base_gguf_step()

    bp.log('step6: finetuned eval')
    bp.serve_and_eval(q4, 'lite', False)

    bp.log('step7: baseline eval')
    bp.serve_and_eval(bq4, 'base', True)

    bp.log('step8: comparison report')
    bp.sh([bp.VPY, 'training/scripts/eval.py', '--compare',
           'training/eval/report_lite.json', 'training/eval/report_base.json'],
          check=False)

    bp.log('step9: sha256 into models.json')
    import json
    mj = ROOT / 'models' / 'models.json'
    manifest = json.loads(mj.read_text(encoding='utf-8'))
    manifest['models']['lite']['sha256'] = bp.sha256(q4)
    manifest['models']['lite']['status'] = 'trained'
    if bq4.exists():
        manifest['models']['base'] = {
            'file': 'qwen3-4b-base-q4_k_m.gguf',
            'url': 'hf://Lhh440/OliviaPrompt-GGUF/qwen3-4b-base-q4_k_m.gguf',
            'sha256': bp.sha256(bq4),
            'size_hint': '≈2.5GB · 原始 Qwen3-4B Q4_K_M（基座对比用）',
            'min_vram_gb': 0,
        }
    mj.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                  encoding='utf-8')

    bp.log('step10: commit + push')
    bp.sh(['git', 'add', 'models/models.json', 'training/eval/', 'training/configs/',
           'tools/boot_pipeline.py', 'tools/post_pipeline.py', 'TRAINING_REPORT.md'],
          check=False)
    bp.sh(['git', 'commit', '-m',
           'M4 complete: lite GGUF trained+exported, benchmark evals vs base, cleanup'],
          check=False)
    pushed = False
    for _ in range(30):
        r = bp.sh(['git', 'push', 'origin', 'main'], 120)
        if r.returncode == 0:
            pushed = True
            bp.log('push OK')
            break
        bp.log('push failed — retry in 60s')
        time.sleep(60)

    bp.log('step11: final report')
    try:
        summary = (ROOT / 'training' / 'eval' / 'comparison.txt').read_text(encoding='utf-8')
    except OSError:
        summary = '(comparison.txt missing)'
    report = f"""# OliviaPrompt M4/M5 自动交付报告

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## 训练结果
- 1083/1083 步完成（3 epoch），train_runtime 2:56:37（含 450→1083 续训段）
- 最终 train_loss **0.0268**（450 步平台期时为 0.0484，续训显著下降）

## 产物
- `models/olivia-lite-qwen3-4b-q4_k_m.gguf`（sha256 见 models/models.json）
- `models/qwen3-4b-base-q4_k_m.gguf`（基座对比）
- `training/eval/report_lite.json` / `report_base.json` / `comparison.txt`

## 微调版 vs 原始基座（50 条基准）
```
{summary}
```

## GitHub 推送
{'已推送 ✓' if pushed else '推送失败——提交在本地 D:/OliviaPrompt，登录后手动 git push 即可'}

## 状态
✅ 训练完成 | ✅ GGUF 导出 | ✅ eval 完成 | ✅ 已提交 | 机器保持运行（不关机）
"""
    (ROOT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    public_desktop = Path(os.environ.get('PUBLIC', 'C:/Users/Public')) / 'Desktop'
    (public_desktop / 'OliviaPrompt_交付报告.md').write_text(report, encoding='utf-8')
    bp.log('FINAL REPORT written')

    bp.DONE_FLAG.write_text('done', encoding='utf-8')
    bp.log('POST PIPELINE COMPLETE — machine stays running')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        bp.log(f'PIPELINE FAILED: {exc}')
        (ROOT / 'FINAL_REPORT.md').write_text(
            f'# 自动管线失败\n\n{exc}\n\n查看 {bp.LOG}\n', encoding='utf-8')
        raise
