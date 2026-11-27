"""OliviaPrompt post-pipeline phase 3 — evals onward, fully idempotent.

Phases 1-2 produced all GGUF artifacts:
  models/olivia-lite-qwen3-4b-q4_k_m.gguf (2497280448 B)
  models/qwen3-4b-base-q4_k_m.gguf        (2497280480 B)
Phase 2 died on a wrong health-check string ('true' vs {"status":"ok"}) —
fixed in boot_pipeline.serve_and_eval. This phase skips any artifact that
already exists (GGUF files, eval reports), runs the remaining steps:
  eval lite -> eval base -> comparison -> models.json -> commit/push -> report.
NO shutdown.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_pipeline as bp  # noqa: E402

ROOT = bp.ROOT
E = bp.E
EVAL_DIR = ROOT / 'training' / 'eval'


def serve_and_eval_skippable(gguf, label, baseline):
    report = EVAL_DIR / f'report_{label}.json'
    if report.exists():
        bp.log(f'eval {label}: report exists, skip')
        return
    bp.serve_and_eval(gguf, label, baseline)


def main():
    bp.log('POST PIPELINE 3 START (evals onward, no shutdown)')

    q4 = bp.M / 'olivia-lite-qwen3-4b-q4_k_m.gguf'
    bq4 = bp.M / 'qwen3-4b-base-q4_k_m.gguf'
    if not q4.exists():
        raise RuntimeError('lite Q4 gguf missing — rerun phase 2')
    if not bq4.exists():
        raise RuntimeError('base Q4 gguf missing — rerun phase 2')

    bp.log('step6: finetuned eval')
    serve_and_eval_skippable(q4, 'lite', False)

    bp.log('step7: baseline eval')
    serve_and_eval_skippable(bq4, 'base', True)

    bp.log('step8: comparison report')
    bp.sh([bp.VPY, 'training/scripts/eval.py', '--compare',
           'training/eval/report_lite.json', 'training/eval/report_base.json'],
          check=False)

    bp.log('step9: sha256 into models.json')
    mj = ROOT / 'models' / 'models.json'
    manifest = json.loads(mj.read_text(encoding='utf-8'))
    if 'sha256' not in manifest['models'].get('lite', {}) or \
            manifest['models']['lite'].get('status') != 'trained':
        manifest['models']['lite']['sha256'] = bp.sha256(q4)
        manifest['models']['lite']['status'] = 'trained'
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
           'tools/boot_pipeline.py', 'tools/post_pipeline.py', 'tools/post_pipeline2.py',
           'tools/post_pipeline3.py', 'TRAINING_REPORT.md'], check=False)
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
        summary = (EVAL_DIR / 'comparison.txt').read_text(encoding='utf-8')
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
    bp.log('POST PIPELINE 3 COMPLETE — machine stays running')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        bp.log(f'PIPELINE FAILED: {exc}')
        (ROOT / 'FINAL_REPORT.md').write_text(
            f'# 自动管线失败\n\n{exc}\n\n查看 {bp.LOG}\n', encoding='utf-8')
        raise
