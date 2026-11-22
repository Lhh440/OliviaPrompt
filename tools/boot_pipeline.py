"""OliviaPrompt boot pipeline — runs at Windows startup (SYSTEM), fully autonomous:

    fresh training (watchdog-protected)
      -> merge LoRA -> convert f16 -> quantize Q4_K_M
      -> base model GGUF (for comparison)
      -> llama-server eval x2 (finetuned / baseline few-shot)
      -> comparison report -> sha256 into models/models.json
      -> git commit + push (best effort, credential-free under SYSTEM)
      -> FINAL_REPORT.md -> shutdown /s /t 120

Logs: E:/OliviaPrompt/tools/boot_pipeline.log
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('D:/OliviaPrompt')
E = Path('E:/OliviaPrompt')
VENV = ROOT / '.venv-train' / 'Scripts'
VPY = str(VENV / 'python.exe')
LFCLI = str(VENV / 'llamafactory-cli.exe')
T = E / 'tools' / 'llama.cpp'
M = E / 'models'
LOG = E / 'tools' / 'boot_pipeline.log'
DONE_FLAG = E / 'tools' / 'CHAIN_DONE'

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['GIT_AUTHOR_NAME'] = 'Olivia Bot'
os.environ['GIT_AUTHOR_EMAIL'] = 'olivia-bot@users.noreply.github.com'
os.environ['GIT_COMMITTER_NAME'] = 'Olivia Bot'
os.environ['GIT_COMMITTER_EMAIL'] = 'olivia-bot@users.noreply.github.com'


def log(msg):
    line = f"[boot] {msg} {time.strftime('%m-%d %H:%M:%S')}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def sh(args, timeout=None, cwd=str(ROOT), check=False):
    r = subprocess.run([str(a) for a in args], cwd=cwd, timeout=timeout,
                       capture_output=True, text=True, errors='replace')
    if r.stdout:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(r.stdout[-3000:] + '\n')
    if r.returncode != 0:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(r.stderr[-3000:] + '\n')
    if check and r.returncode != 0:
        raise RuntimeError(f'command failed: {args}')
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wait_for_training():
    """Launch fresh training with inline watchdog; returns only on clean exit."""
    tlog = E / 'tools' / 'train_lite.log'
    ckpts = E / 'data' / 'training-out' / 'lite-4b'
    for ck in ckpts.glob('checkpoint-*'):
        ck.unlink()
    if tlog.exists():
        tlog.unlink()

    log('step2: fresh training start')
    for attempt in range(12):
        logf = open(tlog, 'ab')
        proc = subprocess.Popen([str(VENV / 'python.exe'),
                                 str(VENV / 'Scripts' / 'llamafactory-cli.exe'),
                                 'train', 'training/configs/lite-4b.yaml',
                                 'model_name_or_path=E:/OliviaPrompt/models/Qwen3-4B',
                                 'dataset_dir=E:/OliviaPrompt/data/out',
                                 'output_dir=E:/OliviaPrompt/data/training-out/lite-4b'],
                                cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                                creationflags=0x00000008 | 0x00000200, close_fds=True)
        log(f'training attempt {attempt + 1} launched (pid {proc.pid})')
        last_step, last_change = None, time.time()
        while True:
            time.sleep(60)
            if proc.poll() is not None:
                rc = proc.returncode
                log(f'training process exited rc={rc}')
                if rc == 0:
                    return
                break  # crashed — outer loop relaunches (resumes from checkpoint)
            try:
                tail = tlog.read_bytes()[-4000:].decode('utf-8', 'replace')
            except OSError:
                continue
            m = re.findall(r'(\d+)/1083', tail)
            cur = m[-1] if m else None
            if cur != last_step:
                last_step, last_change = cur, time.time()
                log(f'progress step={cur}')
                continue
            stalled = time.time() - last_change > 540
            log(f'watchdog step={cur} stale={int(time.time()-last_change)}s')
            if stalled:
                log('STALL — kill and restart (resume from latest checkpoint)')
                kill_training()
                time.sleep(8)
                break
        log(f'attempt {attempt + 1} ended without clean exit')
    raise RuntimeError('training failed after 12 attempts')


def kill_training():
    sh(['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' "
        "-and $_.WorkingSetSize -gt 1GB} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"], 60)
    sh(['taskkill', '/IM', 'llamafactory-cli.exe', '/F'], 30)


def main():
    log('BOOT PIPELINE START')
    time.sleep(90)  # let the GPU driver and desktop settle after boot

    # step 0: git safety (SYSTEM global config + safe.directory)
    sh(['git', 'config', '--global', '--add', 'safe.directory', 'D:/OliviaPrompt'])
    log('step0: git safe.directory set')

    # step 1: fresh training with watchdog
    wait_for_training()

    # step 2: merge
    log('step3: merge LoRA')
    merged = E / 'data' / 'training-out' / 'merged-lite'
    merged.mkdir(parents=True, exist_ok=True)
    merge_yaml = E / 'tools' / 'merge-lite.yaml'
    merge_yaml.write_text(
        'model_name_or_path: E:/OliviaPrompt/models/Qwen3-4B\n'
        'adapter_name_or_path: E:/OliviaPrompt/data/training-out/lite-4b\n'
        'template: qwen3\nfinetuning_type: lora\n'
        f'export_dir: {merged}\nexport_size: 2\nexport_device: cpu\n'
        'export_legacy_format: false\n', encoding='utf-8')
    sh([LFCLI, 'export', str(merge_yaml)], timeout=3600, check=True)

    # step 3: convert to GGUF
    log('step4: convert merged to f16 GGUF')
    f16 = M / 'olivia-lite-qwen3-4b-f16.gguf'
    sh([VPY, str(T / 'convert_hf_to_gguf.py'), str(merged),
        '--outfile', str(f16), '--outtype', 'f16'], timeout=7200, check=True)

    # step 4: quantize
    log('step5: quantize Q4_K_M')
    q4 = M / 'olivia-lite-qwen3-4b-q4_k_m.gguf'
    sh([T / 'llama-quantize.exe', str(f16), str(q4), 'Q4_K_M'],
       timeout=3600, cwd=str(T), check=True)
    f16.unlink()

    # step 5: base model GGUF
    log('step6: base Qwen3-4B GGUF')
    bf16 = M / 'qwen3-4b-base-f16.gguf'
    sh([VPY, str(T / 'convert_hf_to_gguf.py'), str(M / 'Qwen3-4B'),
        '--outfile', str(bf16), '--outtype', 'f16'], timeout=7200, check=True)
    bq4 = M / 'qwen3-4b-base-q4_k_m.gguf'
    sh([T / 'llama-quantize.exe', str(bf16), str(bq4), 'Q4_K_M'],
       timeout=3600, cwd=str(T), check=True)
    bf16.unlink()

    # step 6+7: evals
    def serve_and_eval(gguf, label, baseline):
        log(f'serving {label} GGUF')
        server = subprocess.Popen([str(T / 'llama-server.exe'), '-m', str(gguf),
                                   '--port', '8081', '--host', '127.0.0.1',
                                   '-ngl', '999', '--parallel', '2'],
                                  cwd=str(T),
                                  stdout=open(E / 'tools' / f'llama_server_{label}.log', 'ab'),
                                  stderr=subprocess.STDOUT,
                                  creationflags=0x00000008 | 0x00000200)
        for _ in range(60):
            time.sleep(5)
            r = sh(['curl', '-s', '-m', '3', 'http://127.0.0.1:8081/health'])
            if 'true' in (r.stdout or ''):
                break
        log(f'eval {label} start')
        args = [VPY, 'training/scripts/eval.py', '--base-url',
                'http://127.0.0.1:8081/v1', '--model', label, '--label', label]
        if baseline:
            args.append('--baseline')
        sh(args, timeout=7200, check=False)
        sh(['taskkill', '/PID', str(server.pid), '/F'], 30)
        time.sleep(5)

    log('step7: finetuned eval')
    serve_and_eval(q4, 'lite', False)
    log('step8: baseline eval')
    serve_and_eval(bq4, 'base', True)

    # step 8: comparison
    log('step9: comparison report')
    sh([VPY, 'training/scripts/eval.py', '--compare',
        'training/eval/report_lite.json', 'training/eval/report_base.json'], check=False)

    # step 9: sha256 into models.json
    log('step10: fill sha256')
    mj = ROOT / 'models' / 'models.json'
    manifest = json.loads(mj.read_text(encoding='utf-8'))
    manifest['models']['lite']['sha256'] = sha256(q4)
    manifest['models']['lite']['url'] = 'hf://Lhh440/OliviaPrompt-GGUF/olivia-lite-qwen3-4b-q4_k_m.gguf'
    if bq4.exists():
        manifest['models']['base'] = {
            'file': 'qwen3-4b-base-q4_k_m.gguf',
            'url': 'hf://Lhh440/OliviaPrompt-GGUF/qwen3-4b-base-q4_k_m.gguf',
            'sha256': sha256(bq4),
            'size_hint': '≈2.5GB · 原始 Qwen3-4B Q4_K_M（基座对比用）',
            'min_vram_gb': 0,
        }
    mj.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # step 10: commit + push (best effort under SYSTEM)
    log('step11: git commit + push')
    sh(['git', 'add', 'models/models.json', 'training/eval/', 'training/configs/'], check=False)
    sh(['git', 'commit', '-m',
        'M4 executed: lite GGUF trained+exported, benchmark evals vs base, sha256 manifest',
        ], check=False)
    pushed = False
    for _ in range(30):
        r = sh(['git', 'push', 'origin', 'main'], 120)
        if 'main -> main' in (r.stdout + r.stderr) or r.returncode == 0:
            pushed = True
            log('push OK')
            break
        log('push failed — retry in 60s')
        time.sleep(60)

    # step 11: final report
    log('step12: final report')
    summary = ''
    try:
        summary = (ROOT / 'training' / 'eval' / 'comparison.txt').read_text(encoding='utf-8')
    except OSError:
        pass
    report = f"""# OliviaPrompt M4/M5 自动交付报告

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}

## 产物
- `models/olivia-lite-qwen3-4b-q4_k_m.gguf`（sha256 见 models/models.json）
- `models/qwen3-4b-base-q4_k_m.gguf`（基座对比）
- `training/eval/report_lite.json` / `report_base.json` / `comparison.txt`

## 微调版 vs 原始基座（50 条基准）
```
{summary}
```

## GitHub 推送状态
{'已推送 ✓' if pushed else '推送失败（SYSTEM 账户无凭据）——提交在本地 D:/OliviaPrompt，登录后手动 `git push` 即可'}

## 收尾
自动关机即将执行（2 分钟）。取消：`shutdown /a`
"""
    (ROOT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    (Path(os.environ.get('PUBLIC', 'C:/Users/Public/Desktop')) / 'OliviaPrompt_交付报告.md').write_text(
        report, encoding='utf-8')
    log('FINAL REPORT written')

    (E / 'tools' / 'CHAIN_DONE').write_text('done', encoding='utf-8')
    log('step13: shutdown in 120s')
    sh(['shutdown', '/s', '/t', '120'], 30)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'PIPELINE FAILED: {exc}')
        (ROOT / 'FINAL_REPORT.md').write_text(
            f'# 自动管线失败\n\n{exc}\n\n查看 {LOG}\n', encoding='utf-8')
        raise
