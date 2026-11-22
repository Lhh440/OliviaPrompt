"""OliviaPrompt boot pipeline — autonomous: train -> merge -> GGUF -> evals ->
report -> commit/push -> shutdown.

Launch notes (Windows): the trainer runs inside its OWN console window
(`cmd /c start /min`), i.e. a foreground-class process. Detached background
processes get EcoQoS throttling on this machine and deadlock CUDA sync within
a few training steps (py-spy verified: threads stall inside
Qwen3RMSNorm/CheckpointFunction; a foreground pure-torch stress loop runs
111k matmuls without a single hang).

Logs: E:/OliviaPrompt/tools/boot_pipeline.log
"""

import hashlib
import json
import os
import re
import subprocess
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
BAT = E / 'tools' / 'train_attempt.bat'

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['GIT_AUTHOR_NAME'] = 'Olivia Bot'
os.environ['GIT_AUTHOR_EMAIL'] = 'olivia-bot@users.noreply.github.com'
os.environ['GIT_COMMITTER_NAME'] = 'Olivia Bot'
os.environ['GIT_COMMITTER_EMAIL'] = 'olivia-bot@users.noreply.github.com'


def log(msg):
    line = f"[boot] {msg} {time.strftime('%m-%d %H:%M:%S')}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def sh(args, timeout=None, check=False):
    r = subprocess.run([str(a) for a in args], cwd=str(ROOT), timeout=timeout,
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


def trainer_alive():
    r = sh(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 30)
    return 'python.exe' in (r.stdout or '')


def kill_training():
    sh(['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' "
        "-and $_.CommandLine -notlike '*ComfyUI*'} | ForEach-Object "
        "{ Stop-Process -Id $_.ProcessId -Force }"], 60)
    sh(['taskkill', '/IM', 'llamafactory-cli.exe', '/F'], 30)


def write_attempt_bat(resume):
    """The trainer runs in its own minimized console window = foreground-class
    scheduling. The bat writes its exit code into the log for the poll loop."""
    lines = [
        '@echo off',
        'set PYTHONUNBUFFERED=1',
        'set HF_HUB_OFFLINE=1',
        'set TOKENIZERS_PARALLELISM=false',
        'set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
        'cd /d D:\\OliviaPrompt',
        f'{VENV}\\python.exe {LFCLI} train training/configs/lite-4b.yaml '
        'model_name_or_path=E:/OliviaPrompt/models/Qwen3-4B '
        'dataset_dir=E:/OliviaPrompt/data/out '
        'output_dir=E:/OliviaPrompt/data/training-out/lite-4b'
        + (f' {resume}' if resume else '')
        + ' >> ' + str(E / 'tools' / 'train_lite.log') + ' 2>&1',
        'echo TRAIN_EXIT_CODE %errorlevel% >> ' + str(E / 'tools' / 'train_lite.log'),
    ]
    BAT.write_text('\r\n'.join(lines) + '\r\n', encoding='utf-8')


def wait_for_training():
    """Launch fresh training in its own console; poll + watchdog.

    Completion: `train_runtime` in the log (HF prints it at the very end).
    Crash: the trainer console disappears without it.
    Stall: no new step number for 540s while the process is alive.
    On stall/crash: kill every python, relaunch, resume from the latest
    checkpoint (50-step granularity, so a stall loses at most 50 steps).
    """
    tlog = E / 'tools' / 'train_lite.log'
    if tlog.exists():
        tlog.unlink()

    for attempt in range(12):
        ckpts = E / 'data' / 'training-out' / 'lite-4b'
        resume = 'resume_from_checkpoint=true' if any(ckpts.glob('checkpoint-*')) else ''
        log(f'attempt {attempt + 1} launch (resume={bool(resume)})')
        write_attempt_bat(resume)
        args = [str(VENV / 'python.exe'), str(LFCLI), 'train',
                'training/configs/lite-4b.yaml',
                'model_name_or_path=E:/OliviaPrompt/models/Qwen3-4B',
                'dataset_dir=E:/OliviaPrompt/data/out',
                'output_dir=E:/OliviaPrompt/data/training-out/lite-4b']
        if resume:
            args.append(resume)
        logf = open(tlog, 'ab')
        # CREATE_NEW_CONSOLE: the trainer gets its own console window
        # (foreground-class scheduling, NOT EcoQoS-throttled like
        # DETACHED_PROCESS children) while stdout still lands in tlog.
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=logf,
                                stderr=subprocess.STDOUT,
                                creationflags=0x00000010, close_fds=True)
        last_step, last_change = None, time.time()
        while True:
            time.sleep(60)
            alive = trainer_alive()
            try:
                tail = tlog.read_bytes()[-4000:].decode('utf-8', 'replace')
            except OSError:
                tail = ''
            if 'train_runtime' in tail:
                log('training completed (train_runtime found) — SUCCESS')
                return
            if not alive:
                log('trainer console gone without train_runtime — crash')
                break
            m = re.findall(r'(\d+)/1083', tail)
            cur = m[-1] if m else None
            if cur != last_step:
                last_step, last_change = cur, time.time()
                log(f'progress step={cur}')
                continue
            stalled = time.time() - last_change > 540
            log(f'watchdog step={cur} stale={int(time.time() - last_change)}s')
            if stalled:
                log(f'STALL at step {cur} — kill everything and resume')
                kill_training()
                time.sleep(8)
                break
        log(f'attempt {attempt + 1} ended without clean exit')
    raise RuntimeError('training failed after 12 attempts')


def merge_step():
    log('merge LoRA')
    merged = E / 'data' / 'training-out' / 'merged-lite'
    merged.mkdir(parents=True, exist_ok=True)
    merge_yaml = E / 'tools' / 'merge-lite.yaml'
    merge_yaml.write_text(
        'model_name_or_path: E:/OliviaPrompt/models/Qwen3-4B\n'
        'adapter_name_or_path: E:/OliviaPrompt/data/training-out/lite-4b\n'
        'template: qwen3\nfinetuning_type: lora\n'
        f'export_dir: {merged}\nexport_size: 2\nexport_device: cpu\n'
        'export_legacy_format: false\n', encoding='utf-8')
    sh([LFCLI, 'export', str(merge_yaml)], timeout=7200, check=True)
    if not any(merged.glob('*.safetensors')):
        raise RuntimeError('merge produced no safetensors')
    return merged


def gguf_steps(merged):
    log('convert merged -> f16 GGUF')
    f16 = M / 'olivia-lite-qwen3-4b-f16.gguf'
    sh([VPY, str(T / 'convert_hf_to_gguf.py'), str(merged),
        '--outfile', str(f16), '--outtype', 'f16'], timeout=14400, check=True)
    log('quantize Q4_K_M')
    q4 = M / 'olivia-lite-qwen3-4b-q4_k_m.gguf'
    sh([T / 'llama-quantize.exe', str(f16), str(q4), 'Q4_K_M'],
       timeout=7200, cwd=str(T), check=True)
    f16.unlink()
    return q4


def base_gguf_step():
    log('base Qwen3-4B GGUF (for comparison eval)')
    bf16 = M / 'qwen3-4b-base-f16.gguf'
    sh([VPY, str(T / 'convert_hf_to_gguf.py'), str(M / 'Qwen3-4B'),
        '--outfile', str(bf16), '--outtype', 'f16'], timeout=14400, check=True)
    bq4 = M / 'qwen3-4b-base-q4_k_m.gguf'
    sh([T / 'llama-quantize.exe', str(bf16), str(bq4), 'Q4_K_M'],
       timeout=7200, cwd=str(T), check=True)
    bf16.unlink()
    return bq4


def serve_and_eval(gguf, label, baseline):
    log(f'serving {label} GGUF')
    server = subprocess.Popen([str(T / 'llama-server.exe'), '-m', str(gguf),
                               '--port', '8081', '--host', '127.0.0.1',
                               '-ngl', '999', '--parallel', '2'],
                              cwd=str(T),
                              stdout=open(E / 'tools' / f'llama_server_{label}.log', 'ab'),
                              stderr=subprocess.STDOUT,
                              creationflags=0x00000008 | 0x00000200)
    up = False
    for _ in range(60):
        time.sleep(5)
        r = sh(['curl', '-s', '-m', '3', 'http://127.0.0.1:8081/health'], 15)
        if 'true' in (r.stdout or ''):
            up = True
            break
    if not up:
        raise RuntimeError(f'llama-server ({label}) did not become healthy')
    log(f'eval {label} start')
    args = [VPY, 'training/scripts/eval.py', '--base-url',
            'http://127.0.0.1:8081/v1', '--model', label, '--label', label]
    if baseline:
        args.append('--baseline')
    sh(args, timeout=14400, check=False)
    sh(['taskkill', '/PID', str(server.pid), '/F'], 30)
    time.sleep(5)


def main():
    log('BOOT PIPELINE START')
    time.sleep(60)

    log('step0: git safe.directory')
    sh(['git', 'config', '--global', '--add', 'safe.directory', 'D:/OliviaPrompt'])

    log('step1: fresh training (watchdog protected)')
    wait_for_training()

    log('step2: merge')
    merged = merge_step()
    log('step3: GGUF + quantize')
    q4 = gguf_steps(merged)
    log('step4: base GGUF')
    bq4 = base_gguf_step()

    log('step5: finetuned eval')
    serve_and_eval(q4, 'lite', False)
    log('step6: baseline eval')
    serve_and_eval(bq4, 'base', True)

    log('step7: comparison report')
    sh([VPY, 'training/scripts/eval.py', '--compare',
        'training/eval/report_lite.json', 'training/eval/report_base.json'], check=False)

    log('step8: sha256 into models.json')
    mj = ROOT / 'models' / 'models.json'
    manifest = json.loads(mj.read_text(encoding='utf-8'))
    manifest['models']['lite']['sha256'] = sha256(q4)
    if bq4.exists():
        manifest['models']['base'] = {
            'file': 'qwen3-4b-base-q4_k_m.gguf',
            'url': 'hf://Lhh440/OliviaPrompt-GGUF/qwen3-4b-base-q4_k_m.gguf',
            'sha256': sha256(bq4),
            'size_hint': '≈2.5GB · 原始 Qwen3-4B Q4_K_M（基座对比用）',
            'min_vram_gb': 0,
        }
    mj.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    log('step9: commit + push')
    sh(['git', 'add', 'models/models.json', 'training/eval/', 'training/configs/'], check=False)
    sh(['git', 'commit', '-m',
        'M4 executed: lite GGUF trained+exported, benchmark evals vs base, sha256 manifest'],
       check=False)
    pushed = False
    for _ in range(30):
        r = sh(['git', 'push', 'origin', 'main'], 120)
        if r.returncode == 0:
            pushed = True
            log('push OK')
            break
        log('push failed — retry in 60s')
        time.sleep(60)

    log('step10: final report')
    try:
        summary = (ROOT / 'training' / 'eval' / 'comparison.txt').read_text(encoding='utf-8')
    except OSError:
        summary = '(comparison.txt missing)'
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

## GitHub 推送
{'已推送 ✓' if pushed else '推送失败——提交在本地 D:/OliviaPrompt，登录后手动 git push 即可'}

## 收尾
自动关机即将执行（120 秒）。取消：shutdown /a
"""
    (ROOT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    public_desktop = Path(os.environ.get('PUBLIC', 'C:/Users/Public')) / 'Desktop'
    (public_desktop / 'OliviaPrompt_交付报告.md').write_text(report, encoding='utf-8')
    log('FINAL REPORT written')

    DONE_FLAG.write_text('done', encoding='utf-8')
    log('step11: shutdown in 120s')
    sh(['shutdown', '/s', '/t', '120'], 30)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log(f'PIPELINE FAILED: {exc}')
        (ROOT / 'FINAL_REPORT.md').write_text(
            f'# 自动管线失败\n\n{exc}\n\n查看 {LOG}\n', encoding='utf-8')
        raise
