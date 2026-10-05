"""Portable, bounded deployment entry point. No remote access or automatic uploads."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SPLIT = Path('data/official_v2_20260831b/splits.json')
PROFILE = Path('configs/cloud/stock4090.yaml')
GATE = Path('runs/diagnostic_v2/R002_stock_bce_fp32_lr1e4/learning_gate.json')
DATA_YAMLS = {
    'data/official_yolo_hybrid_v2/steel_defect.yaml': 'data/official_yolo_hybrid_v2',
    'data/diagnostic_v2/diagnostic.yaml': 'data/diagnostic_v2',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def relocate():
    import yaml
    changes = []
    for name, folder in DATA_YAMLS.items():
        path = ROOT / name
        spec = yaml.safe_load(path.read_text(encoding='utf-8'))
        target = (ROOT / folder).as_posix()
        if spec.get('path') != target:
            changes.append({'file': name, 'old': spec.get('path'), 'new': target})
            spec['path'] = target
            path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    write(ROOT / 'portability.json', {'root': str(ROOT), 'changes': changes})
    print(f'Dataset paths ready: {ROOT}')


def verify(full=False):
    import yaml
    manifest = read(ROOT / 'PACKAGE_MANIFEST.json')
    errors = []
    for index, item in enumerate(manifest['files'], 1):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path = ROOT / relative
        if not path.is_file():
            errors.append(f'Missing: {relative}')
        elif item['path'] in DATA_YAMLS:
            actual = yaml.safe_load(path.read_text(encoding='utf-8'))
            expected = item['yaml_spec'].copy()
            expected['path'] = actual.get('path')
            permitted = {(ROOT / DATA_YAMLS[item['path']]).as_posix(), item['yaml_spec']['path']}
            if actual != expected or actual.get('path') not in permitted:
                errors.append(f'Unexpected dataset YAML change: {relative}')
        elif path.stat().st_size != item['size'] or (full and sha(path) != item['sha256']):
            errors.append(f'Content mismatch: {relative}')
        if full and index % 2000 == 0:
            print(f'Verified {index}/{len(manifest["files"])}', flush=True)
    if errors:
        raise ValueError('\n'.join(errors[:30]))
    print(f'PASS: {len(manifest["files"])} files; ' + ('SHA256 checked' if full else 'sizes checked; use --full for SHA256'))


def preflight():
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('Python 3.11 or 3.12 required')
    import torch
    import torchvision
    import cv2
    import yaml
    from steel_defect.training_audit import cuda_witness
    expected = {'torch': '2.5.1', 'torchvision': '0.20.1', 'ultralytics': '8.3.169',
                'numpy': '1.26.4', 'opencv-python': '4.11.0.86', 'Pillow': '10.4.0', 'PyYAML': '6.0.1'}
    installed = {name: importlib.metadata.version(name) for name in expected}
    if any(installed[k].split('+')[0] != v for k, v in expected.items()):
        raise RuntimeError(f'Environment differs from tested core: {installed}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU training. Check platform driver / CUDA wheel.')
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if total - free > 500 * 1024**2:
        raise RuntimeError('Selected GPU is not idle (>500 MiB used). Stop other jobs or set CUDA_VISIBLE_DEVICES.')
    if total < 20 * 1024**3:
        raise RuntimeError('4090 profile expects at least 20 GiB visible VRAM')
    disk_free = shutil.disk_usage(ROOT).free
    if disk_free < 25 * 1024**3:
        raise RuntimeError('Keep at least 25 GiB free after extraction for checkpoints, caches and logs.')
    witness = cuda_witness(42)
    keep = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device='cuda'),
                               torch.tensor([.9], device='cuda'), .5)
    if keep.cpu().tolist() != [0]:
        raise RuntimeError('torchvision CUDA NMS witness failed')
    for name, folder in DATA_YAMLS.items():
        spec = yaml.safe_load(Path(name).read_text(encoding='utf-8'))
        if Path(spec['path']).resolve() != (ROOT / folder).resolve():
            raise RuntimeError('Run python cloud.py relocate first')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'versions': installed, 'gpu': torch.cuda.get_device_name(0),
              'cuda': torch.version.cuda, 'vram_gib': total / 1024**3,
              'disk_free_gib': disk_free / 1024**3, 'cuda_witness': witness,
              'spec_sha256': {p: sha(p) for p in ['requirements.txt', 'setup.sh', str(PROFILE)]}}
    write('.aris/compute/preflight.json', report)
    print(json.dumps(report, indent=2))


def recipe_hash(config):
    # Batch, workers, model, optimizer, augmentation etc remain part of comparison.
    stable = {k: v for k, v in config.items() if k not in {'epochs', 'name', 'project'}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def evaluation_command(weights, output, split='dev', freeze=None, batch=4):
    command = ['scripts/eval.py', '--weights', str(weights), '--source', 'data/official/train',
               '--split-manifest', str(SPLIT), '--split', split, '--output', str(output),
               '--tile-size', '1024', '--imgsz', '1024', '--overlap', '0.25', '--batch', str(batch),
               '--conf', '0.001', '--iou', '0.55', '--local-iou', '0.70',
               '--max-det', '2000', '--global-pass', '--no-half', '--no-tune-thresholds']
    if freeze:
        command += ['--freeze-manifest', str(freeze)]
    return command


def execute(command, logfile):
    print('RUN:', subprocess.list2cmdline([sys.executable, '-u', *command]), flush=True)
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with Path(logfile).open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f'Command failed ({process.returncode}); see {logfile}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise


def choose_epochs(stage, count):
    return [1, 3, 5] if stage == 'baseline' else sorted(set(range(5, count + 1, 5)) | {count})


def assert_training_complete(root, count):
    completion = read(root / 'completion.json')
    if completion.get('status') != 'complete':
        raise ValueError('Training did not finish successfully')
    with (root / 'results.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != count or [int(float(r['epoch'])) for r in rows] != list(range(1, count + 1)):
        raise ValueError('Epoch history is incomplete')
    if any(not math.isfinite(float(v)) for r in rows for k, v in r.items() if k.startswith(('train/', 'val/'))):
        raise ValueError('Non-finite training/validation history; do not proceed')


def finish_evaluations(root, control, config, state):
    """Recoverable original dev evaluation; never silently repeats a valid result."""
    state['evaluations'] = []
    for epoch in choose_epochs(state['stage'], config['epochs']):
        state['current_original_eval_epoch'] = epoch
        write(control / 'state.json', state)
        weights = root / 'weights' / f'epoch{epoch-1}.pt'
        output = root / f'original_dev_epoch{epoch}'
        metrics_path = output / 'metrics.json'
        if not metrics_path.is_file():
            execute(evaluation_command(weights, output, batch=min(4, config['batch'])),
                    control / f'dev_epoch{epoch}.log')
        report = read(metrics_path)
        if (not report['scope_complete'] or report['evaluated_images'] != 459 or report['split'] != 'dev'
                or report['hashes']['weights'] != sha(weights) or report['hashes']['split_manifest'] != sha(SPLIT)):
            raise ValueError(f'Incomplete or changed dev evaluation: {metrics_path}')
        if not all(math.isfinite(report['summary'][k]) for k in ('map50_macro', 'recall_macro')):
            raise ValueError('Non-finite original dev metrics')
        state['evaluations'].append({'epoch': epoch, 'summary': report['summary'],
                                     'metrics': str(metrics_path), 'checkpoint': str(weights)})
        write(control / 'state.json', state)
    best = max(state['evaluations'], key=lambda r: (r['summary']['map50_macro'], r['summary']['recall_macro']))
    selected = root / 'weights' / 'best_original.pt'
    shutil.copy2(best['checkpoint'], selected)
    write(root / 'original_selection.json', {**best, 'checkpoint': str(selected),
          'weights_sha256': sha(selected), 'split_sha256': sha(SPLIT),
          'recipe_sha256': recipe_hash(config), 'stage': state['stage'],
          'criterion': 'dev macro AP50, then macro recall; NOT official competition total score'})
    state['status'] = 'complete_review_required'
    write(control / 'state.json', state)


def evaluate_run(args):
    """After manual exact training resume, finish the same queue's evaluation/selection."""
    import yaml
    from steel_defect.training_audit import dataset_signature
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('Invalid run name')
    project = Path(yaml.safe_load(PROFILE.read_text(encoding='utf-8'))['train']['project'])
    control = project / '_queues' / args.name
    state = read(control / 'state.json')
    config = yaml.safe_load((control / 'resolved_train.yaml').read_text(encoding='utf-8'))['train']
    root = project / args.name
    if state['stage'] not in ('baseline', 'long') or state['recipe_sha256'] != recipe_hash(config):
        raise ValueError('Run stage/recipe changed')
    if state['split_sha256'] != sha(SPLIT):
        raise ValueError('Frozen split changed')
    assert_training_complete(root, config['epochs'])
    if (read(root / 'completion.json')['dataset_signature'] != dataset_signature(config['data'])):
        raise ValueError('Training data changed since completion')
    if state['status'] == 'complete_review_required' and (root / 'original_selection.json').is_file():
        print(f'Already complete: {root / "original_selection.json"}')
        return
    verify(full=True)
    preflight()
    finish_evaluations(root, control, config, state)


def run(args):
    import yaml
    config = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))['train']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('Name must use only letters, digits, underscore or hyphen')
    count = {'smoke': 2, 'baseline': 5, 'long': args.epochs}[args.stage]
    if count not in range(5, 121) and args.stage == 'long':
        raise ValueError('Long training must be 5..120 epochs')
    config.update(name=args.name, epochs=count)
    if args.batch is not None:
        if args.batch not in (2, 4, 8):
            raise ValueError('Explicit batch must be 2, 4 or 8')
        config['batch'] = args.batch
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError('workers must be nonnegative')
        config['workers'] = args.workers
    if args.stage == 'smoke':
        config['data'] = 'data/diagnostic_v2/diagnostic.yaml'
    root = Path(config['project']) / args.name
    control = Path(config['project']) / '_queues' / args.name
    profile_path = control / 'resolved_train.yaml'
    evaluations = [] if args.stage == 'smoke' else choose_epochs(args.stage, count)
    commands = [['scripts/train.py', '--config', str(profile_path)]]
    commands += [evaluation_command(root / 'weights' / f'epoch{e-1}.pt', root / f'original_dev_epoch{e}',
                                   batch=min(4, config['batch'])) for e in evaluations]
    if args.dry_run:
        print(json.dumps({'resolved_config': config, 'commands': commands,
                          'note': 'Read-only plan; no GPU, data or gate verification performed'}, indent=2))
        return
    if args.stage == 'long':
        if not args.after:
            raise ValueError('Long training requires --after <baseline original_selection.json>')
        previous = read(args.after)
        if previous.get('stage') != 'baseline' or previous.get('recipe_sha256') != recipe_hash(config):
            raise ValueError('Complete a baseline using the same recipe first')
        if previous['split_sha256'] != sha(SPLIT) or previous['weights_sha256'] != sha(previous['checkpoint']):
            raise ValueError('Baseline evidence changed')
    if root.exists() or control.exists():
        raise ValueError(f'Run exists: {root}. Choose a new name; do not overwrite historical runs.')
    verify()
    preflight()
    if args.stage != 'smoke':
        from scripts.run_v2_baselines import check_gate
        check_gate(GATE)
    control.mkdir(parents=True)
    profile_path.write_text(yaml.safe_dump({'train': config}, sort_keys=False), encoding='utf-8')
    state = {'status': 'running', 'stage': args.stage, 'root': str(root), 'epochs': count,
             'pid': os.getpid(), 'config': str(profile_path), 'recipe_sha256': recipe_hash(config),
             'split_sha256': sha(SPLIT), 'evaluations': []}
    write(control / 'state.json', state)
    try:
        execute(commands[0], control / 'train.log')
        assert_training_complete(root, count)
        if args.stage == 'smoke':
            state['status'] = 'smoke_complete_not_generalization'
        else:
            finish_evaluations(root, control, config, state)
        write(control / 'state.json', state)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write(control / 'state.json', state)
        raise


def finalize(args):
    selection = read(args.selection)
    weights = Path(selection['checkpoint'])
    if selection['split_sha256'] != sha(SPLIT) or selection['weights_sha256'] != sha(weights):
        raise ValueError('Selected model/split changed')
    output = Path(args.output)
    resume_inference = getattr(args, 'resume_inference', False)
    batch = getattr(args, 'batch', 4)
    if output.exists() and not resume_inference:
        raise ValueError('Final output already exists; refuse accidental repeat evaluation')
    ledger_path = ROOT / 'runs/cloud4090/_final_holdout_consumed.json'
    if ledger_path.exists() and not resume_inference:
        raise ValueError('Final holdout already reserved/consumed. Changing output does not permit another evaluation.')
    policy = dict(tile_size=1024, overlap=.25, imgsz=1024, conf=.001, iou=.55, local_iou=.70,
                  operating_threshold=.05, half=False, tta=False, deblur=False, deblur_threshold=85.,
                  clahe=False, global_pass=True, edge_margin=12, edge_penalty=1., merge='nms',
                  max_det=2000, beta=2., class_thresholds={})
    freeze = output / 'freeze.json'
    commands = [evaluation_command(weights, output / 'final_holdout', split='final', freeze=freeze, batch=batch),
                ['scripts/infer.py', '--weights', str(weights), '--source', 'data/official/test',
                 '--output', str(output / 'predictions_test.json'), '--tile-size', '1024', '--imgsz', '1024',
                 '--overlap', '0.25', '--batch', str(batch), '--conf', '0.001', '--iou', '0.55', '--local-iou', '0.70',
                 '--max-det', '2000', '--global-pass', '--no-half']]
    if args.dry_run:
        print(json.dumps({'commands': commands, 'freeze_policy': policy}, indent=2))
        return
    report_path = output / 'final_holdout/metrics.json'
    if resume_inference:
        ledger = read(ledger_path)
        if (ledger.get('status') != 'final_eval_complete' or ledger['output'] != str(output.resolve())
                or ledger['weights_sha256'] != sha(weights) or ledger['split_sha256'] != sha(SPLIT)
                or ledger['metrics_sha256'] != sha(report_path)
                or read(freeze)['evaluation_policy'] != policy):
            raise ValueError('No matching completed final evaluation to resume inference from')
    verify()
    preflight()
    if not resume_inference:
        output.mkdir(parents=True)
        ledger = {'status': 'final_eval_started', 'output': str(output.resolve()),
                  'weights_sha256': sha(weights), 'split_sha256': sha(SPLIT)}
        # Exclusive creation also prevents concurrent finalization across outputs.
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open('x', encoding='utf-8') as handle:
            json.dump(ledger, handle, indent=2)
        write(freeze, {'weights_sha256': sha(weights), 'split_sha256': sha(SPLIT), 'evaluation_policy': policy})
        execute(commands[0], output / 'final_eval.log')
    report = read(report_path)
    if not report['scope_complete'] or report['evaluated_images'] != 291:
        raise ValueError('Incomplete final holdout evaluation')
    ledger.update(status='final_eval_complete', metrics_sha256=sha(report_path))
    write(ledger_path, ledger)
    execute(commands[1], output / 'test_inference.log')
    write(output / 'completion.json', {'status': 'complete', 'weights_sha256': sha(weights),
          'test_images': len(list(Path('data/official/test').glob('*.jpg'))),
          'note': 'Local predictions only. Check current official submission specification before uploading.'})
    ledger['status'] = 'complete'
    write(ledger_path, ledger)


def main():
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('relocate')
    sub.add_parser('preflight')
    validation = sub.add_parser('verify')
    validation.add_argument('--full', action='store_true')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--stage', choices=['smoke', 'baseline', 'long'], required=True)
    run_parser.add_argument('--name', required=True)
    run_parser.add_argument('--epochs', type=int, default=60)
    run_parser.add_argument('--batch', type=int)
    run_parser.add_argument('--workers', type=int)
    run_parser.add_argument('--after', type=Path)
    run_parser.add_argument('--dry-run', action='store_true')
    recovery = sub.add_parser('evaluate-run')
    recovery.add_argument('--name', required=True)
    finish = sub.add_parser('finalize')
    finish.add_argument('--selection', type=Path, required=True)
    finish.add_argument('--output', type=Path, default=Path('runs/cloud4090/final_delivery'))
    finish.add_argument('--dry-run', action='store_true')
    finish.add_argument('--resume-inference', action='store_true')
    finish.add_argument('--batch', type=int, choices=[2, 4], default=4)
    args = parser.parse_args()
    if args.action == 'relocate':
        relocate()
    elif args.action == 'verify':
        verify(args.full)
    elif args.action == 'preflight':
        preflight()
    elif args.action == 'run':
        run(args)
    elif args.action == 'evaluate-run':
        evaluate_run(args)
    else:
        finalize(args)


if __name__ == '__main__':
    main()
