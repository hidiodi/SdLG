"""Isolated, sequential corrected-source campaign; never edits manuscript/results sources."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 54, 99, 123, 456, 789, 111, 222, 333, 444]
MINIGRID_SEEDS = [42, 54, 99, 123, 456, 111, 222, 333, 444, 789, 1001, 1002, 1003, 1004, 1005]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'source_snapshot'
    for directory in ('src', 'scripts'):
        for src in (ROOT / directory).rglob('*.py'):
            if '__pycache__' in src.parts:
                continue
            dst = snapshot / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    jobs = []

    def add(family, **kwargs):
        job = dict(id=f'{len(jobs)+1:03d}_{family}', family=family, **kwargs)
        jobs.append(job)

    # Existing six corrected pilot runs remain separate and are not repeated.
    for seed in SEEDS[3:]:
        for model in ('Matched_Exact_NoRoPE', 'Matched_Approx_NoRoPE'):
            add('tmaze', study='matched', model=model, seed=seed)
    for seed in SEEDS:
        for model in ('Exact_RoPE', 'Exact_NoRoPE', 'Approx_RoPE', 'Approx_NoRoPE'):
            add('tmaze', study='factorial', model=model, seed=seed)
    for seed in SEEDS[:5]:
        add('doorkey', model='SdLG-Exact', seed=seed)
    add('latency')
    for seed in MINIGRID_SEEDS:
        for condition in ('SdLG-SWA', 'SdLG-None'):
            add('minigrid', condition=condition, seed=seed)
    for seed in SEEDS[:5]:
        for alpha in (0.1, 1.0):
            add('ggs', alpha=alpha, seed=seed)
    for seed in range(42, 47):
        for model, chunk, coupled in [('SdLGHybrid', 16, True), ('SdLGHybrid', 64, True), ('ExactSdLG', 64, False)]:
            add('craftax', model=model, chunk=chunk, coupled=coupled, seed=seed)
    variants = ['SdLG-1GLA-r16 (Rank Scale)', 'SdLG-2GLA-r4 (Recurrent Depth)', 'SdLG-3GLA-r4 (Homogeneous GLA)']
    for seed in SEEDS[:3]:
        for variant in variants:
            add('minesweeper', variant=variant, seed=seed)
    with (ROOT / 'results/popgym_results.csv').open(newline='') as f:
        cells = sorted({(r['Environment'], r['Model'], int(r['Seed'])) for r in csv.DictReader(f)
                        if r['Model'] in ('SdLG-Exact', 'SdLG-Approx')})
    for env, model, seed in cells:
        add('popgym', environment=env, model=model, seed=seed)
    # Preserve references only, not another copy labelled as newly generated results.
    references = ['results/popgym_results.csv', 'results/minesweeper_ablation_results.csv',
                  'scripts/diagnostics/exact_sdlg/unified_tmaze_ablations_zeroshot.csv']
    manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    policy='Only corrected implementations; no manuscript writes or automatic result replacement.',
                    pilot_dependency=str(ROOT / 'results/tmaze_corrected_pilot_20261026'),
                    python=str(ROOT / '.venv/bin/python'), jobs=jobs,
                    source_sha256={str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')},
                    original_references={p: sha(ROOT / p) for p in references},
                    paper_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'papers/latex/cleanRewrite').rglob('*')
                                  if p.is_file() and p.suffix in ('.tex', '.pdf')})
    write_json(out / 'manifest.json', manifest)
    with (out / 'queue.csv').open('w', newline='') as f:
        fields = ['id', 'family', 'model', 'seed', 'study', 'condition', 'alpha', 'chunk', 'coupled', 'variant', 'environment']
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(jobs)
    write_json(out / 'status.json', dict(state='prepared', jobs=len(jobs)))
    print(json.dumps({'directory': str(out), 'jobs': len(jobs)}, indent=2))


def dispatch(job_path):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    output = job_path.parent
    sys.path.insert(0, str(ROOT))
    os.chdir(output)
    family = job['family']

    def load(name):
        return importlib.import_module('scripts.diagnostics.' + name)

    if family == 'tmaze':
        m = load('exact_sdlg.exact_sdlg_tmaze_ablations')
        source_study = 'historical' if job['study'] == 'factorial' else job['study']
        selected = {job['model']: m.STUDY_ABLATIONS[source_study][job['model']]}
        # A distinct label prevents the legacy factorial plot generator running.
        m.STUDY_ABLATIONS['corrected_' + job['study']] = selected
        m.run_tmaze_ablations('corrected_' + job['study'], seed_override=[job['seed']],
                             runtime_csv=str(output / 'runtime.csv'), output_dir=str(output))
    elif family == 'doorkey':
        m = load('benchmarks.xminigrid.doorkey_baseline_benchmark')
        sys.argv = [m.__file__, '--envs', 'MiniGrid-DoorKey-8x8', '--models', 'SdLG-Exact',
                    '--seeds', str(job['seed']), '--results-dir', str(output), '--wandb-mode', 'offline']
        m.run(m.parse_args())
    elif family == 'latency':
        m = load('benchmarks.latency.profile_throughput')
        m.run_matched_tmaze_inference_benchmark(str(output / 'matched_tmaze_inference.csv'), warmup=50, repeats=300)
    elif family == 'minigrid':
        m = load('architecture.swa_ablation.swa_ablation')
        m.run_experiment10(seeds=[job['seed']], model_names=[job['condition']],
                           csv_path=str(output / 'results.csv'), fixed_endpoint=True,
                           manifest_path=str(output / 'runner_manifest.json'))
    elif family == 'ggs':
        m = load('architecture.gradient_shielding.gradient_shielding')
        m.run_experiment13(alphas=[job['alpha']], seeds=[job['seed']], csv_path=str(output / 'results.csv'),
                           manifest_path=str(output / 'runner_manifest.json'))
    elif family == 'craftax':
        m = load('benchmarks.craftax_full.craftax_classic_benchmark')
        m.RUN_CONFIGS = [(job['model'], job['chunk'], job['coupled'])]
        m.TRAINING_SEEDS = [job['seed']]
        m.RunTag = 'corrected-source'
        m.run_benchmark()
    elif family == 'minesweeper':
        m = load('exact_sdlg.ablation_minesweeper_sdlg')
        m.project_root = str(output)
        m.TRAINING_SEEDS = [job['seed']]
        m.ABLATION_VARIANTS = {job['variant']: m.ABLATION_VARIANTS[job['variant']]}
        m.main()
    elif family == 'popgym':
        m = load('exact_sdlg.popgym_benchmark')
        # This runner derives all checkpoint and CSV paths from project_root.
        # Redirect after imports; never discover/resume historical checkpoints.
        m.project_root = str(output)
        m.run_benchmark(total_steps=15_000_000, seeds=[job['seed']],
                        env_list=[job['environment']], model_list=[job['model']], eval_interval=1)
    else:
        raise ValueError(family)


def verify_result(job, folder, returncode):
    family = job['family']
    allowed = (0, 43) if family in ('craftax', 'popgym') else (0,)
    if returncode not in allowed:
        raise RuntimeError(f'worker exited {returncode}')
    paths = {
        'tmaze': 'unified_tmaze_ablations_zeroshot.csv', 'doorkey': 'run_summary.csv',
        'latency': 'matched_tmaze_inference.csv', 'minigrid': 'results.csv', 'ggs': 'results.csv',
        'minesweeper': 'results/minesweeper_ablation_results.csv', 'popgym': 'results/popgym_results.csv',
        'craftax': f"checkpoints/craftax_50m_watchdog_T{job.get('chunk')}/{job.get('model')}_T{job.get('chunk')}_seed_{job.get('seed')}_log.csv",
    }
    path = folder / paths[family]
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('No result rows written')
    endpoints = {'minigrid': ('Total_Steps', 9830400), 'ggs': ('Total_Steps', 20152320),
                 'craftax': ('Total_Steps', 49971200), 'minesweeper': ('Step', 14974976)}
    if family in endpoints:
        field, expected = endpoints[family]
        if int(float(rows[-1][field])) != expected:
            raise RuntimeError(f'Unexpected endpoint: {rows[-1][field]}, expected {expected}')
    if family == 'popgym' and 'Finished [' not in (folder / 'run.log').read_text():
        raise RuntimeError('POPGym did not log successful completion')
    return str(path)


def run(out):
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((out / 'manifest.json').read_text())
    snapshot = out / 'source_snapshot'
    for name, digest in manifest['source_sha256'].items():
        if sha(snapshot / name) != digest:
            raise RuntimeError(f'Snapshot changed: {name}')
    completed = []
    def status(state, **kwargs):
        write_json(out / 'status.json', dict(state=state, pid=os.getpid(), completed=len(completed),
                   total=len(manifest['jobs']), updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs))
    try:
        dependency = Path(manifest['pilot_dependency'])
        status('waiting_for_existing_pilot')
        while True:
            previous = json.loads((dependency / 'status.json').read_text())
            if previous['state'] == 'complete':
                break
            if previous['state'] != 'running':
                raise RuntimeError(f"Existing pilot is {previous['state']}; queue not started")
            try:
                os.kill(previous['supervisor_pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Existing pilot supervisor is absent; queue not started')
            time.sleep(60)
        env = dict(os.environ, HSA_ENABLE_SDMA='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WANDB_MODE='offline', WANDB_PROJECT='corrected-source-recheck', PYTHONUNBUFFERED='1')
        for job in manifest['jobs']:
            folder = out / 'jobs' / job['id']
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'done.json').exists():
                completed.append(job['id'])
                continue
            # A failed/partial job requires deliberate review; do not append a
            # fresh run to its old trajectory or silently auto-resume it.
            if (folder / 'run.log').exists():
                raise RuntimeError(f"Partial job {job['id']} needs review before restarting")
            write_json(folder / 'job.json', job)
            env['WANDB_DIR'] = str(folder)
            command = [manifest['python'], str(snapshot / 'scripts/diagnostics/queue_corrected_experiments.py'),
                       '--job', str(folder / 'job.json')]
            status('running', current=job)
            started = time.monotonic()
            with (folder / 'run.log').open('w') as log:
                result = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        result.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        with (folder / 'run.log').open('rb') as reader:
                            reader.seek(max(0, reader.seek(0, 2) - 65536))
                            tail = reader.read()
                        if b'HSA_STATUS_ERROR_' in tail:
                            result.terminate()
                            try:
                                result.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                result.kill()
                                result.wait(timeout=10)
                            raise RuntimeError(f"GPU runtime error in {job['id']}; stopped stalled worker")
            result_path = verify_result(job, folder, result.returncode)
            write_json(folder / 'done.json', dict(job=job, result=result_path, returncode=result.returncode,
                                                 wall_seconds=time.monotonic()-started))
            completed.append(job['id'])
            write_json(out / 'completed.json', completed)
        status('complete')
    except BaseException as error:
        status('blocked', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--run', type=Path)
    group.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.run:
        run(args.run.resolve())
    else:
        dispatch(args.job)
