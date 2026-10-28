"""Compare the six corrected matched runs with their frozen historical records."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def analyze(directory):
    directory = Path(directory).resolve()
    seeds = (42, 54, 99)
    models = ('Matched_Exact_NoRoPE', 'Matched_Approx_NoRoPE')
    expected = {(model, seed) for model in models for seed in seeds}

    def read(name):
        with (directory / name).open(newline='') as f:
            rows = list(csv.DictReader(f))
        keyed = {(r['model'], int(r['seed'])): r for r in rows}
        if len(keyed) != len(rows) or set(keyed) != expected:
            raise ValueError(f'{name}: expected exactly six unique model/seed endpoints; got {list(keyed)}')
        return keyed

    old = read('original_unified_tmaze_ablations_zeroshot.csv')
    new = read('unified_tmaze_ablations_zeroshot.csv')
    steps = {}
    for version, name in [('original', 'original_unified_tmaze_ablations_training.csv'),
                          ('corrected', 'unified_tmaze_ablations_training.csv')]:
        with (directory / name).open(newline='') as f:
            for row in csv.DictReader(f):
                steps[version, row['model'], int(row['seed'])] = int(row['step'])
    paired = []
    lines = ['# Corrected matched T-Maze screening pilot', '',
             'Three seeds per model (42, 54, 99), compared with frozen original records. '
             'Both corrections were applied together. Nominal 50M-step ceiling, original early stopping, '
             'and deterministic terminal evaluation at H=100/150/200 were retained.', '',
             'This is a comparison with historical runs, not contemporaneous original-versus-corrected '
             'retraining. Three seeds cannot establish equivalence or isolate which correction caused a change.', '',
             '| Model | Horizon | Original mean ± sample SD | Corrected mean ± sample SD | Mean paired change |',
             '|---|---:|---:|---:|---:|']
    for model in models:
        for horizon in (100, 150, 200):
            a, b = [], []
            for seed in seeds:
                key = (model, seed)
                original = float(old[key][f'val_H{horizon}'])
                corrected = float(new[key][f'val_H{horizon}'])
                a.append(original)
                b.append(corrected)
                paired.append(dict(model=model, seed=seed, horizon=horizon,
                                   original=original, corrected=corrected, delta=corrected-original,
                                   original_steps=steps['original', model, seed],
                                   corrected_steps=steps['corrected', model, seed]))
            lines.append(f'| {model} | {horizon} | {statistics.mean(a):.6f} ± {statistics.stdev(a):.6f} '
                         f'| {statistics.mean(b):.6f} ± {statistics.stdev(b):.6f} '
                         f'| {statistics.mean(y-x for x,y in zip(a,b)):+.6f} |')
    lines += ['', '## Individual paired endpoints', '',
              '| Model | Seed | Horizon | Original | Corrected | Change |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['seed']} | {r['horizon']} | {r['original']:.6f} | {r['corrected']:.6f} | {r['delta']:+.6f} |" for r in paired]
    lines += ['', '## Training endpoints', '', '| Model | Seed | Original steps | Corrected steps |', '|---|---:|---:|---:|']
    for model in models:
        for seed in seeds:
            lines.append(f"| {model} | {seed} | {steps['original', model, seed]} | {steps['corrected', model, seed]} |")
    with (directory / 'paired_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(paired[0])); writer.writeheader(); writer.writerows(paired)
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed_runs': len(new), 'report': str(directory / 'comparison.md')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    analyze(parser.parse_args().directory)
