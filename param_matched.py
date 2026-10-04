"""파라미터 수를 논문 Table 4에 맞춘 1D/2D CNN 베이스라인 실험.

논문은 1D CNN(9,589개)·2D CNN(4,944,901개)의 구조를 적지 않아 구조까지 재현할 수는 없다.
대신 크기를 같게 맞춰, 기존 베이스라인(136,293개 / 19,717개)과의 차이가 파라미터 수 때문인지 확인한다.
조건(분할·45 epoch·학습률·배치·시드 42)은 main.py와 같고, 홀드아웃 + 5-fold + 시드 0~4 반복을 한다.

    python param_matched.py [--epochs 45]
"""
import argparse
import json
import os

import numpy as np

import data as D
import evaluation as E
import train as T
from model import MODEL_INPUT, MODEL_LABEL, PAPER_PARAMS, build_model, count_params

MODELS = ['cnn1d_pm', 'cnn2d_pm']
SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = T.get_device()

    for name in MODELS:
        n = count_params(build_model(name))
        assert n == PAPER_PARAMS[name], f'{name}: {n} != {PAPER_PARAMS[name]}'
    print('파라미터 수 일치 확인:', {m: f'{PAPER_PARAMS[m]:,}' for m in MODELS}, flush=True)

    files = D.list_files()
    data = D.build_all(files)
    tr_ids, te_ids = D.holdout_split(files)
    D.assert_trial_level(tr_ids, te_ids)
    tr, te = D.windows_of(data['g'], tr_ids), D.windows_of(data['g'], te_ids)
    y = data['y']

    seed_path = os.path.join(out_dir, 'param_matched_seeds.json')
    seed_rows = json.load(open(seed_path, encoding='utf-8')) if os.path.exists(seed_path) else []

    for name in MODELS:
        h = T.run_holdout(name, data, tr_ids, te_ids, args.epochs, out_dir, device, log=lambda m: None)
        E.plot_confusion(h['confusion_matrix'], MODEL_LABEL[name], os.path.join(out_dir, f'confusion_matrix_{name}.png'))
        print(f'{name} holdout acc={h["accuracy"]:.4f} f1={h["f1"]:.4f}', flush=True)

        cv = T.run_cv(name, data, files, args.epochs, out_dir, device, log=lambda m: None)
        print(f'{name} 5-fold acc={cv["mean_accuracy"]:.4f}+-{cv["std_accuracy"]:.4f}', flush=True)

        X = data[MODEL_INPUT[name]]
        for seed in SEEDS:
            if any(r['model'] == name and r['seed'] == seed for r in seed_rows):
                continue
            model, _, _ = T.fit(name, X[tr], y[tr], args.epochs, seed, device, log=lambda m: None)
            acc = float((T.predict_proba(model, X[te], device).argmax(1) == y[te]).mean())
            seed_rows.append({'model': name, 'seed': seed, 'accuracy': acc})
            json.dump(seed_rows, open(seed_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            print(f'{name} seed={seed} acc={acc:.4f}', flush=True)

    lines = ['### 파라미터를 논문에 맞춘 베이스라인 (45 epoch)', '',
             '| 모델 | 파라미터 | 홀드아웃 Acc | 홀드아웃 F1 | 5-fold Acc | 시드 0~4 Acc 평균 ± 표준편차 (최소~최대) |',
             '|---|---|---|---|---|---|']
    for name in MODELS:
        h = json.load(open(os.path.join(out_dir, f'holdout_{name}.json'), encoding='utf-8'))
        cv = json.load(open(os.path.join(out_dir, f'cv_{name}.json'), encoding='utf-8'))
        a = np.array([r['accuracy'] for r in seed_rows if r['model'] == name]) * 100
        lines.append(f'| {MODEL_LABEL[name]} | {h["n_params"]:,} | {h["accuracy"] * 100:.2f} | {h["f1"] * 100:.2f} | '
                     f'{cv["mean_accuracy"] * 100:.2f} ± {cv["std_accuracy"] * 100:.2f} | '
                     f'{a.mean():.2f} ± {a.std():.2f} ({a.min():.2f}~{a.max():.2f}) |')
    open(os.path.join(out_dir, 'param_matched.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
