"""재현성 점검: 같은 설정을 반복했을 때 홀드아웃 정확도가 얼마나 흔들리는지 잰다.

  same_seed    : 같은 시드(42)로 반복       -> GPU 연산의 비결정성만 반영
  other_seeds  : 시드 0~4로 반복            -> 초기화·배치 순서까지 반영
  window_split : 시드 0~4, 윈도우 단위 무작위 분할(누수) -> other_seeds(시행 단위 분할)와 시드별로 비교
분할(train/test 시행)과 설정(epoch 등)은 main.py와 같다. 이미 계산된 그룹은 건너뛰고 결과를 이어 붙인다.

    python seed_check.py [--epochs 45] [--force]
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

import data as D
import train as T
from model import MODEL_INPUT

PLAN = [('cnn1d', 'same_seed', [42] * 5), ('cnn1d', 'other_seeds', [0, 1, 2, 3, 4]),
        ('cnn1d', 'window_split', [0, 1, 2, 3, 4]), ('resnet18', 'other_seeds', [0, 1, 2]),
        ('densenet161', 'other_seeds', [0, 1, 2])]
LABEL = {'same_seed': '시행 단위, 같은 시드(42) 반복', 'other_seeds': '시행 단위, 시드 0~4',
         'window_split': '윈도우 단위 분할(누수), 시드 0~4'}


def summarize(rows):
    summary = {}
    for name, kind, _ in PLAN:
        a = np.array([r['accuracy'] for r in rows if r['model'] == name and r['kind'] == kind]) * 100
        if len(a):
            summary[f'{name}/{kind}'] = {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()),
                                         'max': float(a.max()), 'n': int(len(a))}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--out', default='results')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'seed_variability.json')
    rows = json.load(open(path, encoding='utf-8'))['rows'] if os.path.exists(path) and not args.force else []
    done = {(r['model'], r['kind']) for r in rows}

    files = D.list_files()
    data = D.build_all(files)
    tr_ids, te_ids = D.holdout_split(files)
    D.assert_trial_level(tr_ids, te_ids)
    tr, te = D.windows_of(data['g'], tr_ids), D.windows_of(data['g'], te_ids)
    device = T.get_device()
    y = data['y']
    tr_w, te_w = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=D.SEED)

    for name, kind, seeds in PLAN:
        if (name, kind) in done:
            continue
        X = data[MODEL_INPUT[name]]
        a_idx, b_idx = (tr_w, te_w) if kind == 'window_split' else (tr, te)
        for k, seed in enumerate(seeds):
            model, _, _ = T.fit(name, X[a_idx], y[a_idx], args.epochs, seed, device, log=lambda m: None)
            acc = float((T.predict_proba(model, X[b_idx], device).argmax(1) == y[b_idx]).mean())
            rows.append({'model': name, 'kind': kind, 'seed': seed, 'run': k + 1, 'accuracy': acc})
            print(f'{name} {kind} seed={seed} run={k + 1} acc={acc:.4f}', flush=True)
        summary = summarize(rows)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'epochs': args.epochs, 'rows': rows, 'summary': summary}, f, ensure_ascii=False, indent=2)

    summary = summarize(rows)
    lines = ['### 재현성 점검: 같은 설정 반복 시 홀드아웃 정확도 (%)', '',
             '| 모델 | 반복 방식 | 횟수 | 평균 | 표준편차 | 최소 ~ 최대 |', '|---|---|---|---|---|---|']
    for key, s in summary.items():
        name, kind = key.split('/')
        seeds = [r['seed'] for r in rows if r['model'] == name and r['kind'] == kind]
        label = f'시행 단위, 시드 {min(seeds)}~{max(seeds)}' if kind == 'other_seeds' else LABEL[kind]
        lines.append(f'| {name} | {label} | {s["n"]} | {s["mean"]:.2f} | {s["std"]:.2f} | '
                     f'{s["min"]:.2f} ~ {s["max"]:.2f} |')
    with open(os.path.join(out_dir, 'seed_variability.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
