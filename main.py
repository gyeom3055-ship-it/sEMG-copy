"""전체 실험 실행.

    python main.py              # 홀드아웃 비교 + 누수 점검 + 오류 분석 + 5-fold 교차검증
    python main.py --quick      # 동작 확인용 (데이터 일부, 1 epoch, results_quick/ 에 저장)
    python main.py --skip-cv    # 교차검증 생략

단계별 결과를 results/ 에 JSON으로 저장하므로, 중간에 끊겨도 같은 명령으로 이어서 실행된다.
처음부터 다시 하려면 --force 를 붙이거나 results/ 안의 JSON을 지운다.
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from importlib import metadata

import torch

import analysis
import data as D
import evaluation as E
import train as T
from model import MODEL_LABEL

MODELS = ['densenet161', 'resnet18', 'cnn2d', 'cnn1d']


def log(msg):
    print(msg, flush=True)


def write_env_info(out_dir, device, args):
    pkgs = ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'PyWavelets', 'torch', 'torchvision']
    lines = [f'python {platform.python_version()} on {platform.platform()}']
    lines += [f'{p} {metadata.version(p)}' for p in pkgs]
    lines.append(f'device {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    lines.append(f'seed {D.SEED}, epochs {args.epochs}, batch {T.BATCH_SIZE}, lr {T.LR}, '
                 f'window {D.WIN}, hop {D.HOP}, cwt scales {len(D.SCALES)}')
    with open(os.path.join(out_dir, 'env_info.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def save_tables(out_dir, holdouts, cvs, sanity):
    with open(os.path.join(out_dir, 'tables.md'), 'w', encoding='utf-8') as f:
        f.write(E.make_tables(holdouts, cvs, sanity))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--out', default='results')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-cv', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.out = 1, 'results_quick'

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = T.get_device()
    write_env_info(out_dir, device, args)
    log(f'device={device}, epochs={args.epochs}, out={out_dir}')

    files = D.list_files()
    if args.quick:
        files = files[::5]
    t0 = time.time()
    log(f'시행 {len(files)}개 -> 윈도우/CWT 변환 중...')
    data = D.build_all(files)
    log(f'변환 완료 ({time.time() - t0:.0f}s): cwt {data["cwt"].shape}, 1d {data["1d"].shape}')

    tr_ids, te_ids = D.holdout_split(files)
    D.assert_trial_level(tr_ids, te_ids)
    log(f'홀드아웃 분할: train {len(tr_ids)}시행 / test {len(te_ids)}시행 (시행 단위, 겹침 없음 확인)')

    holdouts = {}
    for name in MODELS:
        holdouts[name] = T.run_holdout(name, data, tr_ids, te_ids, args.epochs, out_dir, device, args.force, log)
        E.plot_confusion(holdouts[name]['confusion_matrix'], MODEL_LABEL[name],
                         os.path.join(out_dir, f'confusion_matrix_{name}.png'))
    shutil.copyfile(os.path.join(out_dir, 'confusion_matrix_densenet161.png'), os.path.join(out_dir, 'confusion.png'))
    save_tables(out_dir, holdouts, {}, None)

    sanity = analysis.sanity_checks(data, tr_ids, te_ids, holdouts['cnn1d']['accuracy'], args.epochs,
                                    out_dir, device, args.force, log)
    analysis.error_analysis(files, holdouts['densenet161'], tr_ids, out_dir, MODEL_LABEL['densenet161'])
    analysis.position_analysis(files, te_ids, out_dir, MODELS, MODEL_LABEL)
    analysis.pair_similarity(files, holdouts, MODELS, MODEL_LABEL, out_dir)
    analysis.pair_waveforms(files, holdouts['densenet161'], out_dir)
    save_tables(out_dir, holdouts, {}, sanity)

    cvs = {}
    if not args.skip_cv:
        for name in MODELS:
            cvs[name] = T.run_cv(name, data, files, args.epochs, out_dir, device, args.force, log)
            save_tables(out_dir, holdouts, cvs, sanity)

    log('\n모든 단계 완료. 표: results/tables.md')


if __name__ == '__main__':
    main()
