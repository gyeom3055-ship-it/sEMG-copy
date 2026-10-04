"""평가 지표 · 혼동행렬(최대 오류 쌍 표시) · 계산 비용 측정 · 결과 표 생성."""
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data import SUBJECTS
from model import MODEL_LABEL

PAPER = {'accuracy': 94.00, 'f1': 93.99, 'cv_mean': 91.66, 'cv_std': 2.78,
         'max_error': 'D -> C (13)', 'train_seconds': 6441, 'n_params': 26_483_045}


def max_error_pair(cm):
    """대각선을 제외한 가장 큰 칸 = 가장 자주 헷갈린 (실제 -> 예측) 쌍."""
    off = np.array(cm).copy()
    np.fill_diagonal(off, 0)
    if off.max() == 0:
        return None
    t, p = np.unravel_index(off.argmax(), off.shape)
    return {'true': SUBJECTS[t], 'pred': SUBJECTS[p], 'count': int(off[t, p])}


def evaluate(y_true, proba, groups):
    """윈도우 단위 지표 + 혼동행렬, 그리고 시행 단위(윈도우 확률 평균) 정확도."""
    labels = list(range(len(SUBJECTS)))
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='macro', zero_division=0)
    pcp, pcr, pcf, pcs = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)

    gids = np.unique(groups)
    t_true = np.array([y_true[groups == i][0] for i in gids])
    t_pred = np.array([proba[groups == i].mean(axis=0).argmax() for i in gids])
    wrong = np.where(t_true != t_pred)[0]
    return {
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(p), 'recall': float(r), 'f1': float(f),
        'per_class': {SUBJECTS[i]: {'precision': float(pcp[i]), 'recall': float(pcr[i]),
                                    'f1': float(pcf[i]), 'support': int(pcs[i])} for i in labels},
        'confusion_matrix': cm.tolist(),
        'max_error_pair': max_error_pair(cm),
        'trial_accuracy': float((t_true == t_pred).mean()),
        'n_trials': int(len(gids)),
        'trial_errors': [{'file_id': int(gids[i]), 'true': SUBJECTS[t_true[i]],
                          'pred': SUBJECTS[t_pred[i]]} for i in wrong],
    }


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(SUBJECTS)), SUBJECTS)
    ax.set_yticks(range(len(SUBJECTS)), SUBJECTS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    pair = max_error_pair(cm)
    sub = ''
    if pair:
        t, p = SUBJECTS.index(pair['true']), SUBJECTS.index(pair['pred'])
        ax.add_patch(plt.Rectangle((p - 0.5, t - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5))
        sub = f"max error (red box): {pair['true']} -> {pair['pred']}, {pair['count']} windows"
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(f'{title}\n{sub}', fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def measure_inference_ms(model, sample, device, warmup=10, repeat=100):
    """샘플 1개당 추론 시간(ms). 예열 후 측정하고 GPU는 동기화한다."""
    model.eval()
    x = torch.from_numpy(np.asarray(sample)).to(device)
    for _ in range(warmup):
        model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def _pct(v):
    return f'{v * 100:.2f}'


def make_tables(holdouts, cvs, sanity=None):
    """results/tables.md에 들어갈 마크다운 표를 만든다 (README에는 이 표를 그대로 옮긴다)."""
    names = [n for n in MODEL_LABEL if n in holdouts]
    L = []

    L += ['### 베이스라인 비교표 (홀드아웃)', '',
          '| Model | Accuracy (%) | F1-score (%) | Train time (s) | Parameters |', '|---|---|---|---|---|']
    for n in names:
        h = holdouts[n]
        L.append(f'| {MODEL_LABEL[n]} | {_pct(h["accuracy"])} | {_pct(h["f1"])} | '
                 f'{h["train_seconds"]:.1f} | {h["n_params"]:,} |')

    L += ['', '### 홀드아웃 테스트 성능 (macro 평균, %)', '',
          '| Model | Accuracy | Precision | Recall | F1-score | Train acc | Trial-level acc |',
          '|---|---|---|---|---|---|---|']
    for n in names:
        h = holdouts[n]
        L.append(f'| {MODEL_LABEL[n]} | {_pct(h["accuracy"])} | {_pct(h["precision"])} | '
                 f'{_pct(h["recall"])} | {_pct(h["f1"])} | {_pct(h["train_accuracy"])} | '
                 f'{_pct(h["trial_accuracy"])} |')

    L += ['', '### 계산 비용', '',
          '| Model | Parameters | Train time (s) | Inference (ms / sample) |', '|---|---|---|---|']
    for n in names:
        h = holdouts[n]
        L.append(f'| {MODEL_LABEL[n]} | {h["n_params"]:,} | {h["train_seconds"]:.1f} | {h["inference_ms"]:.2f} |')

    cv_names = [n for n in names if n in cvs and len(cvs[n]['folds']) > 0]
    if cv_names:
        k = max(len(cvs[n]['folds']) for n in cv_names)
        L += ['', '### 5-fold 교차검증 정확도 (%)', '',
              '| Model | ' + ' | '.join(f'Fold {i + 1}' for i in range(k)) + ' | Mean ± Std |',
              '|---|' + '---|' * k + '---|']
        for n in cv_names:
            accs = [f['accuracy'] for f in cvs[n]['folds']]
            L.append(f'| {MODEL_LABEL[n]} | ' + ' | '.join(_pct(a) for a in accs) +
                     f' | {np.mean(accs) * 100:.2f} ± {np.std(accs) * 100:.2f} |')

    L += ['', '### 클래스(피험자)별 F1-score (%)', '',
          '| Model | ' + ' | '.join(SUBJECTS) + ' |', '|---|' + '---|' * len(SUBJECTS)]
    for n in names:
        L.append(f'| {MODEL_LABEL[n]} | ' +
                 ' | '.join(_pct(holdouts[n]['per_class'][s]['f1']) for s in SUBJECTS) + ' |')

    L += ['', '### 최대 오류 쌍 (윈도우 기준, 실제 -> 예측)', '', '| Model | 최대 오류 쌍 | 건수 |', '|---|---|---|']
    for n in names:
        p = holdouts[n]['max_error_pair']
        L.append(f'| {MODEL_LABEL[n]} | ' + (f'{p["true"]} -> {p["pred"]} | {p["count"]} |' if p else '- | 0 |'))

    d = holdouts.get('densenet161')
    if d:
        cv = cvs.get('densenet161')
        cv_txt = (f'{cv["mean_accuracy"] * 100:.2f} ± {cv["std_accuracy"] * 100:.2f}'
                  if cv and 'mean_accuracy' in cv else '-')
        p = d['max_error_pair']
        L += ['', '### 논문(DenseNet161) 수치와 비교', '', '| 항목 | 논문 | 내 결과 |', '|---|---|---|',
              f'| 테스트 정확도 (%) | {PAPER["accuracy"]:.2f} | {_pct(d["accuracy"])} |',
              f'| F1-score (%) | {PAPER["f1"]:.2f} | {_pct(d["f1"])} |',
              f'| 5-fold 평균 ± 표준편차 (%) | {PAPER["cv_mean"]:.2f} ± {PAPER["cv_std"]:.2f} | {cv_txt} |',
              f'| 최대 오류 쌍 | {PAPER["max_error"]} | ' + (f'{p["true"]} -> {p["pred"]} ({p["count"]})' if p else '-') + ' |',
              f'| 학습 시간 (s) | {PAPER["train_seconds"]:,} | {d["train_seconds"]:.0f} ({d["epochs"]} epochs) |',
              f'| 파라미터 수 | {PAPER["n_params"]:,} | {d["n_params"]:,} |']

    if sanity:
        L += ['', '### 누수 점검 (1D CNN, 같은 조건)', '', '| 실험 | 테스트 정확도 (%) | 의미 |', '|---|---|---|']
        for row in sanity.get('rows', []):
            L.append(f'| {row["name"]} | {_pct(row["accuracy"])} | {row["note"]} |')
    return '\n'.join(L) + '\n'
