"""오류 분석(최대 오류 쌍의 신호 비교)과 데이터 누수 점검."""
import itertools
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

import data as D
import train as T

FEATURES = ['rms3', 'rms4', 'ratio', 'mdf3', 'mdf4', 't_peak']
FEATURE_TITLE = {'rms3': 'RMS Ch3', 'rms4': 'RMS Ch4', 'ratio': 'RMS Ch3/Ch4',
                 'mdf3': 'Median freq Ch3 (Hz)', 'mdf4': 'Median freq Ch4 (Hz)',
                 't_peak': 'Envelope peak time (s)'}


def trial_features(files):
    """시행별 요약 특징(진폭 · 중앙주파수 · 피크 시점)과 Welch 파워스펙트럼."""
    rows, psd = [], []
    for f in files:
        x = D.preprocess(D.load(f))
        rms = x.std(axis=0)
        row = {'rms3': rms[0], 'rms4': rms[1], 'ratio': rms[0] / rms[1]}
        spectra = []
        for ch, key in enumerate(['mdf3', 'mdf4']):
            freq, p = welch(x[:, ch], fs=D.FS, nperseg=512)
            row[key] = freq[np.searchsorted(np.cumsum(p), p.sum() / 2)]
            spectra.append(p)
        env = np.convolve(np.sqrt((x ** 2).sum(axis=1)), np.ones(100) / 100, mode='same')
        row['t_peak'] = env.argmax() / D.FS
        rows.append(row)
        psd.append(spectra)
    F = {k: np.array([r[k] for r in rows]) for k in FEATURES}
    return F, np.array(psd), freq


def _z_matrix(F):
    """거리 계산용 특징 행렬: log(진폭) 2개 + 중앙주파수 2개 + 피크 시점."""
    return np.column_stack([np.log(F['rms3']), np.log(F['rms4']), F['mdf3'], F['mdf4'], F['t_peak']])


def error_analysis(files, holdout, tr_ids, out_dir, model_label):
    """최대 오류 쌍과 오분류된 시행을 신호 특징으로 비교한다 (클래스 통계는 train 시행만 사용)."""
    F, psd, freq = trial_features(files)
    y = np.array([D.label_of(f) for f in files])
    subjects = D.SUBJECTS

    lines = ['### 피험자별 신호 특징 (전체 50시행, 평균 ± 표준편차)', '',
             '| Subject | ' + ' | '.join(FEATURE_TITLE[k] for k in FEATURES) + ' |',
             '|---|' + '---|' * len(FEATURES)]
    for i, s in enumerate(subjects):
        lines.append(f'| {s} | ' + ' | '.join(f'{F[k][y == i].mean():.3f} ± {F[k][y == i].std():.3f}'
                                               for k in FEATURES) + ' |')

    Z = _z_matrix(F)
    mu, sd = Z[tr_ids].mean(axis=0), Z[tr_ids].std(axis=0) + 1e-12
    Zs = (Z - mu) / sd
    centers = np.array([Zs[tr_ids][y[tr_ids] == i].mean(axis=0) for i in range(len(subjects))])

    errors = holdout['trial_errors']
    lines += ['', f'### {model_label}가 틀린 테스트 시행 (시행 단위, 총 {holdout["n_trials"]}시행 중 {len(errors)}개)', '']
    mis = []
    if errors:
        lines += ['| 파일 | 실제 -> 예측 | RMS Ch3 | RMS Ch4 | MDF Ch3 | MDF Ch4 | 피크 시점 | 실제 클래스까지 거리 | 예측 클래스까지 거리 |',
                  '|---|---|---|---|---|---|---|---|---|']
    for e in errors:
        i = e['file_id']
        t, p = subjects.index(e['true']), subjects.index(e['pred'])
        d_true, d_pred = np.linalg.norm(Zs[i] - centers[t]), np.linalg.norm(Zs[i] - centers[p])
        mis.append({'file': os.path.basename(files[i]), 'file_id': i, 'true': e['true'], 'pred': e['pred'],
                    'dist_true': float(d_true), 'dist_pred': float(d_pred)})
        lines.append(f'| {os.path.basename(files[i])} | {e["true"]} -> {e["pred"]} | {F["rms3"][i]:.3f} | '
                     f'{F["rms4"][i]:.3f} | {F["mdf3"][i]:.0f} | {F["mdf4"][i]:.0f} | {F["t_peak"][i]:.2f} | '
                     f'{d_true:.2f} | {d_pred:.2f} |')
    if not errors:
        lines.append('틀린 테스트 시행이 없다.')

    colors = plt.cm.tab10(np.arange(len(subjects)))
    pair = holdout['max_error_pair']
    focus = {pair['true'], pair['pred']} if pair else set()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ch, ax in enumerate(axes[:2]):
        for i, s in enumerate(subjects):
            bold = s in focus
            ax.semilogy(freq, psd[y == i, ch].mean(axis=0), color=colors[i], label=s,
                        lw=2.4 if bold else 1.0, alpha=1.0 if bold else 0.55)
        ax.set_xlim(0, 300)
        ax.set_xlabel('Hz'); ax.set_ylabel('mean power')
        ax.set_title(f'Mean PSD, Ch{ch + 3} ({"APB" if ch == 0 else "ADM"})')
        ax.legend(ncol=5, fontsize=8)
    ax = axes[2]
    for i, s in enumerate(subjects):
        m = y == i
        ax.scatter(F['rms3'][m], F['rms4'][m], s=14, color=colors[i], label=s, alpha=0.7)
    for e in mis:
        ax.scatter(F['rms3'][e['file_id']], F['rms4'][e['file_id']], marker='x', s=90, color='black')
        ax.annotate(f'{e["true"]}->{e["pred"]}', (F['rms3'][e['file_id']], F['rms4'][e['file_id']]),
                    fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('RMS Ch3'); ax.set_ylabel('RMS Ch4')
    ax.set_title('Per-trial amplitude (x = misclassified test trial)')
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_analysis.png'), dpi=120)
    plt.close(fig)

    with open(os.path.join(out_dir, 'error_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'error_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump({'max_error_pair': pair, 'misclassified_trials': mis}, f, ensure_ascii=False, indent=2)
    return mis


def sanity_checks(data, tr_ids, te_ids, honest_acc, n_epochs, out_dir, device, force=False, log=print):
    """누수 점검: (a) 라벨을 섞으면 우연 수준이어야 하고, (b) 윈도우 단위로 섞어 나누면 부풀려져야 한다."""
    path = os.path.join(out_dir, 'sanity_checks.json')
    if os.path.exists(path) and not force:
        log(f'[sanity] 저장된 결과 사용: {path}')
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    X, y, g = data['1d'], data['y'], data['g']
    tr, te = D.windows_of(g, tr_ids), D.windows_of(g, te_ids)
    rows = [{'name': '시행 단위 분할 (정상)', 'accuracy': honest_acc,
             'note': '이 과제에서 사용한 방식'}]

    y_perm = np.random.RandomState(D.SEED).permutation(y[tr])
    model, _, _ = T.fit('cnn1d', X[tr], y_perm, n_epochs, D.SEED, device, log)
    acc = float((T.predict_proba(model, X[te], device).argmax(1) == y[te]).mean())
    rows.append({'name': '학습 라벨을 무작위로 섞음 (시행 단위 분할)', 'accuracy': acc,
                 'note': '우연 수준(약 20%)이어야 정상 -> 데이터·평가 코드에 숨은 정답 정보가 없다는 증거'})

    all_idx = np.arange(len(y))
    tr_w, te_w = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=D.SEED)
    shared = int(np.intersect1d(g[tr_w], g[te_w]).size)
    model, _, _ = T.fit('cnn1d', X[tr_w], y[tr_w], n_epochs, D.SEED, device, log)
    acc = float((T.predict_proba(model, X[te_w], device).argmax(1) == y[te_w]).mean())
    rows.append({'name': f'윈도우 단위 무작위 분할 (누수, 같은 시행 {shared}개가 양쪽에 존재)', 'accuracy': acc,
                 'note': '같은 시행의 인접 윈도우가 양쪽에 들어가 정확도가 부풀려진다'})

    result = {'rows': rows}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def position_analysis(files, te_ids, out_dir, models, labels):
    """테스트 윈도우의 오류율을 시행 내 위치(시간)별로 보고, 같은 위치의 신호 활동량과 비교한다.

    저장된 홀드아웃 예측 확률(proba_*.npy)만 사용하므로 다시 학습하지 않는다.
    """
    te = np.load(os.path.join(out_dir, 'test_window_index.npy'))
    n_win = (3000 - D.WIN) // D.HOP + 1
    assert len(te) == len(te_ids) * n_win, '모든 시행의 길이가 3000샘플이어야 한다'
    pos = te % n_win
    y = np.array([D.label_of(files[i]) for i in te // n_win])
    centers = (np.arange(n_win) * D.HOP + D.WIN / 2) / D.FS

    error_rate, wrong_per_trial = {}, {}
    for name in models:
        wrong = np.load(os.path.join(out_dir, f'proba_{name}.npy')).argmax(1) != y
        error_rate[name] = [float(wrong[pos == k].mean()) for k in range(n_win)]
        wrong_per_trial[name] = wrong.reshape(len(te_ids), n_win).sum(axis=1).tolist()

    activity = np.zeros(n_win)
    for i in te_ids:
        x = D.preprocess(D.load(files[i]))
        w = D.make_windows(x)
        activity += np.sqrt((w ** 2).mean(axis=(1, 2))) / np.sqrt((x ** 2).mean())
    activity /= len(te_ids)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name in models:
        a1.plot(centers, np.array(error_rate[name]) * 100, marker='o', ms=3, label=labels[name])
    a1.set_ylabel('window error rate (%)')
    a1.set_title('Error rate by window position (test set)')
    a1.legend(fontsize=8)
    a2.plot(centers, activity, color='black', marker='o', ms=3)
    a2.set_ylabel('window RMS / trial RMS')
    a2.set_xlabel('window center time (s)')
    a2.set_title('Relative signal activity (mean over test trials)')
    for ax in (a1, a2):
        for t in (1.0, 2.0):
            ax.axvline(t, color='gray', ls=':', lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_by_position.png'), dpi=120)
    plt.close(fig)

    summary = {'window_center_s': centers.tolist(), 'error_rate': error_rate,
               'relative_activity': activity.tolist(), 'wrong_windows_per_trial': wrong_per_trial,
               'corr_error_vs_activity': {
                   name: float(np.corrcoef(error_rate[name], activity)[0, 1]) for name in models}}
    with open(os.path.join(out_dir, 'error_by_position.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def pair_similarity(files, holdouts, models, labels, out_dir):
    """피험자 쌍의 신호 특징 차이와 그 쌍의 혼동 횟수(양방향 합) 사이의 Spearman 상관.

    윈도우를 두 채널 공통으로 min-max 정규화하면 절대 진폭은 사라지고 Ch3/Ch4 크기 비율은 그대로 남는다.
    따라서 "비율이 비슷한 쌍일수록 더 자주 헷갈리는가"를 확인한다 (쌍 10개, 탐색적 분석).
    """
    F, _, _ = trial_features(files)
    y = np.array([D.label_of(f) for f in files])
    feats = {'Ch3/Ch4 비율(log)': np.log(F['ratio']), 'RMS Ch3(log)': np.log(F['rms3']),
             'RMS Ch4(log)': np.log(F['rms4']), '중앙주파수 Ch3': F['mdf3'], '중앙주파수 Ch4': F['mdf4']}
    n = len(D.SUBJECTS)
    pairs = list(itertools.combinations(range(n), 2))
    mu = {k: np.array([v[y == i].mean() for i in range(n)]) for k, v in feats.items()}

    result = {'subject_mean': {k: v.tolist() for k, v in mu.items()}, 'models': {}}
    for name in models:
        cm = np.array(holdouts[name]['confusion_matrix'])
        conf = np.array([cm[i, j] + cm[j, i] for i, j in pairs])
        entry = {'pair_confusions': {D.SUBJECTS[i] + D.SUBJECTS[j]: int(c) for (i, j), c in zip(pairs, conf)}}
        for k in feats:
            dist = np.array([abs(mu[k][i] - mu[k][j]) for i, j in pairs])
            rho, p = spearmanr(dist, conf)
            entry[k] = {'rho': float(rho), 'p': float(p)}
        result['models'][name] = entry

    lines = ['### 피험자 쌍의 특징 차이와 혼동 횟수의 Spearman 상관 (ρ, p; 쌍 10개)', '',
             '| Model | ' + ' | '.join(feats) + ' |', '|---|' + '---|' * len(feats)]
    for name in models:
        e = result['models'][name]
        lines.append(f'| {labels[name]} | ' + ' | '.join(f'{e[k]["rho"]:+.2f} ({e[k]["p"]:.3f})' for k in feats) + ' |')
    pair_names = [D.SUBJECTS[i] + D.SUBJECTS[j] for i, j in pairs]
    lines += ['', '### 쌍별 혼동 횟수 (양방향 합, 윈도우)', '', '| Model | ' + ' | '.join(pair_names) + ' |',
              '|---|' + '---|' * len(pairs)]
    for name in models:
        c = result['models'][name]['pair_confusions']
        lines.append(f'| {labels[name]} | ' + ' | '.join(str(c[p]) for p in pair_names) + ' |')
    lines += ['', '### 피험자별 평균 (전체 시행)', '', '| 특징 | ' + ' | '.join(D.SUBJECTS) + ' |', '|---|' + '---|' * n]
    for k in feats:
        lines.append(f'| {k} | ' + ' | '.join(f'{v:.2f}' for v in mu[k]) + ' |')

    with open(os.path.join(out_dir, 'pair_similarity.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'pair_similarity.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def pair_waveforms(files, holdout, out_dir):
    """최대 오류 쌍의 두 피험자 신호를 나란히 그린다 (각자 Ch3/Ch4 비율이 중앙값에 가장 가까운 대표 시행)."""
    pair = holdout['max_error_pair']
    if not pair:
        return None
    F, _, _ = trial_features(files)
    y = np.array([D.label_of(f) for f in files])
    fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey='row')
    chosen = {}
    for col, s in enumerate([pair['true'], pair['pred']]):
        idx = np.where(y == D.SUBJECTS.index(s))[0]
        i = idx[np.argmin(np.abs(F['ratio'][idx] - np.median(F['ratio'][idx])))]
        x = D.preprocess(D.load(files[i]))
        t = np.arange(len(x)) / D.FS
        for ch in range(2):
            axes[ch, col].plot(t, x[:, ch], lw=0.5, color=plt.cm.tab10(D.SUBJECTS.index(s)))
            axes[ch, col].set_ylabel(f'Ch{ch + 3} ({"APB" if ch == 0 else "ADM"})')
        axes[0, col].set_title(f'{s}: {os.path.basename(files[i])}  (Ch3/Ch4 RMS = {F["ratio"][i]:.1f})', fontsize=10)
        axes[1, col].set_xlabel('time (s)')
        chosen[s] = {'file': os.path.basename(files[i]), 'ratio': float(F['ratio'][i])}
    fig.suptitle(f'Max error pair {pair["true"]} -> {pair["pred"]}: typical trial of each subject (same y-scale per row)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pair_waveforms.png'), dpi=120)
    plt.close(fig)
    return chosen
