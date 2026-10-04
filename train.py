"""학습 · 예측 · 홀드아웃 / 5-fold 교차검증 실행.

모든 모델은 같은 분할 · 같은 에폭 · 같은 학습률(Adam 1e-3) · 같은 배치(16) · 같은 시드로 학습한다.
각 단계 결과는 JSON으로 저장하고, 이미 있으면 건너뛴다 (중간에 끊겨도 이어서 실행 가능).
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import data as D
import evaluation as E
from model import MODEL_INPUT, build_model, count_params

BATCH_SIZE = 16
LR = 1e-3


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def fit(name, X, y, n_epochs, seed, device, log=print):
    """모델을 새로 만들어 학습한다. -> (model, epoch별 loss, 학습 시간[초])"""
    set_seed(seed)
    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    _sync(device)
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        history.append(tot / len(loader))
        log(f'  [{name}] epoch {epoch + 1}/{n_epochs}  loss={history[-1]:.4f}')
    _sync(device)
    return model, history, time.time() - t0


@torch.no_grad()
def predict_proba(model, X, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_holdout(name, data, tr_ids, te_ids, n_epochs, out_dir, device, force=False, log=print):
    """홀드아웃 분할로 학습 -> 테스트셋 평가 + 계산 비용 측정."""
    path = os.path.join(out_dir, f'holdout_{name}.json')
    if os.path.exists(path) and not force:
        log(f'[holdout:{name}] 저장된 결과 사용: {path}')
        return _read_json(path)

    X, y, g = data[MODEL_INPUT[name]], data['y'], data['g']
    D.assert_trial_level(tr_ids, te_ids)
    tr, te = D.windows_of(g, tr_ids), D.windows_of(g, te_ids)
    log(f'[holdout:{name}] train 윈도우 {len(tr)} / test 윈도우 {len(te)}')

    model, history, seconds = fit(name, X[tr], y[tr], n_epochs, D.SEED, device, log)
    proba_te = predict_proba(model, X[te], device)
    proba_tr = predict_proba(model, X[tr], device)

    result = E.evaluate(y[te], proba_te, g[te])
    result.update(
        model=name, epochs=n_epochs, n_train_windows=int(len(tr)), n_test_windows=int(len(te)),
        train_accuracy=float((proba_tr.argmax(1) == y[tr]).mean()),
        train_seconds=seconds, loss_history=history, n_params=count_params(model),
        inference_ms=E.measure_inference_ms(model, X[te][:1], device))
    np.save(os.path.join(out_dir, f'proba_{name}.npy'), proba_te)
    np.save(os.path.join(out_dir, 'test_window_index.npy'), te)
    torch.save(model.state_dict(), os.path.join(out_dir, f'{name}.pt'))
    _write_json(path, result)
    log(f'[holdout:{name}] acc={result["accuracy"]:.4f} f1={result["f1"]:.4f} '
        f'train_acc={result["train_accuracy"]:.4f} time={seconds:.0f}s')
    return result


def run_cv(name, data, files, n_epochs, out_dir, device, force=False, log=print):
    """시행 단위 층화 5-fold 교차검증. 폴드마다 모델을 새로 만든다."""
    path = os.path.join(out_dir, f'cv_{name}.json')
    state = _read_json(path) if os.path.exists(path) and not force else {'model': name, 'folds': []}

    X, y, g = data[MODEL_INPUT[name]], data['y'], data['g']
    splits = D.cv_splits(files)
    for k, (tr_ids, va_ids) in enumerate(splits):
        if k < len(state['folds']):
            continue
        D.assert_trial_level(tr_ids, va_ids)
        tr, va = D.windows_of(g, tr_ids), D.windows_of(g, va_ids)
        log(f'[cv:{name}] fold {k + 1}/{len(splits)}  train {len(tr)} / val {len(va)} 윈도우')
        model, history, seconds = fit(name, X[tr], y[tr], n_epochs, D.SEED + k, device, log)
        m = E.evaluate(y[va], predict_proba(model, X[va], device), g[va])
        state['folds'].append({'fold': k + 1, 'accuracy': m['accuracy'], 'f1': m['f1'],
                               'train_seconds': seconds, 'final_loss': history[-1]})
        _write_json(path, state)
        log(f'[cv:{name}] fold {k + 1} acc={m["accuracy"]:.4f} f1={m["f1"]:.4f}')
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    accs = [f['accuracy'] for f in state['folds']]
    state['mean_accuracy'] = float(np.mean(accs))
    state['std_accuracy'] = float(np.std(accs))       # 모집단 표준편차(ddof=0), 강의자료와 동일
    state['epochs'] = n_epochs
    _write_json(path, state)
    return state
