"""데이터 로딩 · 전처리 · 슬라이딩 윈도우 · CWT · 시행(파일) 단위 분할."""
import glob
import os

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedKFold, train_test_split

FS = 1000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

SUBJECTS = ['A', 'B', 'C', 'D', 'E']
SUBJECT_TO_LABEL = {s: i for i, s in enumerate(SUBJECTS)}

WIN, HOP = 300, 150          # 300ms 윈도우, 150ms 간격 (50% 오버랩)
SCALES = np.arange(1, 33)    # CWT 스케일 32개 (강의자료·논문 설정)
SEED = 42

_BN, _AN = iirnotch(60, 30, FS)
_B, _A = butter(4, [20 / (FS / 2), 499 / (FS / 2)], btype='band')


def list_files():
    files = sorted(glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True))
    return [f for f in files if not os.path.basename(f).startswith('~$')]


def label_of(filepath):
    return SUBJECT_TO_LABEL[os.path.basename(os.path.dirname(filepath))]


def load(filepath):
    return np.loadtxt(filepath, delimiter=',', skiprows=1)   # (3000, 2)


def preprocess(x):
    """60Hz 노치 필터 -> 20~499Hz 대역통과 필터 (시간축 기준)."""
    x = filtfilt(_BN, _AN, x, axis=0)
    return filtfilt(_B, _A, x, axis=0)


def make_windows(x, win=WIN, hop=HOP):
    """(시간, 채널) -> (윈도우수, win, 채널)"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop: i * hop + win] for i in range(n)])


def minmax(w, eps=1e-8):
    """윈도우별 0~1 정규화. w: (윈도우수, 시간, 채널)"""
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)
    return (w - mn) / (mx - mn + eps)


def to_cwt(one_window, wavelet='morl'):
    """(300, 2) -> (3, 32, 300): 채널별 CWT 크기 + 두 채널 평균."""
    maps = [np.abs(pywt.cwt(one_window[:, ch], SCALES, wavelet)[0]) for ch in range(2)]
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)


def build_all(files):
    """모든 시행을 윈도우로 변환한다.

    윈도우는 시행마다 독립적으로(필터·정규화 모두 시행/윈도우 내부 통계만 사용) 만들어지므로,
    분할은 반드시 g(시행 번호) 기준으로 한다. 반환:
      cwt: (N, 3, 32, 300)  2D 모델 입력
      1d : (N, 2, 300)      1D CNN 입력
      y  : (N,) 피험자 라벨,  g: (N,) 윈도우가 속한 시행(파일) 번호
    """
    cwt, raw, y, g = [], [], [], []
    for gi, f in enumerate(files):
        wn = minmax(make_windows(preprocess(load(f))))
        cwt.append(np.stack([to_cwt(w) for w in wn]))
        raw.append(wn.transpose(0, 2, 1).astype(np.float32))
        y += [label_of(f)] * len(wn)
        g += [gi] * len(wn)
    return {'cwt': np.concatenate(cwt), '1d': np.concatenate(raw),
            'y': np.array(y), 'g': np.array(g)}


def holdout_split(files, test_size=0.2, seed=SEED):
    """시행 번호를 8:2로 층화 분할 -> (train_file_ids, test_file_ids)"""
    ids = np.arange(len(files))
    return train_test_split(ids, test_size=test_size, stratify=[label_of(f) for f in files],
                            random_state=seed)


def cv_splits(files, n_splits=5, seed=SEED):
    """시행 단위 층화 5-fold -> [(train_file_ids, val_file_ids), ...]"""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(files)), [label_of(f) for f in files]))


def windows_of(g, file_ids):
    """시행 번호 목록에 속한 윈도우의 인덱스."""
    return np.where(np.isin(g, file_ids))[0]


def assert_trial_level(train_ids, test_ids):
    """같은 시행이 train/test 양쪽에 들어가지 않았음을 검사 (데이터 누수 방지)."""
    overlap = set(np.asarray(train_ids).tolist()) & set(np.asarray(test_ids).tolist())
    assert not overlap, f'시행 단위 분할 위반: {sorted(overlap)}'


if __name__ == '__main__':
    files = list_files()
    print('files:', len(files))
    tr, te = holdout_split(files)
    assert_trial_level(tr, te)
    d = build_all([files[i] for i in range(3)])
    print({k: v.shape for k, v in d.items()})
