"""모델 정의: 제안 모델(DenseNet161)과 베이스라인(ResNet18, 단순 2D CNN, 1D CNN, 파라미터 수를 논문에 맞춘 1D/2D CNN)."""
import torch.nn as nn
from torchvision.models import densenet161, resnet18

N_CLASSES = 5

# 모델별 입력 종류: 'cwt' = CWT 이미지 (3, 32, 300), '1d' = 원신호 (2, 300)
MODEL_INPUT = {'densenet161': 'cwt', 'resnet18': 'cwt', 'cnn2d': 'cwt', 'cnn1d': '1d',
               'cnn1d_pm': '1d', 'cnn2d_pm': 'cwt'}
MODEL_LABEL = {'densenet161': 'DenseNet161 (proposed)', 'resnet18': 'ResNet18',
               'cnn2d': 'Simple 2D CNN', 'cnn1d': '1D CNN',
               'cnn1d_pm': '1D CNN (params = paper)', 'cnn2d_pm': '2D CNN (params = paper)'}

# 논문 Table 4의 학습 가능 파라미터 수. 논문은 1D/2D CNN의 구조를 적지 않았으므로 구조는 달라도 크기만 맞춘다.
PAPER_PARAMS = {'cnn1d_pm': 9_589, 'cnn2d_pm': 4_944_901}


def build_densenet161():
    model = densenet161(weights=None)
    model.classifier = nn.Linear(2208, N_CLASSES)
    return model


def build_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, N_CLASSES)
    return model


def build_cnn2d():
    """강의자료의 베이스라인 A: 합성곱 2층짜리 단순 2D CNN."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, N_CLASSES))


class CNN1D(nn.Module):
    """CWT 없이 정규화된 원신호 (2, 300)를 그대로 입력받는 1차원 합성곱망. channels로 폭을 조절한다."""

    def __init__(self, in_ch=2, n_classes=N_CLASSES, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, c1, 7, padding=3), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1), nn.BatchNorm1d(c3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c3, c4, 3, padding=1), nn.BatchNorm1d(c4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(c4, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CNN2DPM(nn.Module):
    """파라미터 수를 논문의 2D CNN(4,944,901개)에 맞춘 2D CNN. 입력 (3, 32, 300).

    합성곱 3층(+풀링) 뒤 완전연결층 2개. 4,944,901개 중 대부분이 첫 완전연결층에 있다.
    """

    def __init__(self, channels=(80, 110, 130), hidden=246, n_classes=N_CLASSES):
        super().__init__()
        a, b, c = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, a, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(b, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))      # (c, 4, 37)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c * 4 * 37, hidden), nn.ReLU(),
                                        nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


_BUILDERS = {'densenet161': build_densenet161, 'resnet18': build_resnet18,
             'cnn2d': build_cnn2d, 'cnn1d': CNN1D,
             'cnn1d_pm': lambda: CNN1D(channels=(17, 24, 33, 44)), 'cnn2d_pm': CNN2DPM}


def build_model(name):
    return _BUILDERS[name]()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch
    for name, kind in MODEL_INPUT.items():
        shape = (2, 3, 32, 300) if kind == 'cwt' else (2, 2, 300)
        m = build_model(name).eval()
        n = count_params(m)
        note = f'  (논문 {PAPER_PARAMS[name]:,}: {"일치" if n == PAPER_PARAMS[name] else "불일치"})' if name in PAPER_PARAMS else ''
        print(f'{name:10s} out={tuple(m(torch.randn(*shape)).shape)} params={n:,}{note}')
