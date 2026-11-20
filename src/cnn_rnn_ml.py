"""CNN, RNN 실험 (시나리오 S4): URL 문자열을 문자 단위로 넣는 GRU, LSTM, 얕은/깊은 1D-CNN.

실행 (저장소 최상위에서)
    아래 "실행 설정"의 변수를 고친 뒤
    python -m src.cnn_rnn_ml

    QUICK = True   -> 동작 확인 (train 2만 행, 2 epoch, 시드 0)
    QUICK = False  -> 본 실험: url_stripped(S4_stripped, 주) + 원본 URL(S4, 참고) 각 24회 = 48회
                      한쪽만 돌리려면 INPUTS에서 빼기

출력 (results/cnn_rnn/)
    metrics.csv       실행 1회당 val, test 한 줄씩 (classic_ml.py와 같은 열 + n_params, best_epoch)
    preds/*.csv       uid, part, y_true, proba  (평가 담당 공유용)
    checkpoints/*.pt  학습된 가중치 + 문자 사전 + 최대 길이 (가림 분석 때 모델을 다시 불러오기 위함)
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from src.classic_ml import metrics, save_metrics   # 고전 ML과 같은 지표 함수를 써야 비교가 공정하다
from src.data import PAD, ROOT, build_vocab, choose_max_len, encode_chars, load_dataset, load_split

# ===========================================================================
# 실행 설정: 여기만 고쳐서 돌린다
# ===========================================================================
QUICK = True                          # True: 동작 확인용 축소 실행 / False: 본 실험
INPUTS = ["url_stripped", "URL"]      # url_stripped(주 실험) -> URL(참고 실험) 순서로 둘 다 실행
SPLITS = ["random", "domain"]         # "domain_public"도 가능
MODELS = ["gru", "lstm", "cnn_shallow", "cnn_deep"]   # 일부만 돌리려면 여기서 빼기
SEEDS = [0, 1, 2]                     # 모델 시드 (분할 시드 42는 splits.csv에 이미 고정)
OUT = ROOT / "results" / "cnn_rnn"    # 결과 폴더 (Colab이면 Drive 경로로 바꾼다)
DEVICE = "auto"                       # "auto" / "cuda" / "mps" / "cpu" (맥에서 mps 오류가 나면 "cpu")

SCENARIO = {"URL": "S4", "url_stripped": "S4_stripped"}   # 입력 열 이름 -> 결과 파일에 쓸 시나리오 이름

# 하이퍼파라미터 (모든 모델 공통)
CFG = {
    "emb_dim": 64,       # 문자 하나를 몇 차원 벡터로 바꿀지
    "hidden": 128,       # RNN 은닉 상태 크기 (방향당)
    "channels": 128,     # CNN 필터 개수
    "kernel": 3,         # CNN 필터가 한 번에 보는 문자 수
    "dropout": 0.3,      # 학습 중 뉴런을 끄는 비율 (과적합 방지)
    "lr": 1e-3,          # 학습률
    "weight_decay": 1e-4,
    "batch_size": 256,   # 한 번에 학습하는 URL 수
    "max_epochs": 20,    # 최대 학습 횟수 (데이터 전체를 몇 바퀴 볼지)
    "patience": 3,       # val 성능이 이만큼 연속으로 안 좋아지면 조기 종료
}


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class CharRNN(nn.Module):
    """문자 임베딩 -> 양방향 GRU/LSTM -> 마지막 은닉 상태(앞, 뒤 방향) -> 출력 1개(logit)."""

    def __init__(self, vocab_size, cell, emb_dim, hidden, dropout, **_):
        super().__init__()
        self.cell = cell
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)   # 문자 번호 -> 64차원 벡터, PAD는 항상 0벡터
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)                              # 앞 방향 128 + 뒤 방향 128 -> 1

    def forward(self, x):                                   # x: (B, L) 정수
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()      # URL마다 PAD를 뺀 실제 길이
        # pack: RNN이 PAD 부분을 읽지 않게 한다 (뒤 방향 RNN이 PAD부터 읽기 시작하는 문제 방지)
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.rnn(packed)                             # h: (2, B, hidden) = [앞 방향 끝, 뒤 방향 끝]
        if self.cell == "lstm":                             # LSTM은 (h, c) 두 개를 돌려준다
            h = h[0]
        h = torch.cat([h[-2], h[-1]], dim=1)                # (B, 2*hidden)
        return self.out(self.drop(h)).squeeze(1)            # (B,)


class ConvBlock(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2 -> MaxPool. 패턴을 찾고 길이를 절반으로 줄인다."""

    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.net = nn.Sequential(
            # padding=k//2: 양 끝에 0을 붙여 Conv 뒤에도 길이가 그대로 유지되게 한다
            nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),                # 이웃 두 칸 중 큰 값만 -> 길이 절반 (홀수면 올림)
        )

    def forward(self, x):
        return self.net(x)


class CharCNN(nn.Module):
    """문자 임베딩 -> ConvBlock n개 -> 전역 최대 풀링 -> 출력 1개(logit).
    얕은 구조(1블록)와 깊은 구조(4블록)는 블록 수만 다르다."""

    def __init__(self, vocab_size, n_blocks, emb_dim, channels, kernel, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        blocks, c = [], emb_dim                             # c = 현재 채널 수 (처음엔 임베딩 차원 64)
        for _ in range(n_blocks):
            blocks.append(ConvBlock(c, channels, kernel))   # 첫 블록 64->128, 이후 128->128
            c = channels
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(channels, 1)

    def forward(self, x):                                   # x: (B, L)
        h = self.emb(x).transpose(1, 2)                     # (B, L, E) -> (B, E, L): Conv1d는 채널이 가운데
        h = self.blocks(h)                                  # (B, C, L'): 블록마다 길이 절반
        h = h.max(dim=2).values                             # (B, C): 필터마다 URL 전체에서 가장 강한 반응 하나
        return self.out(self.drop(h)).squeeze(1)            # (B,)


def build_model(name, vocab_size):
    """모델 이름 -> 모델 객체. 얕은/깊은 CNN은 같은 클래스에 블록 수만 다르게 준다."""
    if name in ("gru", "lstm"):
        return CharRNN(vocab_size, cell=name, **CFG)
    return CharCNN(vocab_size, n_blocks=1 if name == "cnn_shallow" else 4, **CFG)


# ---------------------------------------------------------------------------
# 학습과 예측
# ---------------------------------------------------------------------------
def set_seed(seed):
    """가중치 초기값, 배치 순서, 드롭아웃의 무작위를 고정한다 (같은 시드 = 같은 결과)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    """auto면 NVIDIA GPU(cuda) -> 맥 GPU(mps) -> CPU 순서로 고른다."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """GPU는 비동기로 돌아서, 시간을 재기 전에 앞의 계산이 끝날 때까지 기다린다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()                                            # 예측 때는 기울기 계산이 필요 없다 (메모리 절약)
def predict(model, X, batch_size=2048):
    """피싱일 확률(0~1)을 돌려준다."""
    model.eval()                                            # 드롭아웃 끄기, BatchNorm은 학습 때 통계 사용
    out = [torch.sigmoid(model(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.cat(out).float().cpu().numpy()


def fit(model, Xtr, ytr, Xva, yva, seed, max_epochs):
    """학습. 매 epoch 끝에 val PR-AUC를 보고 가장 좋았던 epoch의 가중치를 남긴다 (조기 종료)."""
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()                        # sigmoid + 이진 교차 엔트로피를 한 번에
    gen = torch.Generator().manual_seed(seed)               # 배치 순서용 난수
    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()                                       # 드롭아웃, BatchNorm을 학습 모드로
        t0 = time.perf_counter()
        perm = torch.randperm(len(Xtr), generator=gen).to(Xtr.device)   # 매 epoch마다 순서를 섞는다
        #데이터의 배치 크기에 맞춰서 배치만큼 학습
        for i in range(0, len(Xtr), CFG["batch_size"]):
            idx = perm[i:i + CFG["batch_size"]]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])       # 1) 예측하고 오차 계산
            opt.zero_grad()                                 # 2) 이전 기울기 지우기
            loss.backward()                                 # 3) 역전파로 기울기 계산
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # RNN 기울기 폭주 방지
            opt.step()                                      # 4) 가중치 갱신

        val_ap = average_precision_score(yva, predict(model, Xva))
        print(f"    epoch {epoch:2d} | val PR-AUC {val_ap:.4f} | {time.perf_counter() - t0:.1f}초")

        if val_ap > best_ap + 1e-5:                         # 좋아졌으면 지금 가중치를 복사해 둔다
            best_ap, best_epoch, bad = val_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:                                               # 안 좋아졌으면 카운트, patience번이면 멈춤
            bad += 1
            if bad >= CFG["patience"]:
                break

    model.load_state_dict(best_state)                       # 가장 좋았던 epoch의 가중치로 되돌린다
    return best_epoch


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    #학습시간은 device의 gpu에 따라 달라질 수 있어서 어떤 device를 사용하는지 알기 위해 추가
    device = pick_device(DEVICE)
    seeds, max_epochs, out = SEEDS, CFG["max_epochs"], Path(OUT)
    if QUICK:                                               # 축소판: 시드 1개, 2 epoch, 결과는 별도 폴더
        seeds, max_epochs, out = [0], 2, out.with_name(out.name + "_quick")
    #아래는 각각 예측과 정답을 추가 폴더에 생성해줘서 평가할때 보기 편하게함
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    #어떤 장치에서 결과를 어디에 저장할지 출력
    print(f"장치: {device} | 결과: {out}")

    df = load_dataset()                                     # dataset.csv + splits.csv (한 번만 읽고 재사용)

    for col in INPUTS:                                    # URL / url_stripped
        scen = SCENARIO[col]
        for split in SPLITS:                              # random / domain
            # 1) 데이터: URL 문자열과 라벨을 팀 공용 분할 그대로 받는다
            d = load_split("S4", split, url_col=col, df=df)
            if QUICK:                                       # train에서 2만 행만 무작위로
                keep = np.sort(np.random.default_rng(0).choice(len(d.y_train), 20_000, replace=False))
                d.X_train, d.y_train = d.X_train.iloc[keep], d.y_train.iloc[keep]

            # 2) 문자 -> 숫자: 사전과 최대 길이는 train으로만 정하고 val, test에 그대로 쓴다
            vocab = build_vocab(d.X_train)
            # cnn에 넣을때 최대 길이 기준으로 길이 맞춰주는 처리과정
            max_len = choose_max_len(d.X_train)
            X = {p: torch.from_numpy(encode_chars(getattr(d, f"X_{p}"), vocab, max_len)).long().to(device)
                 for p in ("train", "val", "test")}         # (n, max_len) 정수 텐서
            y = {p: getattr(d, f"y_{p}").to_numpy().copy() for p in ("train", "val", "test")}
            ytr = torch.tensor(y["train"], dtype=torch.float32, device=device)
            vocab_size = len(vocab) + 2                     # + PAD(0), UNK(1)
            print(f"\n=== {scen} / {split}: 문자 {len(vocab)}종, 최대 길이 {max_len}, "
                  f"train {len(ytr):,} / val {len(y['val']):,} / test {len(y['test']):,}")

            #4가지 모델에 대해 모델링 시작.
            for name in MODELS:                           # gru / lstm / cnn_shallow / cnn_deep
                for seed in seeds:                          # 0 / 1 / 2 정해진 값
                    tag = f"{scen}_{split}_{name}_seed{seed}" #어떤 출력인지 부연설명
                    print(f"[{tag}]")

                    # 3) 모델 만들고 학습 (학습 시간 측정)
                    set_seed(seed)
                    model = build_model(name, vocab_size).to(device)
                    #시간을 재기 위해서 동기로 동작시켜줘야함, gpu는 비동기로 동작
                    sync(device)
                    #현재 시각을 정밀하게 돌려준다고 함
                    t0 = time.perf_counter()
                    #학습진행을 fit 안에서. 어떤 모델인지 모델에 들어가는 train,val 데이터들, 해당 시점의 sedd값, 최대 에폭시 전달
                    best_epoch = fit(model, X["train"], ytr, X["val"], y["val"], seed, max_epochs)
                    sync(device)
                    fit_sec = time.perf_counter() - t0

                    # 4) val, test 예측과 지표 (추론 시간 측정)
                    rows, preds = [], []
                    for part in ("val", "test"):
                        sync(device)
                        t0 = time.perf_counter()
                        proba = predict(model, X[part])
                        sync(device)
                        pred_sec = time.perf_counter() - t0     # 예측만 걸린 시간 (지표 계산 전에 잰다)
                        rows.append({"scenario": scen, "split": split, "model": name, "seed": seed,
                                     "part": part, "n_features": max_len, **metrics(y[part], proba),
                                     "fit_sec": round(fit_sec, 2),
                                     "predict_sec": round(pred_sec, 3),
                                     "n_params": sum(p.numel() for p in model.parameters()),
                                     "best_epoch": best_epoch})
                        preds.append(pd.DataFrame({"uid": d.uid[part], "part": part,
                                                   "y_true": y[part], "proba": proba}))

                    # 5) 저장: 지표(같은 실행의 옛 줄은 교체), 예측, 체크포인트
                    save_metrics(out / "metrics.csv", pd.DataFrame(rows))
                    pd.concat(preds).to_csv(out / "preds" / f"{tag}.csv", index=False)
                    torch.save({"state_dict": model.state_dict(), "model": name, "vocab": vocab,
                                "max_len": max_len, "url_col": col}, out / "checkpoints" / f"{tag}.pt")
                    t = rows[1]
                    print(f"  -> test F1 {t['f1']:.4f} | PR-AUC {t['pr_auc']:.4f} | Recall {t['recall']:.4f} "
                          f"| FPR {t['fpr']:.4f} | 학습 {fit_sec:.1f}초")


if __name__ == "__main__":
    main()
