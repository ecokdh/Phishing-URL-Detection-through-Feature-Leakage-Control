"""팀 공용 데이터 불러오기 모듈.

사용 예
-------
>>> from src.data import load_split
>>> d = load_split(scenario="S2", split="domain")
>>> d.X_train.shape, d.y_train.mean()

- 라벨은 항상 `is_phishing` (피싱 = 1, 정상 = 0).
- 표준화 등 학습되는 전처리는 여기서 하지 않는다. 모델 쪽 sklearn Pipeline 안에서 train 통계로만 계산한다.
- Test는 최종 평가에만 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.csv"
SPLITS = ROOT / "data" / "splits" / "splits.csv"
FEATURES = ROOT / "data" / "splits" / "features.json"

SPLIT_COLUMNS = {
    "random": "split_random",
    "domain": "split_domain",               # etld1 (private suffix 포함, 소유자 단위)
    "domain_public": "split_domain_public", # etld1_public (플랫폼 단위)
}
CV_COLUMNS = {
    "random": "cv_fold_random",
    "domain": "cv_fold_domain",
    "domain_public": "cv_fold_domain_public",
}
GROUP_COLUMNS = {"random": "etld1", "domain": "etld1", "domain_public": "etld1_public"}
URL_COLUMNS = ("URL", "url_stripped")


@dataclass
class SplitData:
    X_train: pd.DataFrame | pd.Series
    y_train: pd.Series
    X_val: pd.DataFrame | pd.Series
    y_val: pd.Series
    X_test: pd.DataFrame | pd.Series
    y_test: pd.Series
    cv_fold: pd.Series   # train 행의 5-fold 번호 (GroupKFold 대용: PredefinedSplit 등에 사용)
    groups: pd.Series    # train 행의 도메인 그룹 키
    uid: dict            # {"train": Series, "val": Series, "test": Series}
    features: list[str] | str
    scenario: str
    split: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, y in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            rows.append({"part": name, "n": len(y), "phishing_rate": round(float(y.mean()), 4)})
        return pd.DataFrame(rows).set_index("part")


def load_features() -> dict:
    with open(FEATURES, encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> pd.DataFrame:
    """dataset.csv와 splits.csv를 uid로 합친 전체 표."""
    df = pd.read_csv(DATASET, encoding="utf-8")
    sp = pd.read_csv(SPLITS)
    out = df.merge(sp, on="uid", how="inner", validate="one_to_one")
    assert len(out) == len(df) == len(sp), "dataset.csv와 splits.csv의 uid가 일치하지 않습니다"
    return out


def load_split(scenario: str = "S1", split: str = "domain", url_col: str = "URL",
               df: pd.DataFrame | None = None) -> SplitData:
    """시나리오와 분할 방식에 맞는 train/val/test를 돌려준다.

    scenario : "S1" (전체 정형 피처) | "S2" (누수 후보 제거) | "S3" (URL 문자열)
    split    : "random" | "domain" | "domain_public"
    url_col  : S3에서 쓸 열. "URL" (원본) 또는 "url_stripped" (scheme, www., 끝 슬래시 제거)
    df       : load_dataset() 결과를 재사용할 때 넘긴다 (반복 호출 시 속도용)
    """
    if split not in SPLIT_COLUMNS:
        raise ValueError(f"split은 {list(SPLIT_COLUMNS)} 중 하나")
    feats = load_features()
    if scenario in ("S1", "S2"):
        cols: list[str] | str = feats[scenario]
    elif scenario == "S3":
        if url_col not in URL_COLUMNS:
            raise ValueError(f"url_col은 {URL_COLUMNS} 중 하나")
        cols = url_col
    else:
        raise ValueError("scenario는 S1, S2, S3 중 하나")

    if df is None:
        df = load_dataset()
    sc, cv, gc = SPLIT_COLUMNS[split], CV_COLUMNS[split], GROUP_COLUMNS[split]
    parts = {p: df[df[sc] == p] for p in ("train", "val", "test")}

    def X(part):
        return part[cols].reset_index(drop=True)

    def y(part):
        return part["is_phishing"].reset_index(drop=True)

    tr = parts["train"]
    return SplitData(
        X_train=X(tr), y_train=y(tr),
        X_val=X(parts["val"]), y_val=y(parts["val"]),
        X_test=X(parts["test"]), y_test=y(parts["test"]),
        cv_fold=tr[cv].astype(int).reset_index(drop=True),
        groups=tr[gc].reset_index(drop=True),
        uid={p: v["uid"].reset_index(drop=True) for p, v in parts.items()},
        features=cols, scenario=scenario, split=split,
    )


# ---------------------------------------------------------------------------
# S3: 문자 단위 인코딩 (프레임워크 무관, numpy 배열 반환)
# ---------------------------------------------------------------------------
PAD, UNK = 0, 1


def build_vocab(train_urls, min_freq: int = 1) -> dict[str, int]:
    """train URL에서만 문자 사전을 만든다. 0=PAD, 1=UNK."""
    counts: dict[str, int] = {}
    for u in train_urls:
        for ch in u:
            counts[ch] = counts.get(ch, 0) + 1
    chars = sorted(c for c, n in counts.items() if n >= min_freq)
    return {c: i + 2 for i, c in enumerate(chars)}


def choose_max_len(train_urls, q: float = 0.99) -> int:
    """train URL 길이의 q 분위수."""
    return int(np.ceil(np.quantile(pd.Series(train_urls).str.len(), q)))


def encode_chars(urls, vocab: dict[str, int], max_len: int) -> np.ndarray:
    """URL을 (n, max_len) 정수 배열로. 앞에서부터 max_len자만 쓰고 뒤를 PAD로 채운다."""
    urls = list(urls)
    out = np.full((len(urls), max_len), PAD, dtype=np.int32)
    for i, u in enumerate(urls):
        ids = [vocab.get(ch, UNK) for ch in u[:max_len]]
        out[i, : len(ids)] = ids
    return out
