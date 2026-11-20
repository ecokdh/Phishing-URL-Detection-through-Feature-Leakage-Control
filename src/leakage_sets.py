"""S2A, S2B, S2C (누수 후보 제거 단계별) 피처 목록을 features.json에 추가한다.

11/18 팀 결정: 최종 S2는 S2B. S2A는 S1 -> S2A -> S2B 단계별 비교(ablation)용으로 함께 둔다.
11/20 추가: S2C = S2B에서 NoOfImage, NoOfJS, NoOfCSS까지 제거 (S2B 결과의 RF 중요도 1~3위).
    피싱의 약 69%가 이미지, JS, CSS가 모두 0인 빈 페이지(정상 0%)라 "빈 페이지면 피싱"을 배울 수 있다.
기존 "S2" 키는 02 노트북이 만든 초안(= S2A)이라 이전 결과와의 호환을 위해 그대로 둔다.

제거 기준 (01 노트북 8절과 같은 기준)
-----------------------------------
A : 다른 URL이나 참조 집합과 비교해 계산된 값 (라벨 정보가 직접 섞임)
B : 수집 과정이나 페이지 구성 방식 때문에 라벨과 직접 연결될 수 있는 값
    (정상은 전부 "https://www." 메인 페이지, 피싱은 단일 페이지 위주로 수집됨)

출력
----
data/splits/features.json 에 추가하는 키
    S2A, S2B, S2C        : 학습에 쓸 피처 목록 (S1에서 제거 목록을 뺀 것)
    S2A_removed, S2B_removed, S2C_removed : 각각 제거한 피처 (누적)
    S2_rationale         : 피처별 {stage, criterion, reason}
02 노트북을 다시 돌려 features.json을 새로 만들면 이 키들이 지워지므로 이 스크립트도 다시 실행한다.

실행 (저장소 최상위에서)
------------------------
    python -m src.leakage_sets           # 근거 표 출력 + features.json에 추가
    python -m src.classic_ml --scenarios S1 S2A S2B S2C
    python -m src.leakage_sets --check   # 저장 없이 근거 표만
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data import DATASET, FEATURES

# (피처, 단계, 기준, 근거)
RATIONALE = [
    ("URLSimilarityIndex",   "A", "A", "정상 URL 참조 집합과의 유사도. 정상은 전부 100 (참조 집합에서 온 URL)"),
    ("TLDLegitimateProb",    "A", "A", "정상 URL 집합에서 계산한 TLD 출현 확률 (라벨로 만든 통계)"),
    ("URLCharProb",          "A", "A", "정상/피싱 URL 문자 빈도에서 계산한 확률 (라벨로 만든 통계)"),
    ("LineOfCode",           "A", "B", "페이지 소스 줄 수. 정상은 메인 페이지 전체, 피싱은 짧은 단일 페이지라 페이지 종류 차이가 드러남"),
    ("LargestLineLength",    "A", "B", "페이지 소스 최장 줄 길이. LineOfCode와 같은 이유"),
    ("HasSocialNet",         "A", "B", "SNS 링크 유무. 정상 사이트 메인 페이지 구성 방식과 직결"),
    ("IsHTTPS",              "B", "B", "수집 형태만 반영. 정상은 100% https://www. (11/17 팀 논의)"),
    ("CharContinuationRate", "B", "B", "www.와 TLD를 뗀 도메인에서 (최장 영문자 + 숫자 + 특수문자 연속) / 길이. "
                                       "정상 수집 대상이 단일 단어 브랜드 도메인이라 대부분 1.0"),
    ("HasCopyrightInfo",     "B", "B", "저작권 문구 유무. 메인 페이지 하단 구성과 직결"),
    ("HasDescription",       "B", "B", "meta description 유무. 메인 페이지 구성과 직결"),
    ("NoOfExternalRef",      "B", "B", "외부 링크 수. 페이지 크기(메인 페이지 vs 단일 페이지)에 비례"),
    ("NoOfSelfRef",          "B", "B", "자기 링크 수. 페이지 크기에 비례"),
    ("NoOfImage",            "C", "B", "이미지 수. 피싱의 약 69%가 이미지·JS·CSS 모두 0인 빈 페이지(수집 시점에 삭제·에러 추정), 정상 0%"),
    ("NoOfJS",               "C", "B", "JS 파일 수. NoOfImage와 같은 이유 (정상 중앙값 14, 피싱 0)"),
    ("NoOfCSS",              "C", "B", "CSS 파일 수. NoOfImage와 같은 이유 (정상 중앙값 6, 피싱 0)"),
]
S2A_REMOVE = [f for f, stage, _, _ in RATIONALE if stage == "A"]
S2B_REMOVE = [f for f, stage, _, _ in RATIONALE if stage in ("A", "B")]
S2C_REMOVE = [f for f, _, _, _ in RATIONALE]


def single_feature_acc(x: pd.Series, y: pd.Series) -> float:
    """피처 하나에 임계값 하나만 써서 낼 수 있는 최고 정확도 (분위수 임계값, 양방향)."""
    xs = x.to_numpy(dtype=float)
    yv = y.to_numpy()
    best = max(yv.mean(), 1 - yv.mean())
    for t in np.unique(np.quantile(xs, np.linspace(0.01, 0.99, 99))):
        acc = ((xs > t) == yv).mean()
        best = max(best, acc, 1 - acc)
    return float(best)


def evidence(df: pd.DataFrame) -> pd.DataFrame:
    y = df["is_phishing"]
    rows = []
    for f, stage, crit, _ in RATIONALE:
        rows.append({
            "feature": f, "stage": f"S2{stage}", "criterion": crit,
            "median_legit": df.loc[y == 0, f].median(), "median_phish": df.loc[y == 1, f].median(),
            "single_feature_acc": single_feature_acc(df[f], y),
        })
    return pd.DataFrame(rows).set_index("feature").round(3)


def save() -> None:
    with open(FEATURES, encoding="utf-8") as f:
        feats = json.load(f)
    s1 = feats["S1"]
    missing = [c for c in S2C_REMOVE if c not in s1]
    if missing:
        raise SystemExit(f"S1에 없는 피처가 있습니다: {missing}")
    feats["S2A"] = [c for c in s1 if c not in S2A_REMOVE]
    feats["S2B"] = [c for c in s1 if c not in S2B_REMOVE]
    feats["S2C"] = [c for c in s1 if c not in S2C_REMOVE]
    feats["S2A_removed"] = S2A_REMOVE
    feats["S2B_removed"] = S2B_REMOVE
    feats["S2C_removed"] = S2C_REMOVE
    feats["S2_rationale"] = {f: {"stage": f"S2{s}", "criterion": c, "reason": r} for f, s, c, r in RATIONALE}
    feats["S2_status"] = ("11/18 팀 결정: 최종 S2 = S2B. S2A는 단계별 비교용. 11/20 S2C 추가 (빈 페이지 피처 제거). "
                          "'S2' 키는 초안(= S2A)으로 호환용")
    with open(FEATURES, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    print(f"\nfeatures.json에 추가: S2A {len(feats['S2A'])}개 (제거 {len(S2A_REMOVE)}), "
          f"S2B {len(feats['S2B'])}개 (제거 {len(S2B_REMOVE)}), S2C {len(feats['S2C'])}개 (제거 {len(S2C_REMOVE)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="저장하지 않고 근거 표만 출력")
    a = ap.parse_args()
    df = pd.read_csv(DATASET, encoding="utf-8", usecols=S2C_REMOVE + ["is_phishing"])
    pd.set_option("display.width", 200)
    print("제거 피처 근거 (single_feature_acc: 그 피처 하나로 임계값 하나만 써서 낸 최고 정확도)")
    print(evidence(df).to_string())
    if not a.check:
        save()


if __name__ == "__main__":
    main()
