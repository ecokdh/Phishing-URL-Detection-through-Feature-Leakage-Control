"""S3 피처를 url_stripped 기준으로 다시 계산한다 (누수 통제).

왜 필요한가
-----------
정상 URL은 100%가 "https://www."로 시작하지만 피싱은 2.4%뿐이다. 원본 S3 피처 중 일부는
이 앞부분을 포함해서 계산돼 있어서, 모델이 URL 내용이 아니라 "수집 형태"를 학습할 수 있다.

원본 S3 18개를 확인한 결과 (validate()로 재현 가능)
-----------------------------------------------------
앞부분 영향 O (여기서 다시 계산하는 이유)
    URLLength                 = len(URL)         -> "https://www." 12자 포함
    DomainLength              = len(Domain)      -> "www." 4자 포함
    NoOfSubDomain             = Domain의 점 수 - 1 -> "www"를 서브도메인 1개로 셈
    LetterRatioInURL 등 비율 4개 = 개수 / URLLength -> 분모에 앞부분 포함
앞부분 영향 X (원본도 scheme, www.를 빼고 계산돼 있음. 같은 정의로 다시 계산만 한다)
    NoOfLettersInURL, NoOfDegitsInURL, NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
    NoOfOtherSpecialCharsInURL, HasObfuscation, NoOfObfuscatedChar,
    CharContinuationRate (www.와 TLD를 뗀 도메인 기준), TLDLength, IsDomainIP

정의 (입력 s = url_stripped: scheme, 앞 www., 끝 슬래시를 뗀 문자열)
-------------------------------------------------------------------
host        : s에서 첫 '/', '?', '#' 앞부분
URLLength   : len(s)
DomainLength: len(host)
NoOfSubDomain: host의 점 개수 - 1 (0 미만은 0) — 원본과 같은 방식
문자 개수    : s에서 센다. 특수문자 = 영문자·숫자가 아니고 '=', '?', '&'도 아닌 문자
NoOfObfuscatedChar: %XX 인코딩 1개당 3자, HasObfuscation = 그 값 > 0
비율 4개     : 개수 / URLLength (원본과 같은 식, 분모만 새 길이)
CharContinuationRate: host에서 끝의 ".TLD"를 뗀 문자열에서
              (가장 긴 영문자 연속 + 가장 긴 숫자 연속 + 가장 긴 특수문자 연속) / 길이
TLDLength, IsDomainIP: 앞부분과 무관해서 원본 정의 그대로

출력
----
data/processed/s3_stripped.csv : uid + 피처 18개 (열 이름 뒤에 "_s")
data/splits/features.json      : "S3_stripped" 키 추가 (02 노트북을 다시 돌리면 지워지므로 이 스크립트도 다시 실행)

실행 (저장소 최상위에서)
------------------------
    python -m src.s3_stripped            # 계산 + 저장 + 검증 결과 출력
    python -m src.s3_stripped --check    # 저장 없이 검증만
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from src.data import DATASET, FEATURES, ROOT

OUT = ROOT / "data" / "processed" / "s3_stripped.csv"
SUFFIX = "_s"

S3_COLS = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
]
# 앞부분("https://www.")이 값에 섞여 있던 피처
PREFIX_AFFECTED = ["URLLength", "DomainLength", "NoOfSubDomain", "LetterRatioInURL",
                   "DegitRatioInURL", "SpacialCharRatioInURL", "ObfuscationRatio"]

_HOST_END = re.compile(r"[/?#]")
_OBF = re.compile(r"%[0-9a-fA-F]{2}")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$")


def _host(s: str) -> str:
    return _HOST_END.split(s, maxsplit=1)[0]


def _continuation_rate(s: str) -> float:
    if not s:
        return 0.0
    best = {"a": 0, "d": 0, "o": 0}
    cur, run = None, 0
    for ch in s:
        kind = "a" if ch.isalpha() else "d" if ch.isdigit() else "o"
        run = run + 1 if kind == cur else 1
        cur = kind
        best[kind] = max(best[kind], run)
    return sum(best.values()) / len(s)


def url_features(s: str, tld: str) -> dict:
    """url_stripped 문자열 하나에서 S3 피처 18개를 계산한다."""
    host = _host(s)
    tld = "" if pd.isna(tld) else str(tld)
    core = host[: -(len(tld) + 1)] if tld and host.endswith("." + tld) else host

    n = len(s)
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    special = sum((not ch.isalnum()) and ch not in "=?&" for ch in s)
    obf = 3 * len(_OBF.findall(s))
    ratio = (lambda k: round(k / n, 3)) if n else (lambda k: 0.0)

    return {
        "URLLength": n,
        "DomainLength": len(host),
        "IsDomainIP": int(bool(_IPV4.match(host))),
        "CharContinuationRate": _continuation_rate(core),
        "TLDLength": len(tld),
        "NoOfSubDomain": max(host.count(".") - 1, 0),
        "HasObfuscation": int(obf > 0),
        "NoOfObfuscatedChar": obf,
        "ObfuscationRatio": ratio(obf),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": ratio(letters),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": ratio(digits),
        "NoOfEqualsInURL": s.count("="),
        "NoOfQMarkInURL": s.count("?"),
        "NoOfAmpersandInURL": s.count("&"),
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL": ratio(special),
    }


def compute(df: pd.DataFrame) -> pd.DataFrame:
    rows = [url_features(s, t) for s, t in zip(df["url_stripped"], df["TLD"])]
    out = pd.DataFrame(rows, index=df.index)[S3_COLS]
    out.columns = [c + SUFFIX for c in S3_COLS]
    out.insert(0, "uid", df["uid"].to_numpy())
    return out


def validate(df: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """원본과 새 값을 비교한다.

    - 앞부분 영향 X 피처: 원본과 일치율이 높아야 정의가 같다는 근거가 된다.
    - 앞부분 영향 O 피처: 정상(0)과 피싱(1)의 평균이 어떻게 바뀌었는지 본다.
    """
    rows = []
    y = df["is_phishing"]
    for c in S3_COLS:
        old, nw = df[c].astype(float), new[c + SUFFIX].astype(float)
        tol = 1e-3 if old.dtype == float and (old % 1 != 0).any() else 0
        rows.append({
            "feature": c,
            "prefix_affected": c in PREFIX_AFFECTED,
            "match_rate": float((np.abs(old - nw) <= tol + 1e-9).mean()),
            "old_mean_legit": old[y == 0].mean(), "old_mean_phish": old[y == 1].mean(),
            "new_mean_legit": nw[y == 0].mean(), "new_mean_phish": nw[y == 1].mean(),
        })
    return pd.DataFrame(rows).set_index("feature").round(3)


def save(new: pd.DataFrame) -> None:
    new.to_csv(OUT, index=False)
    with open(FEATURES, encoding="utf-8") as f:
        feats = json.load(f)
    feats["S3_stripped"] = [c + SUFFIX for c in S3_COLS]
    feats["S3_stripped_note"] = ("S3 18개를 url_stripped 기준으로 다시 계산 (src/s3_stripped.py). "
                                 "앞부분 영향 피처: " + ", ".join(PREFIX_AFFECTED))
    with open(FEATURES, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="저장하지 않고 검증 표만 출력")
    a = ap.parse_args()

    df = pd.read_csv(DATASET, encoding="utf-8")
    new = compute(df)
    pd.set_option("display.width", 200)
    print(validate(df, new).to_string())
    if not a.check:
        save(new)
        print(f"\n저장: {OUT.relative_to(ROOT)} ({len(new):,}행), features.json에 'S3_stripped' 추가")


if __name__ == "__main__":
    main()
