"""고전 ML 실험 (트랙 A): LR, DT, RF, MLP.

실행 예 (저장소 최상위에서)
---------------------------
빠른 동작 확인 (학습 데이터 2만 행만 사용, 1~2분):
    python -m src.classic_ml --quick

S1을 random, domain 분할에서 시드 3개로 전체 실행:
    python -m src.classic_ml --scenarios S1 --splits random domain

전부 실행 (S1, S2, S3 x random, domain):
    python -m src.classic_ml --scenarios S1 S2 S3 --splits random domain

시나리오 (피처 목록은 모두 features.json에서 읽는다)
------------------------------------------------
S1 : 전체 정형 피처
S2 : 누수 후보 제거 (11/18 확정 전에는 초안)
S3 : URL에서 직접 계산되는 정형 피처 (참조 비교 피처 3개와 IsHTTPS 제외)
S4 (URL 문자열)는 딥러닝 트랙용이라 여기서는 다루지 않는다.

다시 실행
---------
이미 끝난 (시나리오, 분할, 모델, 시드)는 건너뛴다.
features.json의 피처 목록이 바뀌면 (예: 11/18 S2 확정) 그 조합만 자동으로 다시 튜닝, 학습한다.
--overwrite를 붙이면 전부 다시 한다. 어느 경우든 metrics.csv의 옛 줄은 새 결과로 바뀐다.

절차
----
1. 튜닝: train 안에서 d.cv_fold(도메인 단위 5-fold)로 PredefinedSplit + GridSearchCV.
   시간을 줄이기 위해 train에서 최대 TUNE_ROWS행을 층화 표본추출해 튜닝한다.
2. 최종 학습: 고른 하이퍼파라미터로 train 전체에 시드별로 다시 학습한다.
3. 평가: val, test 모두 지표를 저장한다. 보고용 최종 성능은 test.
   표준화는 Pipeline 안에서 train으로만 계산한다.

출력 (results/classic_ml/)
--------------------------
metrics.csv          : 실행 1회당 val, test 지표 한 줄씩
best_params.json     : (시나리오, 분할, 모델)별 고른 하이퍼파라미터와 CV 점수
preds/*.csv          : uid, part(val/test), y_true, proba  (4번 평가 담당 공유용)
importance/*.csv     : RF, DT 피처 중요도, LR 계수
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import ROOT, load_dataset, load_split

OUT = ROOT / "results" / "classic_ml"
TUNE_ROWS = 40_000          # 튜닝에 쓸 train 표본 크기 (None이면 전체)
SCORING = "average_precision"  # 튜닝 기준: 피싱 클래스 PR-AUC


# ---------------------------------------------------------------------------
# 모델과 탐색 범위
# ---------------------------------------------------------------------------
def make_model(name: str, seed: int, n_jobs: int = 1):
    if name == "lr":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=3000, random_state=seed))])
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "dt":
        est = DecisionTreeClassifier(random_state=seed)
        grid = {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 10, 50]}
    elif name == "rf":
        est = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs)
        grid = {"max_depth": [None, 20], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 5]}
    elif name == "mlp":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", MLPClassifier(early_stopping=True, max_iter=200,
                                              random_state=seed))])
        grid = {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-3]}
    else:
        raise ValueError(name)
    return est, grid


MODELS = ["lr", "dt", "rf", "mlp"]


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def get_split(df, scenario: str, split: str):
    if scenario not in ("S1", "S2", "S3"):
        raise ValueError("고전 ML은 S1, S2, S3만 다룬다 (S4는 URL 문자열)")
    d = load_split(scenario, split, df=df)
    if not isinstance(d.X_train, pd.DataFrame):
        raise SystemExit(f"{scenario}가 정형 피처가 아니라 URL 문자열로 불러와졌습니다. "
                         "EDA 브랜치의 src/data.py와 S3가 들어간 features.json이 필요합니다.")
    return d


def tune_subset(d, n: int | None, seed: int = 0):
    """튜닝용 train 표본. fold 번호를 그대로 유지해서 도메인 단위 CV가 깨지지 않는다."""
    idx = np.arange(len(d.y_train))
    if n is not None and n < len(idx):
        rng = np.random.default_rng(seed)
        y = d.y_train.to_numpy()
        take = []
        for c in (0, 1):
            ci = idx[y == c]
            k = int(round(n * len(ci) / len(idx)))
            take.append(rng.choice(ci, size=k, replace=False))
        idx = np.sort(np.concatenate(take))
    return d.X_train.iloc[idx], d.y_train.iloc[idx], d.cv_fold.iloc[idx].to_numpy()


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def metrics(y, proba, thr: float = 0.5) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def importance(model, name: str, cols: list[str]) -> pd.DataFrame | None:
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if name in ("rf", "dt"):
        vals = clf.feature_importances_
        kind = "importance"
    elif name == "lr":
        vals = clf.coef_.ravel()   # 표준화된 피처 기준 계수 (양수 = 피싱 쪽)
        kind = "coef"
    else:
        return None
    out = pd.DataFrame({"feature": cols, kind: vals})
    return out.reindex(out[kind].abs().sort_values(ascending=False).index)


def save_metrics(path: Path, new: pd.DataFrame) -> None:
    """같은 (시나리오, 분할, 모델, 시드) 줄이 이미 있으면 지우고 새 줄로 바꾼다.
    다시 실행해도 metrics.csv에 옛 결과가 남아 평균이 섞이지 않게 하기 위함."""
    keys = ["scenario", "split", "model", "seed"]
    if path.exists():
        old = pd.read_csv(path)
        hit = old[keys].merge(new[keys].drop_duplicates(), how="left", indicator=True)["_merge"]
        old = old[(hit == "left_only").to_numpy()]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="고전 ML 실험 (LR, DT, RF, MLP)")
    ap.add_argument("--scenarios", nargs="+", default=["S1"], choices=["S1", "S2", "S3"])
    ap.add_argument("--splits", nargs="+", default=["domain"],
                    choices=["random", "domain", "domain_public"])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tune-rows", type=int, default=TUNE_ROWS,
                    help="튜닝에 쓸 train 표본 크기 (0이면 전체)")
    ap.add_argument("--quick", action="store_true",
                    help="동작 확인용: train 2만 행, 튜닝 1만 행, 시드 1개")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 결과도 다시 실행")
    ap.add_argument("--n-jobs", type=int, default=-1, help="사용할 CPU 코어 수 (-1 = 전부)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    tune_rows = None if a.tune_rows == 0 else a.tune_rows
    seeds = a.seeds
    out = a.out
    if a.quick:
        tune_rows, seeds, out = 10_000, seeds[:1], OUT.parent / "classic_ml_quick"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "importance").mkdir(parents=True, exist_ok=True)
    metrics_path, params_path = out / "metrics.csv", out / "best_params.json"
    best_params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}

    print("데이터 불러오는 중...")
    df = load_dataset()

    for scen in a.scenarios:
        for split in a.splits:
            d = get_split(df, scen, split)
            if a.quick:
                keep = np.random.default_rng(0).choice(len(d.y_train), 20_000, replace=False)
                keep.sort()
                d.X_train, d.y_train = d.X_train.iloc[keep], d.y_train.iloc[keep]
                d.cv_fold, d.groups = d.cv_fold.iloc[keep], d.groups.iloc[keep]
            cols = list(d.X_train.columns)
            label = scen
            print(f"\n=== {label} / {split}: 피처 {len(cols)}개, "
                  f"train {len(d.y_train):,} / val {len(d.y_val):,} / test {len(d.y_test):,}")

            for name in a.models:
                key = f"{label}|{split}|{name}"
                # features.json이 바뀌어 피처 목록이 달라졌으면 (예: 11/18 S2 확정) 자동으로 다시 실행
                saved = best_params.get(key, {}).get("features")
                changed = saved is not None and saved != cols
                if changed:
                    print(f"[{name}] 피처 목록이 바뀌어서 다시 튜닝하고 다시 학습합니다")
                redo = a.overwrite or changed
                todo = [s for s in seeds if redo or not
                        (out / "preds" / f"{label}_{split}_{name}_seed{s}.csv").exists()]
                if not todo:
                    print(f"[{name}] 이미 완료, 건너뜀")
                    continue

                # 1) 튜닝 (조합당 한 번, 시드 0)
                if key not in best_params or redo:
                    Xt, yt, folds = tune_subset(d, tune_rows)
                    est, grid = make_model(name, seed=0, n_jobs=1)
                    gs = GridSearchCV(est, grid, scoring=SCORING, cv=PredefinedSplit(folds),
                                      n_jobs=a.n_jobs, refit=False)
                    t0 = time.perf_counter()
                    gs.fit(Xt, yt)
                    best_params[key] = {"params": gs.best_params_,
                                        "cv_" + SCORING: round(float(gs.best_score_), 6),
                                        "tune_rows": len(yt),
                                        "tune_sec": round(time.perf_counter() - t0, 1),
                                        "features": cols}
                    params_path.write_text(json.dumps(best_params, indent=2, ensure_ascii=False,
                                                      default=str), encoding="utf-8")
                    print(f"[{name}] 튜닝 {best_params[key]['tune_sec']}초 -> {gs.best_params_}")
                params = best_params[key]["params"]
                if name == "mlp" and "clf__hidden_layer_sizes" in params:
                    params["clf__hidden_layer_sizes"] = tuple(params["clf__hidden_layer_sizes"])

                # 2) 시드별 최종 학습과 평가
                for seed in todo:
                    model, _ = make_model(name, seed=seed, n_jobs=a.n_jobs)
                    model.set_params(**params)
                    t0 = time.perf_counter()
                    model.fit(d.X_train, d.y_train)
                    fit_sec = time.perf_counter() - t0

                    rows, pred_parts = [], []
                    for part, X, y, uid in [("val", d.X_val, d.y_val, d.uid["val"]),
                                            ("test", d.X_test, d.y_test, d.uid["test"])]:
                        t0 = time.perf_counter()
                        proba = model.predict_proba(X)[:, 1]
                        pred_sec = time.perf_counter() - t0
                        rows.append({"scenario": label, "split": split, "model": name,
                                     "seed": seed, "part": part, "n_features": len(cols),
                                     **metrics(y.to_numpy(), proba),
                                     "fit_sec": round(fit_sec, 2),
                                     "predict_sec": round(pred_sec, 3)})
                        pred_parts.append(pd.DataFrame({"uid": uid, "part": part,
                                                        "y_true": y.to_numpy(), "proba": proba}))

                    # 지표를 먼저 저장하고 예측 파일을 마지막에 쓴다.
                    # 예측 파일이 있으면 "완료"로 보고 건너뛰므로, 중간에 끊겨도 지표가 빠지지 않는다.
                    m = pd.DataFrame(rows)
                    save_metrics(metrics_path, m)
                    imp = importance(model, name, cols)
                    if imp is not None and seed == todo[0]:
                        imp.to_csv(out / "importance" / f"{label}_{split}_{name}.csv", index=False)
                    pd.concat(pred_parts).to_csv(
                        out / "preds" / f"{label}_{split}_{name}_seed{seed}.csv", index=False)
                    t = m[m.part == "test"].iloc[0]
                    print(f"[{name}] seed {seed}: 학습 {fit_sec:.1f}초 | test F1 {t.f1:.4f} "
                          f"PR-AUC {t.pr_auc:.4f} Recall {t.recall:.4f} FPR {t.fpr:.4f}")

    print(f"\n완료. 결과 위치: {out}")
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        s = (m[m.part == "test"].groupby(["scenario", "split", "model"])
             [["f1", "pr_auc", "recall", "fpr"]].agg(["mean", "std"]).round(4))
        print(s.to_string())


if __name__ == "__main__":
    main()
