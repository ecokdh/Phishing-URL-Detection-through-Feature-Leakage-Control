"""그룹을 유지하면서 클래스 비율을 맞추는 분할.

sklearn의 StratifiedGroupKFold와 같은 목적이지만, 그룹이 20만 개 수준이면 sklearn 구현
(그룹마다 모든 fold의 표준편차를 다시 계산)이 매우 느려 직접 구현한다.

방법
1. URL이 2개 이상인 그룹: 크기가 큰 순서로 (동점은 시드로 섞어서) 하나씩,
   넣었을 때 '클래스별 목표 대비 채운 비율'의 최댓값이 가장 작은 조각에 넣는다.
2. URL이 1개인 그룹: 클래스별로 섞은 뒤, 각 조각의 남은 목표량에 비례해 채운다.
   이 단계에서 클래스 비율이 거의 정확히 맞춰진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_group_chunks(groups, y, n_chunks: int, seed: int = 42) -> np.ndarray:
    """각 행이 들어갈 조각 번호(0..n_chunks-1)를 돌려준다. 같은 그룹은 항상 같은 조각."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)
    codes, _ = pd.factorize(pd.Series(groups), sort=False)
    n_groups, n_cls = codes.max() + 1, len(classes)

    cnt = np.zeros((n_groups, n_cls), dtype=np.int64)
    np.add.at(cnt, (codes, y_idx), 1)
    size = cnt.sum(axis=1)
    target = np.bincount(y_idx, minlength=n_cls) / n_chunks

    rng = np.random.default_rng(seed)
    load = np.zeros((n_chunks, n_cls))
    group_chunk = np.full(n_groups, -1, dtype=int)

    # 1) 여러 행짜리 그룹: 큰 것부터 가장 덜 찬 조각으로
    multi = np.where(size > 1)[0]
    multi = multi[rng.permutation(len(multi))]
    multi = multi[np.argsort(-size[multi], kind="stable")]
    for g in multi:
        fill = ((load + cnt[g]) / target).max(axis=1)
        best = np.flatnonzero(fill == fill.min())
        k = best[rng.integers(len(best))]
        group_chunk[g] = k
        load[k] += cnt[g]

    # 2) 한 행짜리 그룹: 클래스별로 남은 목표량에 비례해 배분
    single = np.where(size == 1)[0]
    single_cls = cnt[single].argmax(axis=1)
    for c in range(n_cls):
        gs = single[single_cls == c]
        gs = gs[rng.permutation(len(gs))]
        deficit = np.clip(target[c] - load[:, c], 0, None)
        if deficit.sum() == 0:
            deficit = np.ones(n_chunks)
        raw = deficit / deficit.sum() * len(gs)
        quota = np.floor(raw).astype(int)
        rest = len(gs) - quota.sum()
        quota[np.argsort(-(raw - quota), kind="stable")[:rest]] += 1
        group_chunk[gs] = np.repeat(np.arange(n_chunks), quota)
        load[:, c] += quota

    assert (group_chunk >= 0).all()
    return group_chunk[codes]
