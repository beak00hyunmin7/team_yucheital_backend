"""웅덩이 보정: 빠져나갈 곳이 없는 웅덩이의 물이 시간이 지나도 줄지 않게 붙잡는다.

왜 필요한가
    학습 데이터(OpenFOAM)는 지형 위 1 m 를 0.1 m 칸 10개로 나눈 거친 격자에 2~8 cm 물막을 깔고
    시작한다. 이 설정에서는 계산 중에 물 양이 보존되지 않아서, 가장자리로 넘쳐 나갈 수 없는
    웅덩이 안의 물도 18초 -> 120초 사이에 평균 35% 줄어든다 (웅덩이 있는 지형 297개 중 72%).
    시뮬레이션에는 침투·증발·배수관이 없으므로 물리적으로는 그대로 남아 있어야 한다.
    모델은 이 경향을 배워서, 막힌 곳이든 아니든 시간이 지나면 물을 전체적으로 줄인다.

무엇을 하나 (후처리, 모델은 그대로)
    1. 지형을 가장자리부터 채워(priority flood) '웅덩이'(넘쳐 나갈 높이보다 낮은 칸들의 덩어리)를 찾는다.
       가장자리는 출구로 본다 - 학습 시뮬레이션의 옆면이 열려 있는 것과 같다.
    2. 웅덩이마다 '갇힌 물 부피' = 18초부터 지금 시점까지 각 시점 예측에서 그 웅덩이에 담긴 물 부피 중
       가장 큰 값 (웅덩이 용량까지만). 웅덩이 물은 절반쯤이 18초 뒤에 가장 많아지므로(경사면 물이
       늦게 모임) 18초 하나로는 부족하다. 칸마다 최댓값을 따로 고르면 시점마다 젖은 칸이 달라
       없던 물이 생기므로(실측 +10%) 웅덩이 단위 부피로 센다.
    3. 지금 예측의 웅덩이 물이 그 부피보다 적으면, 그 웅덩이에 물이 가장 많았던 시점의 예측 모양을
       가져와 부피가 맞도록 채운다 (max(지금 예측, a x 그 시점 예측), 물 합 = 갇힌 물 부피).
       그래서 보정한 곳도 AI 예측처럼 깊은 골은 진하고 가장자리는 옅게 보인다. 칸은 올리기만 한다.
    4. 웅덩이 가장자리 몇 칸은 보정을 0 에서 서서히 키워(edge_weight) 보정 안 한 바깥과 끊김 없이 잇는다.
       덜 채운 몫은 안쪽이 조금 더 채워서 부피는 그대로 맞춘다.
    5. [11/28] 너무 작거나 얕은 웅덩이는 보정하지 않고, 작은 웅덩이는 시간이 지나며 조금씩 빠지다가
       다 빠지면 보정을 없앤다 (hold 설명).
    웅덩이 밖(경사면)과 웅덩이를 넘치는 물은 손대지 않는다 - 그 물은 실제로 빠질 수 있다.
    18초 이전은 아직 물이 모이는 중이라 두지 않는다.

    이미지 모드는 높이가 0..1 상대값이라 웅덩이 용량을 m 로 모른다 -> 용량 상한 없이 부피만 맞춘다.

끄기: 환경변수 YUCHE_POND_HOLD=0
"""
from __future__ import annotations

import heapq
import os

import numpy as np
from scipy import ndimage

ENABLED = os.environ.get("YUCHE_POND_HOLD", "1") != "0"

MIN_PIT_M = 0.01        # 고도(m) 격자: 이보다 얕은 칸은 웅덩이로 치지 않음 (1 cm)
MIN_PIT_REL = 0.01      # 상대 높이(0..1) 격자: 전체 높낮이의 1% 보다 얕은 칸은 무시
FEATHER_PX = 3.0        # 웅덩이 가장자리에서 보정이 0 -> 다 들어갈 때까지의 폭 (예측 그림 칸, 128칸 기준 약 25 m)
# [11/28] 작은 웅덩이 (hold 설명 참고)
POND_MIN_AREA = 0.002       # 지도 면적의 0.2% (1km 지도면 약 2,000 m²) 보다 작은 웅덩이는 보정하지 않는다
POND_MIN_DEPTH_M = 0.05     # 가장 깊은 곳이 5 cm 도 안 되는 웅덩이는 보정하지 않는다 (고도 격자)
POND_MIN_DEPTH_REL = 0.02   # 이미지 모드: 전체 높낮이의 2%
POND_AREA_REF = 0.01        # 지도 면적의 1%: 이 크기 웅덩이는 tau = POND_TAU_S 로 빠진다 (2% 이상이면 안 빠짐)
POND_TAU_S = 60.0
POND_GONE = 0.05            # 붙잡는 물이 최대의 5% 아래로 줄면 보정 없앰


def pit_depth(height: np.ndarray) -> np.ndarray:
    """각 칸의 웅덩이 깊이 (넘쳐 나갈 높이 - 지면). 가장자리로 물이 빠지는 칸은 0. (priority flood, 4방향)"""
    z = np.nan_to_num(np.asarray(height, dtype=np.float64))
    ny, nx = z.shape
    filled = z.copy()
    seen = np.zeros(z.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for i in range(ny):
        for j in (0, nx - 1) if 0 < i < ny - 1 else range(nx):
            seen[i, j] = True
            heap.append((z[i, j], i, j))
    heapq.heapify(heap)
    while heap:
        level, i, j = heapq.heappop(heap)
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < ny and 0 <= b < nx and not seen[a, b]:
                seen[a, b] = True
                filled[a, b] = max(z[a, b], level)
                heapq.heappush(heap, (filled[a, b], a, b))
    return filled - z


def pit_labels(pits: np.ndarray, metric: bool) -> tuple[np.ndarray, int]:
    """웅덩이 칸을 덩어리별로 번호 매긴다 (0 = 웅덩이 아님)."""
    return ndimage.label(pits > (MIN_PIT_M if metric else MIN_PIT_REL))


def edge_weight(pit: np.ndarray, radius: float = FEATHER_PX) -> np.ndarray:
    """웅덩이 가장자리 0 -> 안쪽 radius 칸부터 1 로 부드럽게 오르는 가중치 (smoothstep).
    보정한 물이 웅덩이 경계에서 뚝 끊기지 않고 보정 안 한 바깥 예측과 이어지게 한다."""
    dist = ndimage.distance_transform_edt(pit)          # 가장 가까운 웅덩이 밖 칸까지 거리 (가장자리 칸 = 1)
    x = np.clip((dist - 0.5) / max(radius, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _scale_to_volume(cur: np.ndarray, peak: np.ndarray, w: np.ndarray, cap: np.ndarray | None,
                     volume: float) -> np.ndarray:
    """웅덩이 안에서 cur + w*max(a*peak - cur, 0) 의 물 합이 volume 이 되는 a 를 찾아 그 수심을 돌려준다.
    peak = 물이 가장 많았던 시점의 예측 수심 -> 보정한 곳도 AI 예측과 같은 깊이 모양(깊은 골은 진하게)이 된다.
    w = 가장자리 가중치 -> 경계에서는 지금 예측 그대로, 안쪽으로 갈수록 보정이 다 들어간다.
    가장자리에서 덜 채운 만큼은 a 를 조금 키워 안쪽이 메운다 (웅덩이 용량 cap 은 넘지 않는다)."""
    def out(a):
        target = a * peak if cap is None else np.minimum(a * peak, cap)
        return cur + w * np.maximum(target - cur, 0.0)

    def vol(a):
        d = out(a)
        return float((np.minimum(d, cap) if cap is not None else d).sum())

    if vol(0.0) >= volume:
        return cur
    lo, hi = 0.0, 1.0
    while vol(hi) < volume and hi < 4.0:   # 가장자리 몫을 안쪽이 메우도록 최대 4배까지
        lo, hi = hi, hi * 2.0
    for _ in range(30):   # a 가 커지면 담긴 물도 는다
        mid = 0.5 * (lo + hi)
        if vol(mid) < volume:
            lo = mid
        else:
            hi = mid
    return out(hi)


def hold(depth_t: np.ndarray, refs: list[np.ndarray], pits: np.ndarray, metric: bool,
         times: list[float] | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """depth_t(지금 시점 예측)를 보정한 수심 · 요약 · '물을 붙잡은 웅덩이 칸' 마스크.
    refs = 18초~지금 시점 직전 예측들, times = refs + 지금 시점의 시각(초). 모두 같은 모양·방향.

    metric=True  : pits 가 m 단위 웅덩이 깊이 -> 웅덩이 용량을 넘는 물은 빠지는 것으로 본다
    metric=False : pits 가 0..1 상대 높이 (이미지 모드) -> 용량을 m 로 몰라 상한 없이 부피만 맞춘다

    [11/28 다시 검토] 작은 웅덩이까지 물을 끝까지 붙잡았더니, 다 마른 언덕 한가운데 작은 물웅덩이만 덩그러니
    남아 이상해 보였다 (128칸 격자로 줄인 지형의 작은 오목함도 웅덩이로 잡혔다). 그래서
      - 너무 작거나(지도의 POND_MIN_AREA 미만) 얕은(POND_MIN_DEPTH 미만) 웅덩이는 보정하지 않는다
      - 작은 웅덩이는 시간이 지나며 조금씩 빠진다: 붙잡는 부피 = 최대 부피 x exp(-(지금 - 최대 시점) / tau),
        tau = POND_TAU_S x (면적 / POND_AREA_REF)^1.5 (작을수록 빨리). POND_AREA_REF 의 2배 이상은 빠지지 않는다
      - 붙잡는 부피가 최대의 POND_GONE 아래로 줄면 그 웅덩이 보정은 없앤다 (모델 예측 그대로)
    """
    depth_t = np.asarray(depth_t, dtype=np.float64)
    frames = [np.asarray(r, dtype=np.float64) for r in refs] + [depth_t]
    if times is None or len(times) != len(frames):
        times = list(range(len(frames)))   # 시각을 모르면 빠짐 계산 없이 (테스트·예전 호출)
        decay = False
    else:
        decay = True
    t_now = float(times[-1])
    labels, n = pit_labels(pits, metric)
    weight = edge_weight(labels > 0)
    out = depth_t.copy()
    keep = np.zeros(depth_t.shape, dtype=bool)
    min_cells = POND_MIN_AREA * depth_t.size
    min_depth = POND_MIN_DEPTH_M if metric else POND_MIN_DEPTH_REL
    skipped = drained = 0
    for k in range(1, n + 1):
        m = labels == k
        cells = int(m.sum())
        if cells < min_cells or float(pits[m].max()) < min_depth:
            skipped += 1
            continue
        cap = pits[m] if metric else None
        held = [np.minimum(f[m], cap) if metric else f[m] for f in frames]
        vols = [float(h.sum()) for h in held]
        i = int(np.argmax(vols))   # 이 웅덩이에 물이 가장 많았던 시점
        vol = vols[i]
        area = cells / depth_t.size
        if decay and area < 2 * POND_AREA_REF:
            tau = POND_TAU_S * (area / POND_AREA_REF) ** 1.5
            frac = float(np.exp(-max(0.0, t_now - float(times[i])) / max(tau, 1e-6)))
            if frac < POND_GONE:
                drained += 1
                continue
            vol *= frac
        out[m] = _scale_to_volume(depth_t[m], held[i], weight[m], cap, vol)
        keep |= m
    raised = out > depth_t + 1e-4
    info = {
        "applied": bool(raised.any()),
        "pits": int(n),
        "held_pits": int(n - skipped - drained),
        "skipped_small": int(skipped),
        "drained_small": int(drained),
        "pit_area_pct": round(float((labels > 0).mean()) * 100.0, 1),
        "raised_area_pct": round(float(raised.mean()) * 100.0, 1),
        "added_volume_pct": round(float((out - depth_t).sum() / max(depth_t.sum(), 1e-9)) * 100.0, 1),
    }
    return out, info, keep
