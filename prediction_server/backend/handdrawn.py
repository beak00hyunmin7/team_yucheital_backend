"""손그림 등고선 사진 -> 고도 격자 (전시용 '손그림 모드').

관람객이 학습지에 파랑 동그라미를 여러 겹 그리고(안쪽일수록 높음) 꼭대기를 빨강으로 그린 뒤
폰으로 찍어 올리면, 그 사진을 고도 격자로 바꾼다. 서버는 이 격자를 '고도격자 모드'와 똑같이
학습 형식 등고선으로 다시 그려 AI 에 넣으므로, 분석 지도·시간 변화·3D 를 그대로 쓴다.

순서
  1) 모서리 검은 네모 4개 찾기 (왼쪽 위만 속이 빈 네모 -> 사진이 돌아가 있어도 방향을 안다)
  2) 네 네모를 정사각형 꼭짓점으로 펴기 (비스듬히 찍어도 됨)
  3) 파랑·빨강 선 찾기 (튄 잉크 점은 버림)
  4) 바깥(종이 테두리와 이어진 빈 곳) = 0층, 선을 하나 넘을 때마다 +1층
  5) 층 사이를 매끈하게 이어 고도로

numpy / scipy / pillow 만 쓴다 (서버 PC 에 opencv 가 없어도 동작).
"""
from __future__ import annotations

import io
import itertools

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

WORK = 512            # 펴서 읽는 해상도 (한 변 픽셀)
LEVEL_M = 10.0        # 한 층 높이[m]. 모델은 고저차를 안 보므로 값 자체는 결과에 영향 없음
# 그림 밖 땅을 위(북)쪽이 높고 아래(남)쪽이 낮게 살짝 기울인다[m].
# 완전히 평평하면 등고선 그림에서 그림 밖 전체가 '가장 낮은 색'이 되어, AI 가 산 주변 평지 전체를
# 물웅덩이로 본다(학습 지형에는 평평한 땅이 없다). 기울이면 물이 아래쪽·산 사이 골짜기로 모인다.
TILT_M = 15.0


class HandDrawnError(ValueError):
    """관람객에게 그대로 보여줄 안내 문구를 담는다 (서버가 422 로 응답)."""


def _cr(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _open(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)      # 폰 사진의 회전 정보 반영
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HandDrawnError(f"사진을 열 수 없습니다: {exc}") from exc
    s = 1200 / max(img.size)
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    return img


# --------------------------------------------------------------------------- #
# 1) 모서리 네모
# --------------------------------------------------------------------------- #
def find_markers(rgb: np.ndarray):
    """네 모서리 네모 중심 (왼위, 오위, 오아래, 왼아래) 와 빈 네모를 찾았는지. 못 찾으면 None."""
    h, w, _ = rgb.shape
    mx, mn = rgb.max(-1), rgb.min(-1)
    dark = (mx < 95) & (mx - mn < 60)
    lab, _ = ndi.label(dark)
    cands = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
            continue                            # 사진 끝에 걸친 어두운 것(책상·그림자·물건)은 네모가 아님
        m = lab[sl] == i
        area = int(m.sum())
        if not (h * w * 2e-4 < area < h * w * 1.5e-2):
            continue
        bh, bw = m.shape
        if not (0.4 < bh / bw < 2.5):
            continue
        filled_m = ndi.binary_fill_holes(m)
        filled = int(filled_m.sum())
        if filled / (bh * bw) < 0.45:           # 네모·동그라미처럼 꽉 찬 모양만
            continue
        cy, cx = ndi.center_of_mass(filled_m)
        cands.append(dict(x=sl[1].start + cx, y=sl[0].start + cy, area=filled,
                          hollow=filled / area > 1.15))
    if len(cands) < 4:
        return None
    cands.sort(key=lambda c: -c["area"])
    best = None
    for q in itertools.combinations(cands[:12], 4):
        pts = np.array([[c["x"], c["y"]] for c in q])
        ctr = pts.mean(0)
        order = np.argsort(np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0]))   # 화면 시계 방향
        p = pts[order]
        cross = [_cr(p[(i + 1) % 4] - p[i], p[(i + 2) % 4] - p[(i + 1) % 4]) for i in range(4)]
        if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
            continue                            # 볼록 사각형이 아니면 버림
        sizes = np.array([q[j]["area"] for j in order])
        if sizes.max() / sizes.min() > 4:       # 크기가 너무 제각각이면 다른 물체가 섞인 것
            continue
        area = 0.5 * abs(sum(_cr(p[i], p[(i + 1) % 4]) for i in range(4)))
        if area < h * w * 0.08:                 # 사진의 아주 작은 부분이면 네모 4개가 아님
            continue
        if best is None or area > best[0]:
            best = (area, [q[j] for j in order])
    if best is None:
        return None
    quad = best[1]
    hol = [i for i, c in enumerate(quad) if c["hollow"]]
    k = hol[0] if len(hol) == 1 else int(np.argmin([c["x"] + c["y"] for c in quad]))
    quad = quad[k:] + quad[:k]                  # 빈 네모가 왼쪽 위
    return np.array([[c["x"], c["y"]] for c in quad]), len(hol) == 1


# --------------------------------------------------------------------------- #
# 2) 펴기
# --------------------------------------------------------------------------- #
def _persp_coeffs(dst, src):
    """PIL PERSPECTIVE 계수: 출력 좌표(dst) -> 입력 좌표(src)."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float))


def _warp(img: Image.Image, quad: np.ndarray, size: int = WORK) -> Image.Image:
    dst = [(0, 0), (size, 0), (size, size), (0, size)]
    return img.transform((size, size), Image.PERSPECTIVE, tuple(_persp_coeffs(dst, quad)), Image.BICUBIC)


# --------------------------------------------------------------------------- #
# 3) 선 찾기  4) 층 세기  5) 매끈하게
# --------------------------------------------------------------------------- #
def _strokes(rgb: np.ndarray, clear_corners: bool):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (b > r + 35) & (b > g + 15)
    red = (r > g + 60) & (r > b + 35)
    s = rgb.shape[0]
    if clear_corners:                           # 모서리 네모 자리
        c = int(s * 0.07)
        for m in (blue, red):
            m[:c, :c] = m[:c, -c:] = m[-c:, :c] = m[-c:, -c:] = False
    st = ndi.binary_closing(blue | red, structure=np.ones((5, 5)), iterations=1)   # 살짝 끊긴 선 잇기
    lab, n = ndi.label(st)
    if n:
        sizes = ndi.sum(st, lab, range(1, n + 1))
        st &= np.isin(lab, 1 + np.nonzero(sizes >= s * s * 6e-4)[0])            # 튄 잉크 점 제거
    return st, blue & st, red & st


def _levels(st: np.ndarray):
    free = ~st
    lab, n = ndi.label(free)
    sizes = ndi.sum(free, lab, range(1, n + 1))
    half_w = float(np.percentile(ndi.distance_transform_edt(st)[st], 90)) if st.any() else 2.0
    reach = int(2 * half_w + 3)                 # 선 하나를 건너뛸 거리
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
    lvl = {k: 0 for k in edge}
    frontier = list(edge)
    while frontier:                             # 바깥부터 선을 하나씩 넘으며 층 매기기
        nxt = []
        for k in frontier:
            grown = ndi.binary_dilation(lab == k, iterations=reach)
            for j in np.unique(lab[grown & free]).tolist():
                if j and j not in lvl:
                    lvl[j] = lvl[k] + 1
                    nxt.append(j)
        frontier = nxt
    L = np.zeros(st.shape, float)
    valid = np.zeros(st.shape, bool)
    for k, v in lvl.items():
        if sizes[k - 1] >= 30:                  # 선이 겹친 자리의 아주 작은 틈은 무시
            L[lab == k] = v
            valid |= lab == k
    if valid.any():
        idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        L = L[idx[0], idx[1]]                   # 선 자리는 가장 가까운 빈 곳의 층으로
    top = int(L.max())
    regions = {k: v for k, v in lvl.items() if sizes[k - 1] >= 30}
    return L, top, lab, regions, reach


def _height(st: np.ndarray, lab: np.ndarray, regions: dict, reach: int, foot: float = 0.10) -> np.ndarray:
    """층 -> 이어진 고도 (층 단위). 동그라미 선이 곧 등고선이다.

    예전에는 층마다 계단을 만들고 경계만 살짝 기울였더니, 동그라미 사이가 넓으면 그 사이가 평평한
    '계단 논'이 되어 AI 가 산 중턱에 물 띠를 그렸다. 실제 등고선처럼 선 사이를 끊김 없이 잇는다:
      - 선 위 = 그 선 안쪽 층 값 (바깥 선 1, 그다음 2 …)
      - 두 선 사이 = 바깥 선에서 안쪽 선까지 거리 비율로 k -> k+1
      - 가장 안쪽(꼭대기) = 안쪽으로 갈수록 둥글게 조금 더 (+0.6)
      - 바깥 평지 = 산 밑동에서 그림 한 변의 10% 거리 안에서 1 -> 0 으로 완만하게
    """
    S = st.shape[0]
    R = np.full(st.shape, -1.0)
    for k, v in regions.items():
        R[lab == k] = v
    V = ndi.grey_dilation(R, size=2 * reach + 1)          # 선 픽셀 값 = 닿은 빈 곳 중 가장 높은 층(안쪽)
    V[~st] = -1
    top = int(max(regions.values())) if regions else 0
    D = {v: ndi.distance_transform_edt(~(st & (V == v))) if (st & (V == v)).any() else None
         for v in range(1, top + 2)}
    H = np.full(st.shape, np.nan)
    H[st] = np.clip(V[st], 0, None)
    for k, v in regions.items():
        M = lab == k
        inner = st & (V == v + 1)
        has_inner = inner.any() and (ndi.binary_dilation(M, iterations=reach) & inner).any()
        if v == 0:
            if D.get(1) is not None:
                H[M] = np.clip(1.0 - D[1][M] / (foot * S), 0.0, 1.0)
            else:
                H[M] = 0.0
        elif has_inner and D.get(v) is not None:
            do, di = D[v][M], D[v + 1][M]
            H[M] = v + do / (do + di + 1e-6)
        else:                                               # 꼭대기
            do = D[v][M] if D.get(v) is not None else np.zeros(M.sum())
            t = do / max(float(do.max()), 1.0)
            H[M] = v + 0.6 * (1.0 - (1.0 - t) ** 2)
    miss = np.isnan(H)
    if miss.any():                                          # 선이 겹친 작은 틈 등은 가까운 값으로
        idx = ndi.distance_transform_edt(miss, return_distances=False, return_indices=True)
        H = H[idx[0], idx[1]]
    return ndi.gaussian_filter(H, S * 0.006)


def _preview(flat: Image.Image, L: np.ndarray, top: int, size: int = 320) -> Image.Image:
    """'이렇게 읽었어요' 그림: 왼쪽 = 편 사진, 오른쪽 = 읽은 층(높을수록 밝게)."""
    import matplotlib
    cmap = matplotlib.colormaps["terrain"]
    norm = L / max(top, 1)
    col = (np.asarray(cmap(0.25 + 0.7 * norm))[..., :3] * 255).astype(np.uint8)
    right = Image.fromarray(col).resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size * 2 + 12, size), "white")
    canvas.paste(flat.resize((size, size), Image.BILINEAR), (0, 0))
    canvas.paste(right, (size + 12, 0))
    return canvas


def read_drawing(raw: bytes, n: int = 96, tilt_m: float = TILT_M):
    """사진 바이트 -> (terrain[n x n, m, 0행=남쪽], preview 그림, 정보, 주의 문구).

    tilt_m: 그림 아래(남)쪽 끝 0 m -> 위(북)쪽 끝 tilt_m 로 땅 전체를 기울인다. 0 이면 평평.
    """
    img = _open(raw)
    rgb = np.asarray(img).astype(np.int16)
    warns: list[str] = []
    found = find_markers(rgb)
    if found is not None:
        quad, hollow_ok = found
        flat = _warp(img, quad)
        if not hollow_ok:
            warns.append("왼쪽 위 빈 네모를 못 알아봐서 방향은 위치로 추정했습니다.")
    else:
        flat = img.resize((WORK, WORK), Image.BICUBIC)
        warns.append("모서리 네모 4개를 찾지 못해 사진 전체를 그대로 읽었습니다. "
                     "네 모서리 네모가 다 보이게 위에서 찍으면 더 정확합니다.")
    st, blue, red = _strokes(np.asarray(flat).astype(np.int16), clear_corners=found is not None)
    if st.sum() < WORK * WORK * 0.003:
        raise HandDrawnError("파랑·빨강 선을 거의 찾지 못했습니다. 굵은 매직으로 진하게 그린 뒤 다시 찍어 주세요.")
    L, top, lab, regions, reach = _levels(st)
    if top < 1:
        raise HandDrawnError("닫힌 동그라미를 찾지 못했습니다. 동그라미 끝을 꼭 이어서 그려 주세요.")
    if not red.any():
        warns.append("빨강 꼭대기 표시가 없어서 가장 안쪽 동그라미를 꼭대기로 봤습니다.")
    H = _height(st, lab, regions, reach) * LEVEL_M
    H[H < LEVEL_M * 0.05] = 0.0                 # 매끈하게 하며 평지에 남은 아주 작은 높이(등고선 잡음) 제거
    small = np.asarray(Image.fromarray(H.astype(np.float32)).resize((n, n), Image.BILINEAR), dtype=np.float64)
    terrain = np.flipud(small).copy()           # 서버 격자는 0행이 그림 아래쪽(남쪽)
    if tilt_m:
        terrain += np.linspace(0.0, float(tilt_m), n)[:, None]
    info = {"levels": top, "markers_found": found is not None, "tilt_m": float(tilt_m),
            "hollow_found": bool(found and found[1])}
    return terrain, _preview(flat, L, top), info, warns
