"""
확장자별 입력 자동 검증 (마무리 점검용).

make_all_test_inputs.py 가 만든 테스트 파일(목록.csv)을 켜져 있는 서버의 /api/predict 에
화면과 똑같은 방식으로 하나씩 보내고, 결과를 표(결과.csv)와 그림 모음 페이지(결과.html)로 정리한다.
기록(history)은 저장하지 않는다.

사용법 (서버를 먼저 켠 뒤, prediction_server 폴더에서):
    .venv\\Scripts\\python.exe run_input_tests.py --dir C:\\yuche\\test_input
    -> 로그인 이메일/비밀번호를 물어본다 (화면에서 가입한 테스트 계정)
    -> C:\\yuche\\test_input\\_결과\\결과.html 을 브라우저로 열기
"""
import argparse
import base64
import csv
import getpass
import html
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _request(url, data=None, headers=None, timeout=180):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return 0, str(e.reason).encode("utf-8")


def login(server, email, password):
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    status, raw = _request(server + "/api/auth/login", body, {"Content-Type": "application/json"})
    if status != 200:
        return None, f"로그인 실패 ({status}): {raw.decode('utf-8', 'replace')[:200]}"
    return json.loads(raw)["token"], None


def post_predict(server, token, path, mode, rain_mm, time_s):
    boundary = "----yuche" + uuid.uuid4().hex
    fields = {"mode": mode, "rain_mm": str(rain_mm), "time_s": str(time_s)}
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    with open(path, "rb") as f:
        content = f.read()
    fname = os.path.basename(path)
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8") + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = "Bearer " + token
    t0 = time.perf_counter()
    status, raw = _request(server + "/api/predict", b"".join(parts), headers)
    return status, raw, time.perf_counter() - t0, len(content)


# --------------------------------------------------------------------------- #
# 결과 해석
# --------------------------------------------------------------------------- #
def _gray(b64, size=128):
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def judge(expect, status, diff, folder, warned=False):
    """자동 판정 (참고용). 최종 판단은 그림을 보고 사람이 한다."""
    expect_error = "에러 안내" in expect
    if status == 0:
        return "bad", "서버 연결 안 됨"
    if status >= 500:
        return "bad", "서버 에러(500) - 안내 없이 실패"
    if expect_error:
        return ("ok", "안내 메시지로 거부") if status in (400, 413, 422) else ("bad", "거부돼야 하는데 통과")
    if status != 200:
        if "문제 예상" in expect:
            return "ok", "잘못된 입력을 안내 메시지로 거부"
        return ("bad", "실패") if "정상" in expect else ("look", "거부됨 - 메시지 확인")
    if "축척 확인" in expect or "고저차" in expect or "도메인" in expect:
        if warned:
            return "ok", "예측 + 주의 문구 표시"
        return "look", "기준과 비교 (같으면 축척·고저차를 무시하는 것, 주의 문구 없음)"
    if any(k in expect for k in ("문제 예상", "경고 없음", "엉뚱", "확인", "뭉개", "여백", "고일 수", "바뀜")):
        if warned:
            return "ok", "예측 + 주의 문구 표시 (그림도 한번 보기)"
        return "look", "그림 확인 필요 (주의 문구 없음)"
    if folder.startswith("01") and diff is not None and diff > 8.0:
        return "look", f"기준과 다름 ({diff:.1f}%)"
    return "ok", "정상 예측"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="make_all_test_inputs.py 출력 폴더 (목록.csv 있는 곳)")
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--rain", type=float, default=50.0)
    p.add_argument("--time", type=float, default=18.0)
    p.add_argument("--email", default=None)
    p.add_argument("--password", default=None)
    args = p.parse_args()

    root = os.path.abspath(args.dir)
    listing = os.path.join(root, "목록.csv")
    if not os.path.exists(listing):
        sys.exit(f"목록.csv 가 없습니다: {listing}\n먼저 make_all_test_inputs.py 를 실행하세요.")
    out = os.path.join(root, "_결과")
    img_dir = os.path.join(out, "img")
    os.makedirs(img_dir, exist_ok=True)

    status, raw = _request(args.server + "/api/health")
    if status != 200:
        sys.exit(f"서버에 연결할 수 없습니다 ({args.server}). 서버를 먼저 켜세요.")
    health = json.loads(raw)
    token = None
    if health.get("auth_enabled"):
        email = args.email or input("로그인 이메일: ").strip()
        password = args.password or getpass.getpass("비밀번호 (입력해도 화면에 안 보임): ")
        token, err = login(args.server, email, password)
        if err:
            sys.exit(err)

    with open(listing, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    jobs = []
    for r in rows:
        mode_txt = r["넣을 모드"]
        modes = []
        if "이미지" in mode_txt:
            modes.append("image")
        if "데이터" in mode_txt:
            modes.append("terrain")
        for m in modes:
            jobs.append((r, m))

    results = []
    base_pred = {}   # 모드별 기준 예측: 이미지 = contour.png, 고도격자 = terrain.npy (둘 다 01 폴더)
    print(f"{len(jobs)}건 검사 시작 (강우 {args.rain:g} mm, 시간 {args.time:g} s)\n")
    for i, (r, mode) in enumerate(jobs, 1):
        rel = r["파일"]
        path = os.path.join(root, rel)
        folder = rel.replace("\\", "/").split("/")[0]
        st, body, sec, nbytes = post_predict(args.server, token, path, mode, args.rain, args.time)
        res = {"번호": i, "폴더": folder, "파일": os.path.basename(rel), "모드": "이미지" if mode == "image" else "고도격자",
               "확인할 것": r["무엇을 확인하나"], "기대 결과": r["기대 결과"], "HTTP": st, "시간(초)": round(sec, 2),
               "파일크기(KB)": round(nbytes / 1024, 1), "응답크기(KB)": round(len(body) / 1024, 1),
               "메시지": "", "최대침수(cm)": "", "5cm이상면적(%)": "", "격자": "", "셀크기(m)": "",
               "빈칸채움": "", "색체계": "", "경고": "", "기준과차이(%)": "", "img_in": "", "img_ov": ""}
        diff = None
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            data = None
        if st == 200 and isinstance(data, dict):
            stats = data.get("stats") or {}
            meta = data.get("meta") or {}
            ci = data.get("color_info") or {}
            res["최대침수(cm)"] = stats.get("max_depth_cm", "")
            res["5cm이상면적(%)"] = stats.get("wet_area_pct", "")
            res["격자"] = "x".join(str(v) for v in meta.get("grid_shape", [])) if meta.get("grid_shape") else ""
            res["셀크기(m)"] = meta.get("cellsize_m") or meta.get("dx_m") or ""
            res["빈칸채움"] = meta.get("nodata_filled", "")
            res["색체계"] = ci.get("scheme", "") + (" (변환)" if ci.get("converted") else "")
            res["경고"] = " / ".join((data.get("warnings") or []) + (ci.get("warnings") or []))
            tag = f"{i:02d}"
            for key, b64k in (("img_in", "contour_png_base64"), ("img_ov", "overlay_png_base64")):
                if data.get(b64k):
                    name = f"{tag}_{key}.png"
                    with open(os.path.join(img_dir, name), "wb") as f:
                        f.write(base64.b64decode(data[b64k]))
                    res[key] = "img/" + name
            if data.get("prediction_png_base64"):
                pred = _gray(data["prediction_png_base64"])
                relp = rel.replace("\\", "/")
                if mode not in base_pred and relp.endswith(("01_확장자_같은지형/contour.png", "01_확장자_같은지형/terrain.npy")):
                    base_pred[mode] = pred
                    if mode == "terrain" and "image" in base_pred:   # 같은 지형: 데이터 모드 vs 이미지 모드
                        res["메시지"] = ""
                        res["경고"] = (f"이미지 모드(contour.png)와 차이 "
                                     f"{np.mean(np.abs(pred - base_pred['image'])) / 255.0 * 100.0:.1f}%")
                if mode in base_pred:
                    diff = float(np.mean(np.abs(pred - base_pred[mode])) / 255.0 * 100.0)
                    res["기준과차이(%)"] = round(diff, 1)
        else:
            if isinstance(data, dict) and "detail" in data:
                d = data["detail"]
                res["메시지"] = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
            else:
                res["메시지"] = body.decode("utf-8", "replace")[:300]
        kind, verdict = judge(r["기대 결과"], st, diff, folder, warned=bool(res["경고"]))
        res["판정"] = verdict
        res["_kind"] = kind
        results.append(res)
        mark = {"ok": "OK  ", "look": "확인", "bad": "문제"}[kind]
        print(f"[{mark}] {i:2d}/{len(jobs)} {rel} ({res['모드']}) -> {st} {sec:.1f}s  {verdict}")

    cols = ["번호", "판정", "폴더", "파일", "모드", "HTTP", "시간(초)", "메시지", "최대침수(cm)", "5cm이상면적(%)",
            "기준과차이(%)", "격자", "셀크기(m)", "빈칸채움", "색체계", "경고", "파일크기(KB)", "응답크기(KB)",
            "확인할 것", "기대 결과"]
    with open(os.path.join(out, "결과.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    write_html(os.path.join(out, "결과.html"), results, args)
    n = {k: sum(1 for r in results if r["_kind"] == k) for k in ("ok", "look", "bad")}
    print(f"\n끝: 정상 {n['ok']} / 확인 필요 {n['look']} / 문제 {n['bad']}")
    print(f"결과 페이지: {os.path.join(out, '결과.html')}")


def write_html(path, results, args):
    color = {"ok": "#1f7a3a", "look": "#a66300", "bad": "#b3261e"}
    label = {"ok": "정상", "look": "확인", "bad": "문제"}
    e = html.escape
    rows = []
    for r in results:
        imgs = "".join(f'<img src="{e(r[k])}" title="{t}">' for k, t in (("img_in", "모델 입력"), ("img_ov", "결과 오버레이")) if r[k])
        if r["메시지"]:
            summary = f'<div class="msg">{e(str(r["메시지"]))}</div>'
        else:
            bits = [f'최대 {r["최대침수(cm)"]} cm', f'면적 {r["5cm이상면적(%)"]}%']
            if r["기준과차이(%)"] != "":
                bits.append(f'기준과 차이 {r["기준과차이(%)"]}%')
            if r["격자"]:
                bits.append(f'격자 {r["격자"]}')
            if r["셀크기(m)"] != "":
                bits.append(f'셀 {float(r["셀크기(m)"]):.3g} m')
            if r["빈칸채움"]:
                bits.append(f'빈칸 {r["빈칸채움"]}칸 채움')
            if r["색체계"]:
                bits.append(f'색 {r["색체계"]}')
            summary = " · ".join(e(b) for b in bits)
            if r["경고"]:
                summary += f'<div class="warn">경고: {e(r["경고"])}</div>'
        rows.append(f"""<tr class="{r['_kind']}">
<td>{r['번호']}</td>
<td><span class="tag" style="background:{color[r['_kind']]}">{label[r['_kind']]}</span><div class="sub">{e(r['판정'])}</div></td>
<td><b>{e(r['파일'])}</b><div class="sub">{e(r['폴더'])} · {e(r['모드'])} · {r['파일크기(KB)']} KB</div>
<div class="sub">확인: {e(r['확인할 것'])}</div><div class="sub">기대: {e(r['기대 결과'])}</div></td>
<td>HTTP {r['HTTP']}<div class="sub">{r['시간(초)']} 초 · 응답 {r['응답크기(KB)']} KB</div></td>
<td>{summary}</td>
<td class="imgs">{imgs}</td></tr>""")
    n = {k: sum(1 for r in results if r["_kind"] == k) for k in ("ok", "look", "bad")}
    doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>입력 검증 결과</title>
<style>
body{{font-family:'Malgun Gothic',sans-serif;margin:24px;background:#f6f7f9;color:#1b1f24}}
h1{{font-size:20px;margin:0 0 6px}} .meta{{color:#555;margin-bottom:14px;font-size:13px}}
.filters button{{margin-right:6px;padding:6px 12px;border:1px solid #ccc;background:#fff;border-radius:6px;cursor:pointer}}
table{{border-collapse:collapse;width:100%;background:#fff;margin-top:12px}}
td{{border-bottom:1px solid #e3e5e8;padding:8px;vertical-align:top;font-size:13px}}
.tag{{color:#fff;border-radius:4px;padding:2px 8px;font-weight:bold;font-size:12px}}
.sub{{color:#666;font-size:12px;margin-top:3px}} .msg{{color:#b3261e}} .warn{{color:#a66300;margin-top:4px}}
.imgs img{{width:150px;height:150px;object-fit:contain;border:1px solid #ddd;margin-right:6px;background:#fafafa}}
</style></head><body>
<h1>확장자별 입력 검증 결과</h1>
<div class="meta">서버 {e(args.server)} · 강우 {args.rain:g} mm · 시간 {args.time:g} s · 정상 {n['ok']} / 확인 {n['look']} / 문제 {n['bad']}<br>
그림: 왼쪽 = 모델에 들어간 등고선, 오른쪽 = 결과 오버레이. '기준과 차이'는 같은 모드의 01 폴더 기준 파일(이미지 = contour.png, 고도격자 = terrain.npy) 예측과의 평균 차이(%)입니다. 판정은 참고용이고 최종 판단은 그림을 보고 하세요.</div>
<div class="filters"><button onclick="f('')">전체</button><button onclick="f('bad')">문제만</button><button onclick="f('look')">확인만</button><button onclick="f('ok')">정상만</button></div>
<table>{''.join(rows)}</table>
<script>function f(k){{document.querySelectorAll('tr').forEach(t=>t.style.display=(!k||t.classList.contains(k))?'':'none')}}</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


if __name__ == "__main__":
    main()
