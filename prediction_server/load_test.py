"""
동시 접속 테스트: 관람객 여러 명이 같은 순간에 '예측'을 눌렀을 때 각자 얼마나 기다리는지 잰다.

한 사람 = 화면이 실제로 보내는 요청과 같게
  1) 첫 화면 요청 1번 (18초, 분석 지도는 안 그림: 화면은 '결과 이미지'를 열 때만 그린다)
  2) 시간 변화 장면을 묶어서 (/api/predict/series): 38·59·79초 한 번(빗금·배수구용), 100·120초 한 번
을 순서대로 보낸다. N명을 동시에 출발시키고 '첫 화면까지'와 '전체 끝까지' 걸린 시간을 표로 보여 준다.

사용법 (서버를 먼저 켠 뒤, prediction_server 폴더에서):
    .venv\\Scripts\\python.exe load_test.py --file C:\\yuche\\test_input\\01_확장자_같은지형\\terrain.npy
    -> 1명 · 3명 · 5명 · 10명을 차례로 잰다 (로그인 계정을 물어본다)
    --users 1,5     잴 인원
    --mode hand --file 사진.jpg     손그림 사진으로
    --analysis      첫 화면에서 분석 지도까지 그리기 (예전 방식과 비교용)
    --per-frame     시간 변화 장면을 하나씩 요청 (예전 방식과 비교용)
"""
import argparse
import getpass
import json
import os
import statistics
import sys
import threading
import time

from run_input_tests import _request, login   # 같은 폴더의 HTTP 도우미 재사용

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

LIGHT_TIMES = [38, 59, 79, 100, 120]


def predict(server, token, content, fname, mode, time_s, rain, light=False, no_analysis=False, series=None):
    import uuid
    boundary = "----yuche" + uuid.uuid4().hex
    fields = {"mode": mode, "rain_mm": str(rain), "time_s": str(time_s)}
    if series:
        fields = {"mode": mode, "rain_mm": str(rain), "times": ",".join(str(t) for t in series)}
    if light:
        fields["light"] = "true"
    if no_analysis:
        fields["no_analysis"] = "true"
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items()]
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8") + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = "Bearer " + token
    url = server + ("/api/predict/series" if series else "/api/predict")
    status, _raw = _request(url, b"".join(parts), headers, timeout=180)
    return status


def one_user(i, start, out, args, token, content, fname):
    start.wait()
    t0 = time.perf_counter()
    st = predict(args.server, token, content, fname, args.mode, 18, args.rain, no_analysis=not args.analysis)
    first = time.perf_counter() - t0
    fails = 0 if st == 200 else 1
    risk = None
    if not args.per_frame:   # 화면과 같게: 빗금 구간(38·59·79초)을 한 묶음, 나머지(100·120초)를 한 묶음
        if predict(args.server, token, content, fname, args.mode, 0, args.rain, series=LIGHT_TIMES[:3]) != 200:
            fails += 1
        risk = time.perf_counter() - t0
        if predict(args.server, token, content, fname, args.mode, 0, args.rain, series=LIGHT_TIMES[3:]) != 200:
            fails += 1
        out[i] = (first, time.perf_counter() - t0, fails, st, risk)
        return
    for t in LIGHT_TIMES:
        if predict(args.server, token, content, fname, args.mode, t, args.rain, light=True) != 200:
            fails += 1
        if t == 79:   # 화면은 79초 장면까지 받으면 침수 위험지역(빗금)·배수구(빨간 원)를 띄운다
            risk = time.perf_counter() - t0
    out[i] = (first, time.perf_counter() - t0, fails, st, risk)


def run(n, args, token, content, fname):
    start, out = threading.Event(), [None] * n
    ts = [threading.Thread(target=one_user, args=(i, start, out, args, token, content, fname)) for i in range(n)]
    [t.start() for t in ts]
    time.sleep(0.2)
    start.set()                                   # 모두 같은 순간에 출발
    [t.join() for t in ts]
    firsts = [o[0] for o in out]
    totals = [o[1] for o in out]
    fails = sum(o[2] for o in out)
    risks = [o[4] for o in out]
    return firsts, risks, totals, fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="보낼 파일 (고도 격자 .npy 등, 손그림이면 사진)")
    p.add_argument("--mode", default=None, help="terrain | image | hand (기본: 확장자로 판단)")
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--users", default="1,3,5,10")
    p.add_argument("--rain", type=float, default=50.0)
    p.add_argument("--analysis", action="store_true", help="첫 화면에서 분석 지도까지 그리기 (예전 방식)")
    p.add_argument("--per-frame", action="store_true", help="시간 변화 장면을 하나씩 따로 요청 (예전 방식)")
    p.add_argument("--email", default=None)
    p.add_argument("--password", default=None)
    args = p.parse_args()
    if not args.mode:
        ext = os.path.splitext(args.file)[1].lower()
        args.mode = "image" if ext in (".png", ".jpg", ".jpeg", ".webp") else "terrain"

    status, raw = _request(args.server + "/api/health")
    if status != 200:
        sys.exit(f"서버에 연결할 수 없습니다 ({args.server}). 서버를 먼저 켜세요.")
    token = None
    if json.loads(raw).get("auth_enabled"):
        email = args.email or input("로그인 이메일: ").strip()
        password = args.password or getpass.getpass("비밀번호 (입력해도 화면에 안 보임): ")
        token, err = login(args.server, email, password)
        if err:
            sys.exit(err)
    with open(args.file, "rb") as f:
        content = f.read()
    fname = os.path.basename(args.file)

    print(f"\n동시 접속 테스트 · {args.mode} · {fname} · 분석 지도 {'그림' if args.analysis else '안 그림'}")
    print("동시 인원 | 수심 지도 (평균/가장 늦은) | 빗금·배수구 (평균/가장 늦은) | 전체 (평균/가장 늦은) | 실패")
    predict(args.server, token, content, fname, args.mode, 18, args.rain, no_analysis=True)   # 캐시 데우기
    for n in [int(x) for x in args.users.split(",") if x.strip()]:
        firsts, risks, totals, fails = run(n, args, token, content, fname)
        print(f"{n:>6}명   |  {statistics.mean(firsts):4.1f}초 / {max(firsts):4.1f}초       |"
              f"  {statistics.mean(risks):4.1f}초 / {max(risks):4.1f}초          |"
              f"  {statistics.mean(totals):4.1f}초 / {max(totals):4.1f}초  | {fails}")
        time.sleep(1.0)
    print("\n수심 지도 = 첫 화면, 빗금·배수구 = 79초 장면까지 받은 때(화면이 빗금·빨간 원을 띄움),"
          " 전체 = 시간 변화 장면까지 다 받기까지 (서버 처리 + 대기, 인터넷 시간 제외)")


if __name__ == "__main__":
    main()
