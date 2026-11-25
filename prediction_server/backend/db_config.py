"""DB 접속 정보 한 곳에서 관리.

비밀번호를 코드에 적지 않는다. 아래 순서로 찾는다.
  1. 환경변수 YUCHE_DB_URL (전체 주소, 예: mysql+pymysql://user:pw@host:3306/db)
  2. 환경변수 YUCHE_DB_HOST / YUCHE_DB_PORT / YUCHE_DB_USER / YUCHE_DB_PASSWORD / YUCHE_DB_NAME
  3. prediction_server/.env 파일의 같은 이름 값  (git 에 안 올라감. .env.example 을 복사해서 만든다)
  4. 기본값: 127.0.0.1:3306, root, 비밀번호 없음, DB yuche_ai_db

명령어로도 쓴다 (run_server.bat 이 서버 켜기 전에 부른다):
    python backend/db_config.py --create-db    # DB 접속 확인 + yuche_ai_db 없으면 만들기
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

SERVER_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = SERVER_DIR / ".env"

DEFAULTS = {
    "YUCHE_DB_HOST": "127.0.0.1",
    "YUCHE_DB_PORT": "3306",
    "YUCHE_DB_USER": "root",
    "YUCHE_DB_PASSWORD": "",
    "YUCHE_DB_NAME": "yuche_ai_db",
}


def _read_env_file(path: Path | None = None) -> dict[str, str]:
    """KEY=VALUE 줄만 읽는 간단한 .env 파서 (python-dotenv 없이). # 주석·빈 줄은 무시."""
    path = path or ENV_FILE
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key.strip()] = val
    return out


def settings() -> dict[str, str]:
    file_vals = _read_env_file()
    return {k: os.environ.get(k) or file_vals.get(k) or v for k, v in DEFAULTS.items()} | {
        "YUCHE_DB_URL": os.environ.get("YUCHE_DB_URL") or file_vals.get("YUCHE_DB_URL") or ""}


def db_url() -> str:
    s = settings()
    if s["YUCHE_DB_URL"]:
        return s["YUCHE_DB_URL"]
    # 비밀번호의 ! @ : / 같은 특수문자가 주소를 깨지 않도록 인코딩한다
    return (f"mysql+pymysql://{quote_plus(s['YUCHE_DB_USER'])}:{quote_plus(s['YUCHE_DB_PASSWORD'])}"
            f"@{s['YUCHE_DB_HOST']}:{s['YUCHE_DB_PORT']}/{s['YUCHE_DB_NAME']}?charset=utf8mb4")


def pymysql_kwargs(with_db: bool = True) -> dict:
    """pymysql.connect(**pymysql_kwargs()) 용. YUCHE_DB_URL 을 쓰는 경우에도 같은 값으로 풀어준다."""
    s = settings()
    if s["YUCHE_DB_URL"]:
        from sqlalchemy.engine import make_url
        u = make_url(s["YUCHE_DB_URL"])
        kw = {"host": u.host or "127.0.0.1", "port": int(u.port or 3306), "user": u.username or "root",
              "password": u.password or "", "database": u.database}
    else:
        kw = {"host": s["YUCHE_DB_HOST"], "port": int(s["YUCHE_DB_PORT"]), "user": s["YUCHE_DB_USER"],
              "password": s["YUCHE_DB_PASSWORD"], "database": s["YUCHE_DB_NAME"]}
    if not with_db:
        kw.pop("database")
    kw["charset"] = "utf8mb4"
    return kw


def db_name() -> str:
    return pymysql_kwargs()["database"]


def _create_db() -> int:
    import pymysql
    kw = pymysql_kwargs(with_db=False)
    name = db_name()
    try:
        c = pymysql.connect(**kw)
    except pymysql.err.OperationalError as exc:
        code = exc.args[0] if exc.args else None
        if code == 1045:
            print(f"[ERROR] MySQL 비밀번호가 틀렸습니다 (사용자 {kw['user']}).")
            print(f"        {ENV_FILE} 의 YUCHE_DB_PASSWORD 를 확인하세요."
                  + ("" if ENV_FILE.exists() else " (.env 파일이 없습니다: .env.example 을 복사해서 만드세요)"))
        else:
            print(f"[ERROR] MySQL 에 연결할 수 없습니다 ({kw['host']}:{kw['port']}): {exc}")
            print("        MySQL80 서비스가 켜져 있는지 확인하세요 (서비스 앱 -> MySQL80 -> 시작).")
        return 1
    with c:
        c.cursor().execute(f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4")
        c.commit()
    print(f"  DB ready: {name}")
    return 0


if __name__ == "__main__":
    if "--create-db" in sys.argv:
        sys.exit(_create_db())
    print(db_url().replace(quote_plus(settings()["YUCHE_DB_PASSWORD"]) or "\0", "****"))
