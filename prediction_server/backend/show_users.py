"""yuche_ai_db 의 가입 계정과 사용자별 분석 기록 수를 보여준다 (비밀번호 해시는 출력하지 않음)."""
import pymysql

import db_config  # 접속 정보는 prediction_server/.env (비밀번호를 코드에 적지 않음)

c = pymysql.connect(**db_config.pymysql_kwargs())
cur = c.cursor()
cur.execute("""
    SELECT u.id, u.email, u.created_at, COUNT(h.id)
    FROM users u LEFT JOIN analysis_history h ON h.user_email = u.email
    GROUP BY u.id, u.email, u.created_at ORDER BY u.id
""")
rows = cur.fetchall()
print(f"\n  {db_config.db_name()}.users : {len(rows)} accounts\n")
print(f"  {'id':>4}  {'email':<36} {'joined':<20} records")
print("  " + "-" * 72)
for i, email, created, n in rows:
    print(f"  {i:>4}  {email:<36} {str(created):<20} {n}")
print()
