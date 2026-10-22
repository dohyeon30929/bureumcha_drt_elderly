"""우선순위점수 결과에 목적별 파일 기반 '귀가 목적 비중'을 붙여 교차검증.

귀가 비중 = (귀가(move_purpose==3) 고령층 도착량) / (전체 목적 1~7 고령층 도착량), 같은 16~21시(hour 16~20).
비중이 낮으면 거주지가 아니라 업무·상업지구라서 그 시간대 도착이 귀가가 아닐 가능성이 크다.
'귀가비중_낮음' 표시 = 필터를 통과한 전체 동의 귀가비중 하위 10%.

선행: homebound_elderly_mode_share.py 실행 결과(귀가시간대_고령층_수단비율_행정동.csv)를 읽는다.
출력: 귀가시간대_우선순위_교차검증.csv (필터 통과 전 동, 우선순위 순)
"""
import csv
import glob
import io
import zipfile
from collections import defaultdict

HOURS = range(16, 21)
HOME = "3"
TOP = 30
IN = "data/interim/재현확인/귀가시간대_고령층_수단비율_행정동.csv"
OUT = "data/interim/재현확인/귀가시간대_우선순위_교차검증.csv"
ELDER = ("male_60_cnt", "male_70_cnt", "feml_60_cnt", "feml_70_cnt")

xw = {r["mois_admdong_cd"]: r["bnd_admdong_cd"]
      for r in csv.DictReader(open("data/interim/재현확인/행정동_코드매핑_생활이동_경계.csv", encoding="utf-8-sig"))}

home = defaultdict(float)
allp = defaultdict(float)
zpath = [p for p in glob.glob("data/raw/*.zip") if "성 연령별, 도착지" in p][0]
with zipfile.ZipFile(zpath) as z:
    for name in z.namelist():
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
                t = int(r["time_cd"])
                if (t // 100 if t >= 100 else t) not in HOURS or r["d_admdong_cd"] not in xw:
                    continue
                v = sum(float(r[c] or 0) for c in ELDER)
                b = xw[r["d_admdong_cd"]]
                allp[b] += v
                if r["move_purpose"] == HOME:
                    home[b] += v

rows = list(csv.DictReader(open(IN, encoding="utf-8-sig")))
for r in rows:
    b = r["경계코드"]
    r["귀가목적고령층"] = round(home[b], 1)
    r["전체목적고령층"] = round(allp[b], 1)
    r["귀가비중"] = round(home[b] / allp[b], 4) if allp[b] else ""

shares = sorted(r["귀가비중"] for r in rows if r["귀가비중"] != "")
cut = shares[len(shares) // 10]
for r in rows:
    r["귀가비중_낮음"] = "낮음" if r["귀가비중"] != "" and r["귀가비중"] < cut else ""

cols = list(rows[0].keys())
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

med = shares[len(shares) // 2]
print(f"귀가비중 분포(필터 통과 {len(shares)}개 동): 최소 {shares[0]:.1%}, 하위10% 컷 {cut:.1%}, "
      f"중앙값 {med:.1%}, 상위10% {shares[len(shares) * 9 // 10]:.1%}, 최대 {shares[-1]:.1%}")
print(f"\n== 우선순위 상위 {TOP} + 귀가비중 (낮음 = 하위10% 컷 {cut:.1%} 미만)")
for r in rows[:TOP]:
    print(f"{r['우선순위_순위']:>2} {r['자치구']} {r['동이름']:<9} 점수{float(r['우선순위점수']):+.3f} "
          f"차{float(r['차량비율']):.1%} 대중교통{float(r['대중교통비율(버스+지하철)']):.1%} "
          f"귀가비중{float(r['귀가비중']):.1%} {r['귀가비중_낮음']}")
low = [r for r in rows[:TOP] if r["귀가비중_낮음"]]
print(f"\n상위 {TOP} 중 귀가비중 낮음 {len(low)}개:", [f"{r['자치구']} {r['동이름']}" for r in low])
print("전체 하위10% 동:", [f"{r['자치구']} {r['동이름']}({float(r['귀가비중']):.0%})"
                           for r in sorted(rows, key=lambda r: float(r["귀가비중"]))[:len(shares) // 10]])
