"""귀가 시간대(16~21시, 즉 hour 16~20) 도착 고령층의 수단별 비율 + 우선순위점수를 행정동(경계 기준)별로 집계.

코드(수도권 생활이동 매뉴얼 p.12, p.20-21)
  이동수단: 1=항공 2=기차 3=고속버스 4=광역버스 5=일반버스 6=지하철 7=도보 8=차량 9=기타
- 입력: 수단별 생활이동 zip(8월 31일치), 행정동_코드매핑_생활이동_경계.csv
- 고령층 = male_60 + male_70 + feml_60 + feml_70
- 분모 '전체' = 해당 동 귀가시간대 고령층 도착량 합계(move_trans 1~9 전부)
- 경계 코드 기준 합산(용신동 두 코드는 여기서 합쳐짐)
- 수단별 파일에는 목적 구분이 없어 귀가 외 목적의 도착도 포함된다
- 표본 필터: 귀가시간대고령층합계 MIN_TOTAL 미만 동 제외
- 우선순위점수 = 차량비율 - (버스+지하철)비율 (메인 정렬 기준). 차량비율 순위/대중교통낮음 순위는 참고용
"""
import csv
import glob
import io
import zipfile
from collections import defaultdict

HOURS = range(16, 21)
MIN_TOTAL = 50000
TOP_REF = 20
TOP_SCORE = 30
OUT = "data/interim/재현확인/귀가시간대_고령층_수단비율_행정동.csv"
ELDER = ("male_60_cnt", "male_70_cnt", "feml_60_cnt", "feml_70_cnt")
CAR, BUS, SUBWAY, WALK = ("8",), ("4", "5"), ("6",), ("7",)

xw = {}
for r in csv.DictReader(open("data/interim/재현확인/행정동_코드매핑_생활이동_경계.csv", encoding="utf-8-sig")):
    xw[r["mois_admdong_cd"]] = (r["bnd_admdong_cd"], r["sigungu"], r["admdong_nm"])

zpath = [p for p in glob.glob("data/raw/*.zip") if "수단" in p][0]
agg = defaultdict(lambda: defaultdict(float))
with zipfile.ZipFile(zpath) as z:
    for name in z.namelist():
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
                t = int(r["time_cd"])
                if (t // 100 if t >= 100 else t) not in HOURS:
                    continue
                code = r["d_admdong_cd"]
                if code not in xw:
                    continue  # 서울 외 코드
                agg[xw[code][0]][r["move_trans"]] += sum(float(r[c] or 0) for c in ELDER)

names = {v[0]: (v[1], v[2]) for v in xw.values()}
def share(modes, keys, tot):
    return sum(modes.get(k, 0) for k in keys) / tot

rows, excluded = [], []
for bcd, modes in agg.items():
    tot = sum(modes.values())
    sgg, nm = names[bcd]
    if tot < MIN_TOTAL:
        excluded.append((nm, sgg, tot))
        continue
    car, bus, sub, walk = (share(modes, k, tot) for k in (CAR, BUS, SUBWAY, WALK))
    rows.append({"동이름": nm, "자치구": sgg, "귀가시간대고령층합계": round(tot, 1),
                 "차량비율": car, "버스비율": bus, "지하철비율": sub, "도보비율": walk,
                 "대중교통비율(버스+지하철)": bus + sub, "우선순위점수": car - (bus + sub),
                 "경계코드": bcd})

by_score = sorted(rows, key=lambda r: -r["우선순위점수"])
by_car = sorted(rows, key=lambda r: -r["차량비율"])
by_pt = sorted(rows, key=lambda r: r["대중교통비율(버스+지하철)"])
for lst, key in ((by_score, "우선순위_순위"), (by_car, "차량비율_순위"), (by_pt, "대중교통낮음_순위")):
    for i, r in enumerate(lst, 1):
        r[key] = i
top_car = {r["경계코드"] for r in by_car[:TOP_REF]}
top_pt = {r["경계코드"] for r in by_pt[:TOP_REF]}
for r in rows:
    r["참고_두조건겹침(20위)"] = "Y" if r["경계코드"] in top_car & top_pt else ""

cols = ["우선순위_순위", "동이름", "자치구", "귀가시간대고령층합계", "우선순위점수", "차량비율", "버스비율",
        "지하철비율", "도보비율", "대중교통비율(버스+지하철)", "차량비율_순위", "대중교통낮음_순위",
        "참고_두조건겹침(20위)", "경계코드"]
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in by_score:
        w.writerow({k: (round(v, 4) if isinstance(v, float) and k != "귀가시간대고령층합계" else v)
                    for k, v in r.items()})

if __name__ == "__main__":
    sums = sorted(sum(m.values()) for m in agg.values())
    p10 = sums[len(sums) // 10]
    print(f"집계 동 {len(agg)}개 / 제외(<{MIN_TOTAL:,}) {len(excluded)}개 / 결과 {len(rows)}개 "
          f"(참고: 하위10% 컷값 {p10:,.0f}, 하위10% 개수 {len(sums) // 10})")
    print("제외된 동:", ", ".join(f"{g} {n}({t / 1000:.0f}k)" for n, g, t in sorted(excluded, key=lambda x: x[2])))
    print(f"\n== 우선순위점수 상위 {TOP_SCORE} (점수 = 차량 - 버스 - 지하철)")
    for r in by_score[:TOP_SCORE]:
        print(f"{r['우선순위_순위']:>2} {r['자치구']} {r['동이름']:<9} 점수{r['우선순위점수']:+.3f} 합계{r['귀가시간대고령층합계']:>8.0f} "
              f"차{r['차량비율']:.1%} 버{r['버스비율']:.1%} 지{r['지하철비율']:.1%} 도{r['도보비율']:.1%}")
    print("\n참고 - 차량비율 순위/대중교통낮음 순위 20위 겹침:",
          [f"{r['자치구']} {r['동이름']}" for r in by_car[:TOP_REF] if r["경계코드"] in top_car & top_pt])
