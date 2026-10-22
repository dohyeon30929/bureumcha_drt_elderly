"""보조 검증: 한 달치(2026년 8월 31일치) 생활이동 목적별 파일로 시흥1동 도착 70세 이상 시간대 분포를 뽑아
DRT 가는 호출 시간 분포(O-D 2026-08-31 하루, 60세 이상, 시흥3·4·5동 -> 시흥1동 출발 시각)와 비교.

실행: 프로젝트 루트에서 `python scripts/금천구_시흥1동도착_검증.py`
입력: data/raw/수도권 생활이동 (성 연령별, 도착지 기준)-내국인.zip (seoul_purpose_admdong1_in_202608DD.csv 31개: 도착 동·도착 시각·목적·성연령)
      outputs/OD검증_시간대별_시흥1동_왕복_60세이상.csv (시뮬레이션 가는 호출 분포의 출처)
비교 방식
  - 목적별 파일은 출발 동이 없음 -> '시흥1동에 도착하는 70세 이상 전체(출발지 무관)'. 귀가(move_purpose=3)는 집으로 돌아오는 시흥1동 주민이라
    가는 이동 비교에서 빼고(귀가 제외 = 가는 이동 근사), 귀가만 따로도 집계
  - 시각: 목적별 = 도착 시각, O-D = 출발 시각(동네 안 짧은 이동이라 대부분 같은 시간대). time_cd 20분 단위는 시 단위로 합침
  - 평일만 / 31일 전체 둘 다. 분포 비교는 시뮬레이션 접수 시간(8~16시) 안 비중으로 정규화
"""
import csv
import glob
import io
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = "outputs"
S1 = "11545670"  # 시흥1동(행안부 코드)
E70 = ("male_70_cnt", "feml_70_cnt")
PUR = {"1": "출근", "2": "등교", "3": "귀가", "4": "쇼핑", "5": "관광", "6": "병원", "7": "기타"}

zp = [p for p in glob.glob("data/raw/*.zip") if "성 연령별, 도착지 기준" in p][0]
agg = defaultdict(float)  # (일자, 시, 목적) -> 70+
with zipfile.ZipFile(zp) as z:
    names = sorted(z.namelist())
    for name in names:
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
                if r["d_admdong_cd"] != S1:
                    continue
                t = int(r["time_cd"])
                h = t // 100 if t >= 100 else t
                agg[(r["etl_ymd"], h, r["move_purpose"])] += sum(float(r[c] or 0) for c in E70)
a = pd.Series(agg).rename_axis(["일자", "시", "목적"]).reset_index(name="v")
a["평일"] = pd.to_datetime(a.일자).dt.dayofweek < 5
ndays = a.일자.nunique()
nwd = a.loc[a.평일, "일자"].nunique()

tab = pd.DataFrame(index=range(24))
tab["도착70_귀가제외_평일평균"] = a[a.평일 & (a.목적 != "3")].groupby("시").v.sum() / nwd
tab["도착70_귀가제외_전체평균"] = a[a.목적 != "3"].groupby("시").v.sum() / ndays
tab["도착70_귀가만_평일평균"] = a[a.평일 & (a.목적 == "3")].groupby("시").v.sum() / nwd
for p, nm in PUR.items():
    if p != "3":
        tab[f"평일_{nm}"] = a[a.평일 & (a.목적 == p)].groupby("시").v.sum() / nwd
od = pd.read_csv(f"{OUT}/OD검증_시간대별_시흥1동_왕복_60세이상.csv", encoding="utf-8-sig").set_index("출발시")
tab["OD_가는_3·4·5동→시흥1동_60세이상(8/31)"] = od["가는_3·4·5동→시흥1동"]
tab = tab.fillna(0)
tab.index.name = "시"

W = range(8, 17)  # 시뮬레이션 가는 호출 접수 8~16시
cmp_cols = {"목적별 도착 70+ 귀가 제외(평일 평균)": "도착70_귀가제외_평일평균", "목적별 도착 70+ 귀가 제외(31일 평균)": "도착70_귀가제외_전체평균",
            "O-D 가는 60+ (8/31, 시뮬레이션 입력)": "OD_가는_3·4·5동→시흥1동_60세이상(8/31)"}
share = pd.DataFrame({k: tab.loc[W, c] / tab.loc[W, c].sum() * 100 for k, c in cmp_cols.items()})
tab.round(2).to_csv(f"{OUT}/검증_시흥1동도착_70세_시간대.csv", encoding="utf-8-sig")
share.round(1).to_csv(f"{OUT}/검증_시흥1동도착_vs_OD_8_16시비중.csv", encoding="utf-8-sig")

ref = share["O-D 가는 60+ (8/31, 시뮬레이션 입력)"]
summ = []
for k in share:
    s = share[k]
    summ.append({"분포": k, "8~16시 중 8~9시_%": s.loc[8:9].sum(), "10~13시_%": s.loc[10:13].sum(), "14~16시_%": s.loc[14:16].sum(), "최대 시": int(s.idxmax()),
                 "O-D와 상관": np.corrcoef(s, ref)[0, 1], "O-D와 차이 절대합/2_%p": (s - ref).abs().sum() / 2,
                 "하루 전체 중 8~16시 비중_%": tab.loc[W, cmp_cols[k]].sum() / tab[cmp_cols[k]].sum() * 100})
summ = pd.DataFrame(summ)
summ.round(2).to_csv(f"{OUT}/검증_시흥1동도착_요약.csv", index=False, encoding="utf-8-sig")
pur = {nm: tab[f"평일_{nm}"].sum() for p, nm in PUR.items() if p != "3"}
pd.set_option("display.width", 220)
print(f"파일 {len(names)}개, 일수 {ndays}(평일 {nwd})")
print("8~16시 안 시간대 비중(%):\n", share.round(1).to_string())
print(summ.round(2).to_string(index=False))
print("평일 하루 평균 시흥1동 도착 70+ 목적(귀가 제외):", {k: round(v) for k, v in pur.items()})
