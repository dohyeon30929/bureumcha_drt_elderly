"""2단계 권역별 목적지 점검 + 최종 정류소 16곳 비대칭 도로거리 행렬.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_권역.py`  (선행: 금천구_MCLP_최종.py)
- 승차 11곳: outputs/MCLP_최종_p11_승차정류소.csv. 권역 A = 시흥3동, 권역 B = 시흥4·5동
- 목적지 5곳: 금천구청역, 시흥사거리, 현대시장.시흥4동주민센터, 무지개아파트(시흥1동 쪽 정류소), 석수역(1호선, 안양시)
- 목적지 매력도: 정류소 300m 안 병의원·사회복지시설 수(중복 제거, 클러스터 내부 시설). 석수역은 서울 밖이라
  안양시 쪽 시설이 데이터에 없음 -> 금천구 쪽 시설만 세어 과소평가됨(한계)
- 모든 정류소를 정차 가능 도로에 스냅해 행렬 계산(행=출발, 열=도착, m). 승차 정류소 좌표는 이미 도로 위
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

OUT = "outputs"
DEST_R = 300
ZONE = {"시흥3동": "A", "시흥4동": "B", "시흥5동": "B"}

b = pd.read_csv(f"{OUT}/MCLP_최종_p11_승차정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig")
mu = fac[(fac.시설명 == "무지개아파트") & (fac.행정동 == "시흥1동")].iloc[0]
ss = fac[(fac.시설명 == "석수역") & (fac.유형 == "지하철역(연계)")].iloc[0]
dest_list = [(r.이름, r.x, r.y, "시흥1동") for r in fixed.itertuples()] + [
    ("무지개아파트", mu.x_5179, mu.y_5179, "시흥1동"), ("석수역", ss.x_5179, ss.y_5179, "안양시(서울 밖)")]

b = b.sort_values(["동", "배정_70세이상_명"], ascending=[True, False]).reset_index(drop=True)
stops = pd.concat([
    pd.DataFrame({"구분": "승차", "권역": b.동.map(ZONE), "이름": b.표시이름, "원이름": b.원이름, "후보유형": b.후보유형, "동": b.동,
                  "x": b.x_5179, "y": b.y_5179, "배정_70세이상_명": b.배정_70세이상_명}),
    pd.DataFrame({"구분": "목적지", "권역": "", "이름": [d[0] for d in dest_list], "원이름": [d[0] for d in dest_list],
                  "후보유형": "목적지 정류소", "동": [d[3] for d in dest_list], "x": [d[1] for d in dest_list],
                  "y": [d[2] for d in dest_list], "배정_70세이상_명": 0})], ignore_index=True)
stops.insert(0, "정류소ID", [f"S{i + 1:02d}" for i in range(len(stops))])

dest = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.시흥클러스터 == "내부")]
dxy = dest[["x_5179", "y_5179"]].to_numpy()
stops["반경300m_목적지수"] = [int((np.hypot(*(dxy - (x, y)).T) <= DEST_R).sum()) if g == "목적지" else None
                         for g, x, y in zip(stops.구분, stops.x, stops.y)]
covered = np.zeros(len(dest), bool)
for r in stops[stops.구분 == "목적지"].itertuples():
    covered |= np.hypot(*(dxy - (r.x, r.y)).T) <= DEST_R
s1 = (dest.행정동 == "시흥1동").to_numpy()
print(f"목적지 정류소 5곳 300m 안: 시흥1동 목적지 {covered[s1].sum()}/{s1.sum()} ({covered[s1].mean():.1%}), "
      f"클러스터 전체 {covered.sum()}/{len(dest)} ({covered.mean():.1%})")

roads = Roads()
snaps = [roads.snap(x, y) for x, y in zip(stops.x, stops.y)]
stops["스냅거리_m"] = [round(s["스냅거리_m"], 1) for s in snaps]
stops["x_5179_스냅"], stops["y_5179_스냅"] = [s["sx"] for s in snaps], [s["sy"] for s in snaps]
ll = gpd.GeoSeries(gpd.points_from_xy(stops.x_5179_스냅, stops.y_5179_스냅), crs=5179).to_crs(4326)
stops["위도"], stops["경도"] = ll.y.round(6), ll.x.round(6)
M = roads.drive_matrix(snaps)
ids = stops.정류소ID.tolist()
pd.DataFrame(np.round(M), index=ids, columns=ids).to_csv(f"{OUT}/DRT_도로거리행렬_16x16_m.csv", encoding="utf-8-sig")
stops.drop(columns=["x", "y"]).to_csv(f"{OUT}/DRT_정류소16.csv", index=False, encoding="utf-8-sig")
off = ~np.eye(len(ids), dtype=bool)
print(f"행렬 {len(ids)}x{len(ids)}: 도달 불가 {np.isinf(M[off]).sum()}칸, 왕복 비대칭 >100m "
      f"{(np.abs(M - M.T)[np.triu_indices(len(ids), 1)] > 100).sum()}쌍")

# ---- 권역별 승차 -> 목적지 도로거리
bi = stops.index[stops.구분 == "승차"]
di = stops.index[stops.구분 == "목적지"]
tab = pd.DataFrame(np.round(M[np.ix_(bi, di)]), columns=stops.loc[di, "이름"])
tab.insert(0, "승차 정류소", stops.loc[bi, "이름"].values)
tab.insert(0, "권역", stops.loc[bi, "권역"].values)
tab.insert(2, "배정_70세이상_명", stops.loc[bi, "배정_70세이상_명"].values)
tab.to_csv(f"{OUT}/DRT_권역별_승차to목적지_m.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n승차 -> 목적지 도로거리(m):")
print(tab.to_string(index=False))
summ = []
for z, g in tab.groupby("권역"):
    w = g.배정_70세이상_명
    for d in stops.loc[di, "이름"]:
        summ.append({"권역": z, "목적지": d, "인구가중평균_m": round(np.average(g[d], weights=w)), "최대_m": int(g[d].max())})
summ = pd.DataFrame(summ)
summ = summ.merge(stops[stops.구분 == "목적지"][["이름", "반경300m_목적지수"]].rename(columns={"이름": "목적지"}), on="목적지")
summ.to_csv(f"{OUT}/DRT_권역별_목적지요약.csv", index=False, encoding="utf-8-sig")
print("\n권역별 요약 (배정 고령인구 가중 평균 도로거리):")
print(summ.pivot(index="목적지", columns="권역", values="인구가중평균_m").join(summ.drop_duplicates("목적지").set_index("목적지").반경300m_목적지수).to_string())
print("\n정류소 16곳:")
print(stops[["정류소ID", "구분", "권역", "이름", "동", "배정_70세이상_명", "반경300m_목적지수", "스냅거리_m"]].to_string(index=False))
