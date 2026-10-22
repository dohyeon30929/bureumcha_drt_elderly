"""1단계: 행정동(경계 426개)별 대중교통 접근성 원지표 계산.

입력: bnd_dong_11_2025_2Q.zip (EPSG:5179), 서울시버스정류소위치정보(*.xlsx, WGS84 경도/위도),
      서울시 역사마스터 정보.csv (cp949, WGS84 위도/경도, 수도권 784행), 행정동_코드매핑_생활이동_경계.csv(자치구명)
좌표: 버스/지하철 WGS84(EPSG:4326) -> 경계와 같은 EPSG:5179(미터)로 변환해서 계산한다.
지표:
  - 버스정류소수(한강선착장 제외), 마을버스 제외 정류소수, 면적당 밀도(개/km2)
  - 지하철역 최단거리(행정동 중심점 기준, m), 반경 500m / 1km 내 역 수
    * 환승역은 같은 역사명이면서 500m 이내인 행을 1개 역으로 합침
    * 중심점이 폴리곤 밖(오목한 동)이면 폴리곤 안의 대표점을 쓰고 '중심점방식'에 표시
    * 지하철은 서울 밖 수도권 역도 포함(경계 인근 동의 최단거리 왜곡 방지)
출력: 접근성_지표_행정동.csv
"""
import csv
import glob

import geopandas as gpd
import numpy as np
import pandas as pd

CRS_M = "EPSG:5179"
MERGE_M = 500
OUT = "data/interim/재현확인/접근성_지표_행정동.csv"

# ---------- 경계
dong = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
assert dong.crs.to_epsg() == 5179
dong["ADM_NM"] = dong["ADM_NM"].str.strip()
dong["면적_km2"] = dong.geometry.area / 1e6
gu = {r["bnd_admdong_cd"]: r["sigungu"]
      for r in csv.DictReader(open("data/interim/재현확인/행정동_코드매핑_생활이동_경계.csv", encoding="utf-8-sig"))}
dong["자치구"] = dong["ADM_CD"].map(gu)

# ---------- 버스정류소
b = pd.read_excel(glob.glob("data/raw/서울시버스정류소위치정보*.xlsx")[0], dtype={"NODE_ID": str, "ARS_ID": str})
n_all = len(b)
b = b[b["정류소타입"] != "한강선착장"].copy()
bus = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b["X좌표"], b["Y좌표"]), crs="EPSG:4326").to_crs(CRS_M)
j = gpd.sjoin(bus, dong[["ADM_CD", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]  # 경계선 위 점이 두 폴리곤에 걸리면 첫 번째만
unassigned = j["ADM_CD"].isna()
n_un = int(unassigned.sum())
if n_un:  # 한강·경계 틈에 걸린 점은 100m 이내 최근접 동으로 배정
    near = gpd.sjoin_nearest(bus.loc[j.index[unassigned]], dong[["ADM_CD", "geometry"]], how="left",
                             max_distance=100, distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    j.loc[near.index, "ADM_CD"] = near["ADM_CD"]
n_out = int(j["ADM_CD"].isna().sum())
cnt = j.groupby("ADM_CD").size()
cnt_nv = j[j["정류소타입"] != "마을버스"].groupby("ADM_CD").size()
dong["버스정류소수"] = dong["ADM_CD"].map(cnt).fillna(0).astype(int)
dong["버스정류소수_마을버스제외"] = dong["ADM_CD"].map(cnt_nv).fillna(0).astype(int)
dong["버스정류소밀도_개per_km2"] = dong["버스정류소수"] / dong["면적_km2"]

# ---------- 지하철역 (환승역 병합)
s = pd.read_csv(glob.glob("data/raw/서울시 역사마스터*.csv")[0], encoding="cp949")
sg = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["경도"], s["위도"]), crs="EPSG:4326").to_crs(CRS_M)
sg["x"], sg["y"] = sg.geometry.x, sg.geometry.y
stations = []
for name, g in sg.groupby("역사명"):
    pts = g[["x", "y"]].to_numpy()
    used = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        m = (np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]) <= MERGE_M) & ~used
        used |= m
        stations.append((name, pts[m, 0].mean(), pts[m, 1].mean()))
st = pd.DataFrame(stations, columns=["역사명", "x", "y"])
print(f"역사마스터 {len(s)}행 -> 고유 역 {len(st)}개 (환승·중복 병합 {len(s) - len(st)}행)")

cen = dong.geometry.centroid
inside = cen.within(dong.geometry)
cen = cen.where(inside, dong.geometry.representative_point())
dong["중심점방식"] = np.where(inside, "centroid", "대표점(중심점이 폴리곤 밖)")
cx, cy = cen.x.to_numpy(), cen.y.to_numpy()
D = np.hypot(cx[:, None] - st["x"].to_numpy()[None, :], cy[:, None] - st["y"].to_numpy()[None, :])
dong["지하철최단거리_m"] = D.min(axis=1).round(0)
dong["지하철역수_500m"] = (D <= 500).sum(axis=1)
dong["지하철역수_1km"] = (D <= 1000).sum(axis=1)
dong["최근접역"] = st["역사명"].to_numpy()[D.argmin(axis=1)]

out = dong.rename(columns={"ADM_CD": "경계코드", "ADM_NM": "동이름"})[
    ["경계코드", "동이름", "자치구", "면적_km2", "버스정류소수", "버스정류소수_마을버스제외", "버스정류소밀도_개per_km2",
     "지하철최단거리_m", "최근접역", "지하철역수_500m", "지하철역수_1km", "중심점방식"]].copy()
out[["면적_km2", "버스정류소밀도_개per_km2"]] = out[["면적_km2", "버스정류소밀도_개per_km2"]].round(3)
out.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"버스: 전체 {n_all} - 한강선착장 {n_all - len(b)} = {len(b)}개 사용 | 경계 내부 직접 배정 {len(b) - n_un}, "
      f"경계 밖/틈이라 100m 최근접 재배정 시도 {n_un} (그래도 미배정 {n_out}) | 동에 배정된 합계 {int(dong['버스정류소수'].sum())}")
print(f"중심점이 폴리곤 밖이라 대표점 사용한 동: {int((~inside).sum())}개")
print("버스 0개인 동:", out.loc[out["버스정류소수"] == 0, "동이름"].tolist())

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    num = ["버스정류소수", "버스정류소수_마을버스제외", "버스정류소밀도_개per_km2", "지하철최단거리_m", "지하철역수_500m", "지하철역수_1km"]
    print("\n== 서울 426개 동 분포")
    print(out[num + ["면적_km2"]].describe(percentiles=[.25, .5, .75]).round(1).T[["min", "25%", "50%", "75%", "max"]].to_string())
    v = pd.read_csv("data/interim/재현확인/취약지대_후보_9곳.csv", encoding="utf-8-sig", dtype={"경계코드": str})
    m = v[["우선순위_순위", "경계코드"]].merge(out, on="경계코드")
    print("\n== 취약지대 후보 9곳 원지표")
    print(m[["우선순위_순위", "자치구", "동이름", "면적_km2", "버스정류소수", "버스정류소수_마을버스제외",
             "버스정류소밀도_개per_km2", "지하철최단거리_m", "최근접역", "지하철역수_500m", "지하철역수_1km"]].to_string(index=False))
