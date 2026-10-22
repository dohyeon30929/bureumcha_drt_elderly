"""최종_이동취약지대.csv: 취약지대 후보 9곳을 '확정 7곳'과 '참고 2곳'으로 나누고 모든 지표를 한 행에 모은 마스터 테이블.

- 확정: 접근성 하위 25% 이내(독산2동, 시흥3동, 시흥5동, 시흥4동, 신월7동, 신월1동, 방화2동)
- 참고: 접근성 하위권 밖이라 '인프라는 있는데 안 타는' 후보(시흥1동, 신월2동). 추가조사 보류
- 중심좌표: 접근성 계산과 동일한 방식(폴리곤 중심점, 밖이면 대표점). 미터 좌표(EPSG:5179)와 WGS84 둘 다 제공.
  DRT 거점 배치/MCLP 모델의 수요점·거리 계산 입력용(거리 계산은 x_5179,y_5179를 쓸 것).
"""
import geopandas as gpd
import pandas as pd

CONFIRMED = ["독산2동", "시흥3동", "시흥5동", "시흥4동", "신월7동", "신월1동", "방화2동"]
REFERENCE = ["시흥1동", "신월2동"]
OUT = "data/interim/재현확인/최종_이동취약지대.csv"

v = pd.read_csv("data/interim/재현확인/취약지대_후보_9곳.csv", encoding="utf-8-sig", dtype={"경계코드": str})
assert set(v["동이름"]) == set(CONFIRMED + REFERENCE)
main = pd.read_csv("data/interim/재현확인/귀가시간대_우선순위_교차검증.csv", encoding="utf-8-sig", dtype={"경계코드": str})
acc = pd.read_csv("data/interim/재현확인/접근성_지수_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
ver = pd.read_csv("data/interim/재현확인/접근성_검증_9곳.csv", encoding="utf-8-sig")
raw = pd.read_csv("data/interim/재현확인/접근성_지표_행정동.csv", encoding="utf-8-sig", dtype={"경계코드": str})

t = v[["경계코드"]].merge(
    main[["경계코드", "동이름", "자치구", "우선순위_순위", "우선순위점수", "차량비율", "버스비율", "지하철비율", "도보비율",
          "대중교통비율(버스+지하철)", "귀가시간대고령층합계", "귀가비중", "소득분위", "가구수", "소득구분"]], on="경계코드")
t = t.merge(acc[["경계코드", "접근성지수", "접근성순위", "접근성_하위%", "버스정류소밀도_개per_km2", "마을버스제외밀도",
                 "지하철최단거리_m", "최근접역", "지하철역수_1km"]], on="경계코드")
t = t.merge(ver[["동이름", "판정"]].rename(columns={"판정": "접근성_판정"}), on="동이름")
t = t.merge(raw[["경계코드", "면적_km2", "버스정류소수", "중심점방식"]], on="경계코드")

# 중심좌표 (접근성 계산과 동일 방식)
g = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
g["ADM_NM"] = g["ADM_NM"].str.strip()
g = g[g["ADM_CD"].isin(t["경계코드"])].set_index("ADM_CD")
cen = g.geometry.centroid
cen = cen.where(cen.within(g.geometry), g.geometry.representative_point())
pts = gpd.GeoSeries(cen, crs=g.crs)
ll = pts.to_crs("EPSG:4326")
coord = pd.DataFrame({"x_5179": pts.x.round(1), "y_5179": pts.y.round(1),
                      "경도_WGS84": ll.x.round(6), "위도_WGS84": ll.y.round(6)}).reset_index().rename(columns={"ADM_CD": "경계코드"})
t = t.merge(coord, on="경계코드")

t["구분"] = t["동이름"].map(lambda n: "확정" if n in CONFIRMED else "참고")
t["비고"] = t["구분"].map({"확정": "접근성 하위 25% 이내 + 차량의존 + 소득 낮음(<=6분위)",
                         "참고": "접근성 하위권 밖(인프라는 있는데 안 타는 후보), 추가조사 보류"})
t["구분_순서"] = t["구분"].map({"확정": 0, "참고": 1})
t = t.sort_values(["구분_순서", "접근성지수"]).drop(columns="구분_순서")

cols = ["구분", "동이름", "자치구", "경계코드", "우선순위_순위", "우선순위점수", "차량비율", "버스비율", "지하철비율", "도보비율",
        "대중교통비율(버스+지하철)", "귀가시간대고령층합계", "귀가비중", "소득분위", "가구수", "소득구분",
        "접근성지수", "접근성순위", "접근성_하위%", "접근성_판정", "버스정류소수", "버스정류소밀도_개per_km2", "마을버스제외밀도",
        "지하철최단거리_m", "최근접역", "지하철역수_1km", "면적_km2", "x_5179", "y_5179", "경도_WGS84", "위도_WGS84",
        "중심점방식", "비고"]
t = t[cols]
t.to_csv(OUT, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print(f"저장: {OUT}  ({len(t)}행 x {len(t.columns)}열)  확정 {int((t['구분'] == '확정').sum())} / 참고 {int((t['구분'] == '참고').sum())}")
print("결측 컬럼:", t.columns[t.isna().any()].tolist() or "없음")
print(t[["구분", "자치구", "동이름", "우선순위_순위", "우선순위점수", "차량비율", "소득분위", "접근성지수", "접근성_하위%", "귀가비중",
         "경도_WGS84", "위도_WGS84", "중심점방식"]].round(3).to_string(index=False))
