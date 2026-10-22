"""행정동별 고령인구(65세 이상) 추정: SGIS 2024 1km 격자 인구 -> 행정동 경계에 면적비례 배분.

- 65세 이상 = in_age_014(65~69) ~ in_age_021(100세 이상) 합 (in_age_022 '나이미상'은 제외; 코드집 statistics_code.xlsx로 확인)
- 격자: grid_다사_1K (EPSG:5179, 1km 셀). 서울 25개 구는 전부 '다사' 타일(adm_grid_mapping으로 확인)
- 배분: 셀 인구 x (셀과 동 폴리곤 교차면적 / 셀 전체면적). 셀 안 인구가 균등하다는 가정이라 산지·하천·공원이
  섞인 셀에서는 오차가 크다(셀 크기 1km2 ~ 행정동 평균 1.4km2). 셀 경계 밖(서울 밖) 부분은 버려진다.
출력: 행정동별_고령인구.csv
"""
import glob
import io
import warnings
import zipfile

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")
OLD_ITEMS = [f"in_age_{i:03d}" for i in range(14, 22)]
ALL_ITEMS = [f"in_age_{i:03d}" for i in range(1, 23)]
OUT = "data/interim/재현확인/행정동별_고령인구.csv"

P = glob.glob("data/raw/국가데이터처_SGIS 격자*.zip")[0]
z = zipfile.ZipFile(P)
def nm(i):
    try:
        return i.filename.encode("cp437").decode("cp949")
    except Exception:
        return i.filename
ent = {nm(i): i for i in z.infolist() if not i.is_dir()}
pop_path = [n for n in ent if "인구_다사_1K" in n][0]
raw = pd.read_csv(io.BytesIO(z.read(ent[pop_path].filename)), encoding="cp949", dtype=str)
print("인구 CSV 컬럼:", list(raw.columns), " 행수:", len(raw), " 기준연도:", raw["기준연도"].unique().tolist())
val = pd.to_numeric(raw["통계값"], errors="coerce")
print("통계값 비숫자(마스킹) 행:", int(val.isna().sum()), "  예:", raw.loc[val.isna(), "통계값"].unique()[:5].tolist())
raw["v"] = val.fillna(0)

pv = raw[raw["통계항목"].isin(ALL_ITEMS)].pivot_table(index="격자코드", columns="통계항목", values="v", aggfunc="sum").fillna(0)
cell = pd.DataFrame({"고령인구": pv[[c for c in OLD_ITEMS if c in pv.columns]].sum(axis=1),
                     "총인구": pv[[c for c in ALL_ITEMS if c in pv.columns]].sum(axis=1)}).reset_index().rename(columns={"격자코드": "GRID_CD"})
print(f"격자 인구: {len(cell)}셀, 65+ 합계(타일 전체) {cell['고령인구'].sum():,.0f}, 총인구 {cell['총인구'].sum():,.0f}")

shp = [i.filename for i in z.infolist() if i.filename.endswith("grid_다사_1K.shp")][0]
grid = gpd.read_file(f"zip://{P}!{shp}", encoding="utf-8").merge(cell, on="GRID_CD", how="left").fillna({"고령인구": 0, "총인구": 0})
dong = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
dong["ADM_NM"] = dong["ADM_NM"].str.strip()
assert grid.crs == dong.crs

# 서울과 만나는 셀만 사용
seoul_union = dong.union_all()
g = grid[grid.intersects(seoul_union)].copy()
g["셀면적"] = g.geometry.area
inter = gpd.overlay(g[["GRID_CD", "고령인구", "총인구", "셀면적", "geometry"]], dong[["ADM_CD", "geometry"]], how="intersection", keep_geom_type=True)
inter["frac"] = inter.geometry.area / inter["셀면적"]
inter["old"] = inter["고령인구"] * inter["frac"]
inter["tot"] = inter["총인구"] * inter["frac"]
est = inter.groupby("ADM_CD")[["old", "tot"]].sum()

# ---- 검증: 질량 보존
cell_in_seoul = g[g.geometry.centroid.within(seoul_union)]
covered = inter.groupby("GRID_CD")["frac"].sum()
print(f"\n서울과 만나는 셀 {len(g)}개 (중심이 서울 안 {len(cell_in_seoul)}개)")
print(f"서울 폴리곤 총면적 {dong.geometry.area.sum() / 1e6:.1f}km2 (실제 서울 약 605km2 -> 한강 포함 여부 확인)")
print(f"중심이 서울 안인 셀의 65+ 합계 {cell_in_seoul['고령인구'].sum():,.0f} vs 동 배분 합계 {est['old'].sum():,.0f} "
      f"(배분율 {est['old'].sum() / cell_in_seoul['고령인구'].sum():.1%}; 셀의 서울 밖 부분은 버림)")
print(f"셀 커버율(서울 폴리곤이 셀을 덮는 비율): 중심이 서울 안인 셀 중 90% 미만 {int((covered.reindex(cell_in_seoul['GRID_CD']).fillna(0) < 0.9).sum())}개")

out = dong.rename(columns={"ADM_CD": "경계코드", "ADM_NM": "동이름"})[["경계코드", "동이름"]].copy()
out["면적_km2"] = (dong.geometry.area / 1e6).round(3).to_numpy()
out["고령인구_추정"] = out["경계코드"].map(est["old"]).fillna(0).round(0)
out["총인구_추정"] = out["경계코드"].map(est["tot"]).fillna(0).round(0)
out["고령화율"] = (out["고령인구_추정"] / out["총인구_추정"]).round(4)
out["고령인구밀도_명per_km2"] = (out["고령인구_추정"] / out["면적_km2"]).round(0)
n_cells = inter.groupby("ADM_CD")["GRID_CD"].nunique()
out["걸친_셀수"] = out["경계코드"].map(n_cells).fillna(0).astype(int)
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n저장: {OUT} ({len(out)}행). 서울 65+ 추정 합계 {out['고령인구_추정'].sum():,.0f}, 총인구 {out['총인구_추정'].sum():,.0f}, 고령화율 {out['고령인구_추정'].sum() / out['총인구_추정'].sum():.1%}")
print("동별 고령인구 분포:", out["고령인구_추정"].describe(percentiles=[.05, .25, .5, .75]).round(0).to_dict())
print("고령인구 100 미만 동:", out.loc[out["고령인구_추정"] < 100, ["동이름", "고령인구_추정"]].values.tolist())
