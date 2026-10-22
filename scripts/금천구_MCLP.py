"""시흥 클러스터 DRT 승차 거점 MCLP(최대커버링입지모델).

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP.py`  (선행: scripts/금천구_목적지후보.py)

수요점: 시흥3·4·5동 70세 이상 인구(SGIS 2024 1km 격자, in_age_015~021)를 100m 하위격자로 배분
  - 1km 격자 인구를 '녹지(OSM 산림·공원)를 뺀 면적' 비례로 하위격자에 나누므로 산지·공원에는 수요가 생기지 않는다.
  - 하위격자 중 3개 동 안에 있는 조각의 대표점 = 수요점, 배분 인구 = 가중치
  - 한계: 녹지 밖은 균등분포 가정(건물·세대 위치 미반영), OSM 녹지 경계 정확도에 의존
후보 승차 정류소(시흥1·3·4·5동 클러스터):
  - 기존 버스정류소(클러스터 내부 + 경계 300m, 같은 동·같은 이름의 방향별 쌍은 1개로 병합)
  - 250m 간격 격자점(녹지 제외, 기존 버스정류소 100m 이내 제거)
  - 경로당·주민센터(고령자가 찾기 쉬운 랜드마크)
고정 목적지 정류소(최적화 대상 아님): 금천구청역 + 시흥1동 병의원·사회복지시설 밀집 지점(아래 HOTSPOT 규칙)
MCLP: max sum(w_i * y_i)  s.t.  y_i <= sum_{j: d_ij<=r} x_j,  sum x_j = p,  x,y binary  (PuLP + CBC)
  - 거리 = EPSG:5179 직선거리(m). 실제 보행 경로(언덕·우회)는 이보다 김
  - r = 300/400/500m, p = 3/5/8 (+ 수확체감 곡선용 p=1~12)
"""
import glob
import io
import json
import zipfile

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pulp
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Patch

RAW, INTERIM, OUT = "data/raw", "data/interim", "outputs"
CRS = 5179
DEMAND_DONGS = {"11180590": "시흥3동", "11180600": "시흥4동", "11180610": "시흥5동"}
CLUSTER_DONGS = {"11180570": "시흥1동", **DEMAND_DONGS}
AGE70 = [f"in_age_{i:03d}" for i in range(15, 22)]  # 70~74 ... 100세 이상 (022 나이미상 제외)
SUB_M = 100            # 수요 하위격자 크기
GRID_SPACING_M = 250   # 후보 격자점 간격
GRID_NEAR_BUS_M = 100  # 기존 버스정류소와 이 거리 이내 격자점은 중복으로 제거
BUS_MERGE_M = 200      # 같은 이름 정류소를 1개로 합치는 최대 거리
HOTSPOT_R = 300        # 목적지 밀집 지점 판단 반경
HOTSPOT_MIN_GAIN = 10  # 두 번째 밀집 지점은 추가로 이만큼 이상의 목적지를 더 묶을 때만 채택
RADII = [300, 400, 500]
PS = [3, 5, 8]
P_CURVE = range(1, 13)
MAP_SCENARIO = (5, 400)  # 지도에 그릴 (p, r)

# ---------------------------------------------------------------- 경계·녹지
bnd = gpd.read_file(f"zip://{RAW}/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(CLUSTER_DONGS)][["ADM_CD", "ADM_NM", "geometry"]]
dem_poly = cl[cl.ADM_CD.isin(DEMAND_DONGS)]
cl_union = cl.union_all()

osm = json.load(open(glob.glob(f"{RAW}/OSM_금천권역_산림공원_*.json")[0], encoding="utf-8"))
gpolys = []
for e in osm["elements"]:
    if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
        gpolys.append(Polygon([(p["lon"], p["lat"]) for p in e["geometry"]]).buffer(0))
    elif e["type"] == "relation":
        ring = lambda role: [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                             for m in e["members"] if m.get("role") == role and m.get("geometry")]
        outer = unary_union(list(polygonize(unary_union(ring("outer")))))
        inner = ring("inner")
        gpolys.append((outer.difference(unary_union(list(polygonize(unary_union(inner))))) if inner else outer).buffer(0))
green = gpd.GeoSeries(gpolys, crs=4326).to_crs(CRS).union_all()

# ---------------------------------------------------------------- 1. 시설 정리(중복 제거)
fac = pd.read_csv(f"{INTERIM}/금천구_목적지후보_상세.csv", encoding="utf-8-sig", dtype={"행정동코드": str})
fac = fac[fac.시흥클러스터.notna() & fac.x_5179.notna()]
keep = fac[~fac.유형.isin(["버스정류소(연계)"])].drop_duplicates(["유형", "시설명", "행정동"])
bus_raw = fac[fac.유형 == "버스정류소(연계)"]
merged = []
# 같은 동 안의 같은 이름(방향별 쌍)만 병합: 동 경계 교차로(예: 은행나무입구사거리)는 동별로 따로 둠
for _, g in bus_raw.groupby(["시설명", "행정동"], dropna=False):
    xy = g[["x_5179", "y_5179"]].to_numpy()
    used = np.zeros(len(g), bool)
    for i in range(len(g)):
        if used[i]:
            continue
        m = (np.hypot(*(xy - xy[i]).T) <= BUS_MERGE_M) & ~used
        used |= m
        sub = g[m]
        merged.append({**sub.iloc[0].to_dict(), "x_5179": sub.x_5179.mean(), "y_5179": sub.y_5179.mean(),
                       "경도": sub.경도.mean(), "위도": sub.위도.mean(), "원본ID": "|".join(sub.원본ID), "병합수": int(m.sum())})
bus = pd.DataFrame(merged)
clean = pd.concat([keep, bus], ignore_index=True)
clean.to_csv(f"{INTERIM}/MCLP_시설_중복제거.csv", index=False, encoding="utf-8-sig")
print("[1] 중복 제거 후:", clean.유형.value_counts().to_dict(), f"(버스 원본 {len(bus_raw)}개 -> {len(bus)}개)")

# ---------------------------------------------------------------- 2. 수요점
P = glob.glob(f"{RAW}/*SGIS*.zip")[0]
z = zipfile.ZipFile(P)
pop_name = [n for n in z.namelist() if "인구_다사_1K" in n][0]
raw = pd.read_csv(io.BytesIO(z.read(pop_name)), encoding="cp949", dtype=str)
raw["v"] = pd.to_numeric(raw.통계값, errors="coerce").fillna(0)
pop70 = raw[raw.통계항목.isin(AGE70)].groupby("격자코드").v.sum()
grid = gpd.read_file(f"zip://{P}!" + [n for n in z.namelist() if n.endswith("grid_다사_1K.shp")][0])
grid = grid.set_crs(CRS, allow_override=True) if grid.crs is None else grid.to_crs(CRS)
cells = grid[grid.intersects(dem_poly.union_all())].copy()
cells["pop70"] = cells.GRID_CD.map(pop70).fillna(0)

subs = []
for c in cells.itertuples():
    x0, y0, _, _ = c.geometry.bounds
    for i in range(10):
        for j in range(10):
            sb = box(x0 + i * SUB_M, y0 + j * SUB_M, x0 + (i + 1) * SUB_M, y0 + (j + 1) * SUB_M)
            subs.append({"GRID_CD": c.GRID_CD, "pop70_cell": c.pop70, "geometry": sb.difference(green)})
subs = gpd.GeoDataFrame(subs, crs=CRS)
subs["hab"] = subs.area
cell_hab = subs.groupby("GRID_CD").hab.transform("sum")
subs["w_sub"] = np.where(cell_hab > 0, subs.pop70_cell * subs.hab / cell_hab, 0)  # 녹지 밖 면적 비례 배분
pieces = gpd.overlay(subs[subs.hab > 0], dem_poly, how="intersection", keep_geom_type=True)
pieces["w"] = pieces.w_sub * pieces.area / pieces.hab
pieces = pieces[pieces.w > 0.01].reset_index(drop=True)
pieces["geometry"] = pieces.geometry.representative_point()
demand = pieces[["GRID_CD", "ADM_NM", "w", "geometry"]].rename(columns={"ADM_NM": "동"})
demand["x"], demand["y"] = demand.geometry.x, demand.geometry.y
# 비교용: 녹지 마스크 없이 균등배분했을 때 3개 동 합계
uni = gpd.overlay(gpd.GeoDataFrame({"pop": cells.pop70.values}, geometry=cells.geometry.values, crs=CRS), dem_poly, how="intersection")
uni_tot = (uni["pop"] * uni.area / 1e6).groupby(uni.ADM_NM).sum()
print(f"[2] 1km 격자 {len(cells)}개 -> 수요점 {len(demand)}개, 70+ 합계 {demand.w.sum():,.0f}명 "
      f"(동별 {demand.groupby('동').w.sum().round(0).to_dict()}; 녹지 마스크 없이 균등배분 시 {uni_tot.round(0).to_dict()})")
d_out = demand.to_crs(4326)
pd.DataFrame({"동": demand.동, "가중치_70세이상": demand.w.round(2), "격자": demand.GRID_CD, "x_5179": demand.x.round(1),
              "y_5179": demand.y.round(1), "위도": d_out.geometry.y.round(6), "경도": d_out.geometry.x.round(6)}
             ).to_csv(f"{OUT}/MCLP_수요점_시흥345_70세.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 3. 후보 승차 정류소
cand_bus = pd.DataFrame({"후보유형": "기존 버스정류소", "이름": bus.시설명, "동": bus.행정동, "x": bus.x_5179, "y": bus.y_5179})
lm = clean[clean.유형.isin(["경로당", "주민센터"]) & (clean.시흥클러스터 == "내부")]
cand_lm = pd.DataFrame({"후보유형": lm.유형, "이름": lm.시설명.str.strip(), "동": lm.행정동, "x": lm.x_5179, "y": lm.y_5179})
xmin, ymin, xmax, ymax = cl_union.bounds
gx, gy = np.meshgrid(np.arange(np.floor(xmin / GRID_SPACING_M) * GRID_SPACING_M, xmax, GRID_SPACING_M),
                     np.arange(np.floor(ymin / GRID_SPACING_M) * GRID_SPACING_M, ymax, GRID_SPACING_M))
gp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS)
gp = gp[gp.within(cl_union) & ~gp.within(green)]
n_grid_all = len(gp)
bxy = bus_raw[["x_5179", "y_5179"]].to_numpy()
near_bus = np.array([np.hypot(*(bxy - (p.x, p.y)).T).min() <= GRID_NEAR_BUS_M for p in gp.geometry])
gp = gp[~near_bus]
gp = gpd.sjoin(gp, cl[["ADM_NM", "geometry"]], how="left", predicate="within")
cand_grid = pd.DataFrame({"후보유형": "격자점", "이름": [f"G{i:03d}" for i in range(len(gp))], "동": gp.ADM_NM.values,
                          "x": gp.geometry.x.values, "y": gp.geometry.y.values})
cand = pd.concat([cand_bus, cand_grid, cand_lm], ignore_index=True)
cand["후보ID"] = [f"C{i:03d}" for i in range(len(cand))]
cg = gpd.GeoSeries(gpd.points_from_xy(cand.x, cand.y), crs=CRS).to_crs(4326)
cand["위도"], cand["경도"] = cg.y.round(6), cg.x.round(6)
cand.to_csv(f"{OUT}/MCLP_후보정류소.csv", index=False, encoding="utf-8-sig")
print(f"[3] 후보 {len(cand)}개 = 기존 버스정류소 {len(cand_bus)} + 격자점 {len(cand_grid)} "
      f"(250m 격자 {n_grid_all}개 중 버스 {GRID_NEAR_BUS_M}m 이내 {near_bus.sum()}개 제거) + 경로당·주민센터 {len(cand_lm)}")

# ---------------------------------------------------------------- 4. 목적지·고정 목적지 정류소
dest = clean[clean.유형.isin(["병의원", "사회복지시설"]) & (clean.시흥클러스터 == "내부")]
dxy = dest[["x_5179", "y_5179"]].to_numpy()
gcs = clean[clean.시설명 == "금천구청역"].iloc[0]
fixed = [{"이름": "금천구청역", "근거": "지하철 환승(경부선)", "x": gcs.x_5179, "y": gcs.y_5179}]
covered = np.hypot(*(dxy - (gcs.x_5179, gcs.y_5179)).T) <= HOTSPOT_R
print(f"[4] 목적지(병의원+사회복지, 클러스터 내부) {len(dest)}개. 금천구청역 {HOTSPOT_R}m 내 {covered.sum()}개")
s1_bus = bus[bus.행정동 == "시흥1동"]
for k in range(2):  # 시흥1동 버스정류소 중, 아직 안 묶인 목적지를 반경 300m 안에 가장 많이 묶는 곳(탐욕적 선택)
    gains = [((np.hypot(*(dxy - (b.x_5179, b.y_5179)).T) <= HOTSPOT_R) & ~covered).sum() for b in s1_bus.itertuples()]
    bi = int(np.argmax(gains))
    if k == 1 and gains[bi] < HOTSPOT_MIN_GAIN:
        print(f"    두 번째 밀집 지점 후보 {s1_bus.iloc[bi].시설명}: 추가 {gains[bi]}개 < {HOTSPOT_MIN_GAIN} -> 채택 안 함")
        break
    b = s1_bus.iloc[bi]
    in_r = np.hypot(*(dxy - (b.x_5179, b.y_5179)).T) <= HOTSPOT_R
    fixed.append({"이름": b.시설명, "근거": f"반경 {HOTSPOT_R}m 내 병의원·사회복지 {in_r.sum()}개(신규 {gains[bi]}개)",
                  "x": b.x_5179, "y": b.y_5179})
    covered |= in_r
    print(f"    밀집 지점 {k + 1}: {b.시설명} (반경 {HOTSPOT_R}m 내 {in_r.sum()}개, 신규 {gains[bi]}개)")
fixed = pd.DataFrame(fixed)
fg = gpd.GeoSeries(gpd.points_from_xy(fixed.x, fixed.y), crs=CRS).to_crs(4326)
fixed["위도"], fixed["경도"] = fg.y.round(6), fg.x.round(6)
fixed["반경300m_목적지수"] = [int((np.hypot(*(dxy - (r.x, r.y)).T) <= 300).sum()) for r in fixed.itertuples()]
fixed.to_csv(f"{OUT}/MCLP_고정목적지정류소.csv", index=False, encoding="utf-8-sig")
print(f"    목적지 정류소 {len(fixed)}곳이 반경 {HOTSPOT_R}m로 묶는 목적지 {covered.sum()}/{len(dest)}개")

# ---------------------------------------------------------------- 5. MCLP
D = np.hypot(demand.x.values[:, None] - cand.x.values[None, :], demand.y.values[:, None] - cand.y.values[None, :])
W = demand.w.values
TOT = W.sum()


def mclp(p, r):
    A = D <= r
    m = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(len(cand)), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
    m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
    for i in range(len(W)):
        js = np.flatnonzero(A[i])
        m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
    m += pulp.lpSum(x.values()) == p
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
    sel = [j for j in range(len(cand)) if x[j].value() > 0.5]
    cov = A[:, sel].any(axis=1)
    return pulp.LpStatus[m.status], sel, W[cov].sum()


upper = {r: W[(D <= r).any(axis=1)].sum() / TOT for r in RADII}
rows, sel_rows, curve = [], [], []
for r in RADII:
    for p in P_CURVE:
        status, sel, cw = mclp(p, r)
        curve.append({"반경_m": r, "p": p, "커버율": cw / TOT, "상태": status})
        if p not in PS:
            continue
        rows.append({"반경_m": r, "거점수_p": p, "커버_70세이상_명": round(cw), "커버율_%": round(cw / TOT * 100, 1),
                     "후보 전체 선택 시 상한_%": round(upper[r] * 100, 1), "최적화상태": status})
        for j in sel:
            c = cand.iloc[j]
            own = (D[:, j] <= r)
            dd = np.hypot(dxy[:, 0] - c.x, dxy[:, 1] - c.y)
            fd = np.hypot(fixed.x - c.x, fixed.y - c.y)
            sel_rows.append({"반경_m": r, "거점수_p": p, "후보ID": c.후보ID, "후보유형": c.후보유형, "이름": c.이름, "동": c.동,
                             "위도": c.위도, "경도": c.경도, "단독커버_70세이상_명": round(W[own].sum()),
                             f"반경내_목적지수": int((dd <= r).sum()),
                             "최근접_목적지정류소": fixed.이름.iloc[int(np.argmin(fd))], "최근접_목적지정류소_거리_m": round(fd.min())})
cmp_ = pd.DataFrame(rows)
curve = pd.DataFrame(curve)
cmp_["p 증가 시 추가 커버율_%p"] = cmp_.groupby("반경_m")["커버율_%"].diff().round(1)
cmp_.to_csv(f"{OUT}/MCLP_비교표.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(sel_rows).to_csv(f"{OUT}/MCLP_선정거점.csv", index=False, encoding="utf-8-sig")
curve.to_csv(f"{OUT}/MCLP_수확체감곡선.csv", index=False, encoding="utf-8-sig")
print(f"\n[5] 수요 {len(W)}점, 70+ {TOT:,.0f}명, 후보 {len(cand)}개")
print(cmp_.to_string(index=False))
print("\n수확체감 곡선(커버율 %):")
print((curve.pivot(index="p", columns="반경_m", values="커버율") * 100).round(1).T.to_string())
print("최적화 상태:", curve.상태.value_counts().to_dict())

# ---------------------------------------------------------------- 6. 그림
SEL, INK, INK2, MUTED, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FIX = "#eb6834"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 6-1 수확체감 곡선
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
cols3 = {300: "#2a78d6", 400: "#1baf7a", 500: "#eda100"}
for r in RADII:
    c = curve[curve.반경_m == r]
    ax.plot(c.p, c.커버율 * 100, color=cols3[r], lw=2, marker="o", ms=4)
for p in PS:
    ax.axvline(p, color="#e6e5e0", lw=1, zorder=0)
ax.set_xticks(list(P_CURVE))
ax.set_xlabel("승차 거점 수 (p)", color=INK2)
ax.set_ylabel("70세 이상 수요 커버율 (%)", color=INK2)
ax.set_ylim(0, 100)
ax.set_title("거점 수·커버 반경별 커버율 (시흥3·4·5동 70세 이상, 직선거리)", loc="left", fontsize=11)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([], [], color=cols3[r], lw=2, label=f"반경 {r}m") for r in RADII], frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/MCLP_수확체감곡선.png", facecolor=SURF)
plt.close(fig)

# 6-2 지도
p0, r0 = MAP_SCENARIO
ms = pd.DataFrame(sel_rows)
ms = ms[(ms.반경_m == r0) & (ms.거점수_p == p0)]
msel = cand.set_index("후보ID").loc[ms.후보ID]
cov0 = (D[:, [cand.index[cand.후보ID == i][0] for i in ms.후보ID]] <= r0).any(axis=1)
fig, ax = plt.subplots(figsize=(9, 9), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
gpd.GeoSeries([green.intersection(cl_union.buffer(400))], crs=CRS).plot(ax=ax, color="#e3efe3", zorder=0)
cl.boundary.plot(ax=ax, color=INK2, lw=1, zorder=1)
LABEL_SHIFT = {"시흥4동": (420, 80)}  # 고정 정류소 라벨과 겹치지 않게
for r in cl.itertuples():
    pt = r.geometry.representative_point()
    dx, dy = LABEL_SHIFT.get(r.ADM_NM, (0, 0))
    ax.text(pt.x + dx, pt.y - 120 + dy, r.ADM_NM, fontsize=12, color=INK2, ha="center", zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
ax.scatter(cand.x, cand.y, s=6, color="#c3c2b7", zorder=2, label="후보 정류소")
ax.scatter(demand.x[~cov0], demand.y[~cov0], s=demand.w[~cov0] * 1.2, color="#898781", alpha=0.6, zorder=3, linewidths=0)
ax.scatter(demand.x[cov0], demand.y[cov0], s=demand.w[cov0] * 1.2, color="#86b6ef", alpha=0.9, zorder=3, linewidths=0)
for c in msel.itertuples():
    ax.add_patch(Circle((c.x, c.y), r0, facecolor=SEL, alpha=0.08, edgecolor=SEL, lw=1.2, zorder=4))
ax.scatter(msel.x, msel.y, s=90, color=SEL, edgecolor=SURF, linewidth=2, zorder=7)
for c in msel.itertuples():
    ax.annotate(("격자점 " if c.후보유형 == "격자점" else "") + c.이름, (c.x, c.y), xytext=(6, 6), textcoords="offset points",
                fontsize=8.5, color=INK, zorder=8, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
ax.scatter(fixed.x, fixed.y, s=180, marker="*", color=FIX, edgecolor=SURF, linewidth=1, zorder=7)
for f in fixed.itertuples():
    ax.annotate(f.이름, (f.x, f.y), xytext=(7, -12), textcoords="offset points", fontsize=8.5, color=INK, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
pad = 250
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
cv = cmp_[(cmp_.반경_m == r0) & (cmp_.거점수_p == p0)]["커버율_%"].iloc[0]
ax.set_title(f"DRT 승차 거점 {p0}곳 (반경 {r0}m) — 시흥3·4·5동 70세 이상 수요 {cv}% 커버", loc="left", fontsize=12)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=9, label=f"선정 승차 거점 (반경 {r0}m 원)"),
                   Line2D([], [], marker="*", ls="", color=FIX, ms=13, label="고정 목적지 정류소"),
                   Line2D([], [], marker="o", ls="", color="#86b6ef", ms=7, label="수요점 (커버됨, 크기=70세 이상 인구)"),
                   Line2D([], [], marker="o", ls="", color="#898781", ms=7, label="수요점 (미커버)"),
                   Line2D([], [], marker="o", ls="", color="#c3c2b7", ms=4, label="후보 정류소"),
                   Patch(color="#e3efe3", label="녹지(OSM 산림·공원, 수요 제외)")],
          frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "수요: SGIS 2024 1km 격자 70세 이상을 녹지 제외 100m 하위격자로 배분. 거리: 직선거리. 배경 녹지: © OpenStreetMap contributors",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/MCLP_거점지도_p{p0}_r{r0}.png", facecolor=SURF)
plt.close(fig)
print(f"\n저장: {OUT}/MCLP_비교표.csv, MCLP_선정거점.csv, MCLP_수확체감곡선.csv/png, MCLP_거점지도_p{p0}_r{r0}.png, "
      f"MCLP_후보정류소.csv, MCLP_수요점_시흥345_70세.csv, MCLP_고정목적지정류소.csv")
