"""MCLP v3: 수요점을 SGIS 100m 격자 총인구 + 주민등록 동별 70세 이상으로 보정해 재생성, 보행 300/400/500m 민감도.

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP_v3.py`  (TSP·시뮬레이션은 하지 않음)
v2 대비 변경: 수요점만 바뀜. 후보지·개수 결정 기준·보행망은 v2(금천구_MCLP_보행.py)와 같다.
수요점
  - data/raw/2024년_인구_다사_100M.csv (cp949, 헤더 없음: 연도, 격자코드, 항목, 값). to_in_001 = 총인구 사용(연령별 없음)
  - 격자코드 '다사'+xxx+yyy -> 좌하단 (900000 + xxx*100, 1900000 + yyy*100), EPSG:5179. 1km 격자와 같은 원점(검증: 1km 총인구와 상관 1.0)
  - 100m 칸을 시흥3·4·5동 경계와 교차, 경계에 걸친 칸은 면적 비율로 총인구를 나눔
  - 칸(조각)의 70세 이상 = 동 주민등록 70세 이상 x (조각 총인구 / 동 안 조각 총인구 합) -> 동별 합계가 주민등록과 정확히 일치
  - 주민등록 70세 이상(2026.9, 사용자 제공): 시흥3동 1,864 / 시흥4동 3,431 / 시흥5동 3,522
  - 한계: 격자 값이 0, 5, 8, 13, 18 … 처럼 비공개 처리(소수 인구 칸 = 5)되어 소인구 칸이 과대, 총인구 분포를 70세 이상 분포로 가정
"""
import glob
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pulp
from shapely.geometry import box

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "outputs"
DONGS = {"11180590": "시흥3동", "11180600": "시흥4동", "11180610": "시흥5동"}
REG70 = {"시흥3동": 1864, "시흥4동": 3431, "시흥5동": 3522}
ORIGIN = (900000, 1900000)
RADII, BASE_R = [300, 400, 500], 400
P_MAX = 30
TARGET, MIN_GAIN, NEAR_BUS = 0.95, 0.02, 100

# ---------------------------------------------------------------- 1. 수요점
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
dp = bnd[bnd.ADM_CD.isin(DONGS)][["ADM_NM", "geometry"]]
h = pd.read_csv("data/raw/2024년_인구_다사_100M.csv", encoding="cp949", dtype=str, header=None, names=["연도", "격자", "항목", "값"])
h = h[h.항목 == "to_in_001"]
h["총인구"] = h.값.astype(int)
h["x0"] = ORIGIN[0] + h.격자.str[2:5].astype(int) * 100
h["y0"] = ORIGIN[1] + h.격자.str[5:8].astype(int) * 100
xmin, ymin, xmax, ymax = dp.total_bounds
h = h[(h.x0 + 100 >= xmin) & (h.x0 <= xmax) & (h.y0 + 100 >= ymin) & (h.y0 <= ymax)]
cells = gpd.GeoDataFrame(h[["격자", "총인구"]], geometry=[box(x, y, x + 100, y + 100) for x, y in zip(h.x0, h.y0)], crs=5179)
pc = gpd.overlay(cells, dp, how="intersection", keep_geom_type=True)
pc["면적비"] = pc.area / 1e4
pc["총인구_조각"] = pc.총인구 * pc.면적비
pc = pc[pc.총인구_조각 > 0].reset_index(drop=True)
pc["w"] = pc.총인구_조각 / pc.groupby("ADM_NM").총인구_조각.transform("sum") * pc.ADM_NM.map(REG70)
pc["geometry"] = pc.geometry.representative_point()
dem = pd.DataFrame({"동": pc.ADM_NM, "격자": pc.격자, "총인구_조각": pc.총인구_조각.round(2), "가중치_70세이상": pc.w,
                    "x_5179": pc.geometry.x, "y_5179": pc.geometry.y})
ll = gpd.GeoSeries(pc.geometry, crs=5179).to_crs(4326)
dem["위도"], dem["경도"] = ll.y.round(6), ll.x.round(6)
dem.round({"가중치_70세이상": 3, "x_5179": 1, "y_5179": 1}).to_csv(f"{OUT}/MCLP_v3_수요점_100m보정.csv", index=False, encoding="utf-8-sig")
n_cells = cells[cells.intersects(dp.union_all())]
print(f"[1] 100m 칸 {len(n_cells)}개가 3개 동과 겹침 -> 총인구>0 조각 {len(dem)}개 (경계 분할 포함)")
print("    칸 총인구 값 분포(3개 동):", n_cells.총인구.value_counts().head(8).to_dict(), f"| 값=5 칸 {(n_cells.총인구 == 5).mean():.0%}")

old = pd.read_csv(f"{OUT}/MCLP_수요점_시흥345_70세_v2.csv", encoding="utf-8-sig")
cmp_d = pd.DataFrame({"v2_면적배분+녹지제외": old.groupby("동").가중치_70세이상.sum(), "v3_100m보정": dem.groupby("동").가중치_70세이상.sum(),
                      "주민등록_2026.9": pd.Series(REG70), "v2_수요점수": old.groupby("동").size(), "v3_수요점수": dem.groupby("동").size()})
cmp_d.loc["합계"] = cmp_d.sum()
cmp_d["v2_주민등록대비_%"] = ((cmp_d["v2_면적배분+녹지제외"] / cmp_d["주민등록_2026.9"] - 1) * 100).round(1)
cmp_d = cmp_d.round(0)
cmp_d.to_csv(f"{OUT}/MCLP_v3_수요점_동별비교.csv", encoding="utf-8-sig")
print("\n동별 70세 이상 합계 비교:")
print(cmp_d.to_string())
# 같은 100m 격자 위치끼리 분포 비교 (v2 수요점도 같은 원점의 100m 하위격자)
key = lambda x, y: (np.floor(x / 100) * 100).astype(int).astype(str) + "_" + (np.floor(y / 100) * 100).astype(int).astype(str)
o = old.assign(k=key(old.x_5179, old.y_5179)).groupby(["동", "k"]).가중치_70세이상.sum()
n = dem.assign(k=key(dem.x_5179, dem.y_5179)).groupby(["동", "k"]).가중치_70세이상.sum()
both = pd.concat([o.rename("v2"), n.rename("v3")], axis=1).fillna(0)
top = lambda s, q: s.sort_values(ascending=False).cumsum().le(s.sum() * q).sum() + 1
print(f"    100m 위치별 상관 {both.corr().iloc[0, 1]:.2f} | 수요 50%가 몰린 칸 수: v2 {top(both.v2, .5)} / v3 {top(both.v3, .5)}"
      f" | v2에만 수요 있는 칸 {((both.v2 > 0) & (both.v3 == 0)).sum()}, v3에만 {((both.v3 > 0) & (both.v2 == 0)).sum()}")

# ---------------------------------------------------------------- 2·3. 후보지 + 보행 MCLP (반경 300/400/500)
cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
raw_bus = pd.read_csv("data/interim/금천구_목적지후보_상세.csv", encoding="utf-8-sig")
raw_bus = raw_bus[(raw_bus.유형 == "버스정류소(연계)") & raw_bus.시흥클러스터.notna()]
roads = Roads()
c = cand.copy()
c["도로명"] = None
g = c.후보유형 == "격자점"
for i in c.index[g]:
    s = roads.snap(c.at[i, "x"], c.at[i, "y"])
    c.loc[i, ["x", "y", "도로명"]] = [s["sx"], s["sy"], s["도로명"]]
bxy = raw_bus[["x_5179", "y_5179"]].to_numpy()
c = c[~(np.array([np.hypot(*(bxy - (x, y)).T).min() <= NEAR_BUS for x, y in zip(c.x, c.y)]) & g)].reset_index(drop=True)
W = dem.가중치_70세이상.to_numpy()
TOT = W.sum()
dxy = dem[["x_5179", "y_5179"]].to_numpy()
cxy = c[["x", "y"]].to_numpy()
WD = roads.walk_dist(cxy, dxy, max(RADII))
print(f"\n[2] 후보 {len(c)}개, 수요 {len(W)}점 {TOT:,.0f}명")


def mclp(A, p):
    m = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(A.shape[1]), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
    m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
    for i in range(len(W)):
        js = np.flatnonzero(A[i])
        m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
    m += pulp.lpSum(x.values()) == p
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    return pulp.LpStatus[m.status], [j for j in range(A.shape[1]) if x[j].value() > 0.5]


res, curves = {}, []
for r in RADII:
    A = WD <= r
    ub = W[A.any(axis=1)].sum() / TOT
    rows, sols = [], {}
    for p in range(1, P_MAX + 1):
        st, sel = mclp(A, p)
        sols[p] = sel
        rows.append({"반경_m": r, "p": p, "커버_명": W[A[:, sel].any(axis=1)].sum(), "상태": st})
        if rows[-1]["커버_명"] >= TOT * min(ub, 0.999) - 1e-6 and p >= 20:
            break
    t = pd.DataFrame(rows)
    t["커버율"] = t.커버_명 / TOT
    t["추가커버_명"] = t.커버_명.diff().fillna(t.커버_명)
    t["추가커버_%p"] = t.커버율.diff().fillna(t.커버율) * 100
    nxt = t["추가커버_%p"].shift(-1)
    pa = t.loc[t.커버율 >= TARGET, "p"]
    pA = int(pa.min()) if len(pa) else None
    pB = int(t.loc[nxt < MIN_GAIN * 100, "p"].min())
    psel = max(v for v in (pA, pB) if v is not None)
    res[r] = {"A": A, "ub": ub, "pA": pA, "pB": pB, "p": psel, "sel": sols[psel], "sols": sols, "t": t}
    curves.append(t)
    print(f"    보행 {r}m: 상한 {ub:.1%}, 기준A {pA} / 기준B {pB} -> p = {psel} ({t.loc[t.p == psel, '커버율'].iloc[0]:.1%})")
cur = pd.concat(curves)
cur_out = cur.assign(커버_명=cur.커버_명.round(0), 커버율_pct=(cur.커버율 * 100).round(1), 추가커버_명=cur.추가커버_명.round(0),
                     **{"추가커버_%p": cur["추가커버_%p"].round(1)})[["반경_m", "p", "커버_명", "커버율_pct", "추가커버_명", "추가커버_%p", "상태"]]
cur_out.to_csv(f"{OUT}/MCLP_v3_보행_p곡선.csv", index=False, encoding="utf-8-sig")
print("\n보행 400m p=1~20:")
print(cur_out[(cur_out.반경_m == BASE_R) & (cur_out.p <= 20)].drop(columns="반경_m").to_string(index=False))

# ---------------------------------------------------------------- 반경 민감도 표
base_ids = set(c.후보ID.iloc[res[BASE_R]["sel"]])
sens = []
for r in RADII:
    sel = res[r]["sel"]
    A = res[r]["A"]
    cov = A[:, sel].any(axis=1)
    cov400 = (WD[:, sel] <= BASE_R).any(axis=1)
    row = {"반경_m": r, "정류소수": len(sel), "기준A(95%)": res[r]["pA"], "기준B(2%p)": res[r]["pB"], "커버율_%": round(W[cov].sum() / TOT * 100, 1),
           "보행400m로 잰 커버율_%": round(W[cov400].sum() / TOT * 100, 1), "후보전체 상한_%": round(res[r]["ub"] * 100, 1)}
    for d in REG70:
        row[f"{d}_정류소"] = int((c.동.iloc[sel] == d).sum())
    row["시흥1동 등 기타_정류소"] = int((~c.동.iloc[sel].isin(REG70)).sum())
    for d in REG70:
        row[f"{d}_미커버_명"] = round(W[~cov & (dem.동 == d).to_numpy()].sum())
    row["미커버_합계_명"] = round(W[~cov].sum())
    row["400m안과 겹치는 정류소"] = len(set(c.후보ID.iloc[sel]) & base_ids)
    sens.append(row)
sens = pd.DataFrame(sens)
sens.to_csv(f"{OUT}/MCLP_v3_반경민감도.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n[3] 반경 민감도:")
print(sens.T.to_string())

# ---------------------------------------------------------------- 4. 이전 최종안(v2, 14곳)과 비교
v2 = pd.read_csv(f"{OUT}/MCLP_보행400_p14_승차정류소_v2.csv", encoding="utf-8-sig")
v2xy = v2[["x_5179", "y_5179"]].to_numpy()
v2cov_new = (roads.walk_dist(v2xy, dxy, BASE_R) <= BASE_R).any(axis=1)
sel = res[BASE_R]["sel"]
s = c.iloc[sel].copy()
s_ids = set(s.후보ID)
kept = v2[v2.후보ID.isin(s_ids)]
dropped = v2[~v2.후보ID.isin(s_ids)]
added = s[~s.후보ID.isin(set(v2.후보ID))]
sxy = s[["x", "y"]].to_numpy()


def nearest(xy, pool_xy, pool_names):
    d = np.hypot(pool_xy[:, 0] - xy[0], pool_xy[:, 1] - xy[1])
    k = int(np.argmin(d))
    return pool_names[k], round(d[k])


fb = bus[["x_5179", "y_5179"]].to_numpy()


def nm(r):
    if r.후보유형 != "격자점":
        return r.이름.strip()
    k = int(np.argmin(np.hypot(*(fb - (r.x, r.y)).T)))
    d = np.hypot(*(fb[k] - (r.x, r.y)))
    parts = [r.도로명 if isinstance(r.도로명, str) else None, f"{bus.iloc[k].시설명} 정류소 {d:.0f}m" if d <= 250 else None]
    return " · ".join(p for p in parts if p) or f"{r.동} 무명도로"


s["표시이름"] = [nm(r) for r in s.itertuples()]
s["표시이름"] = s.표시이름.replace({"석수역": "석수역 버스정류소(승차)"})
moves = []
for r in dropped.itertuples():
    nn, dd = nearest((r.x_5179, r.y_5179), sxy, s.표시이름.tolist())
    moves.append({"변화": "빠짐", "정류소": r.표시이름, "동": r.동, "가장 가까운 새 정류소": nn, "거리_m": dd})
for r in added.itertuples():
    nn, dd = nearest((r.x, r.y), v2xy, v2.표시이름.tolist())
    moves.append({"변화": "추가", "정류소": s.loc[r.Index, "표시이름"], "동": r.동, "가장 가까운 이전 정류소": nn, "거리_m": dd})
mv = pd.DataFrame(moves)
mv.to_csv(f"{OUT}/MCLP_v3_vs_v2_정류소변화.csv", index=False, encoding="utf-8-sig")
cov_new = res[BASE_R]["A"][:, sel].any(axis=1)
v2_tot_old = 337
cmp_rows = []
for lab, cnt, cv, unc in [("v2 최종(14곳, 면적배분 수요로 평가)", len(v2), 95.8, {"시흥3동": 106, "시흥4동": 80, "시흥5동": 151}),
                          ("v2 최종 14곳을 새 수요로 평가", len(v2), round(W[v2cov_new].sum() / TOT * 100, 1),
                           {d: round(W[~v2cov_new & (dem.동 == d).to_numpy()].sum()) for d in REG70}),
                          (f"v3 새 400m안({len(sel)}곳)", len(sel), round(W[cov_new].sum() / TOT * 100, 1),
                           {d: round(W[~cov_new & (dem.동 == d).to_numpy()].sum()) for d in REG70})]:
    row = {"안": lab, "정류소수": cnt, "보행400m_커버율_%": cv}
    src = v2 if "v2" in lab else s
    for d in REG70:
        row[f"{d}_정류소"] = int((src.동 == d).sum())
    for d in REG70:
        row[f"{d}_미커버_명"] = unc[d]
    row["미커버_합계_명"] = sum(unc.values())
    cmp_rows.append(row)
cmp_v = pd.DataFrame(cmp_rows)
cmp_v.to_csv(f"{OUT}/MCLP_v3_vs_v2_비교.csv", index=False, encoding="utf-8-sig")
print(f"\n[4] 이전 최종안 대비: 유지 {len(kept)}곳, 빠짐 {len(dropped)}곳, 추가 {len(added)}곳")
print(cmp_v.to_string(index=False))
print(mv.to_string(index=False))

# ---------------------------------------------------------------- 5. 최종 목록
WS = np.where(WD[:, sel] <= BASE_R, WD[:, sel], np.inf)
assign = np.where(np.isfinite(WS).any(axis=1), WS.argmin(axis=1), -1)
s["배정_70세이상_명"] = [round(W[assign == k].sum()) for k in range(len(sel))]
s["단독커버_명"] = [round(W[np.isfinite(WS[:, k])].sum()) for k in range(len(sel))]
TYPE = {"기존 버스정류소": "기존 버스정류소", "격자점": "신규 지점", "경로당": "경로당·주민센터", "주민센터": "경로당·주민센터"}
s["유형"] = s.후보유형.map(TYPE)
s["v2와"] = np.where(s.후보ID.isin(set(v2.후보ID)), "유지", "신규")
ll = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), crs=5179, index=s.index).to_crs(4326)
s["위도"], s["경도"] = ll.y.round(6), ll.x.round(6)
s = s.sort_values(["동", "배정_70세이상_명"], ascending=[True, False]).reset_index(drop=True)
s.insert(0, "ID", [f"B{i + 1:02d}" for i in range(len(s))])
fin = s[["ID", "표시이름", "동", "유형", "배정_70세이상_명", "단독커버_명", "v2와", "후보ID", "위도", "경도", "x", "y"]].rename(
    columns={"표시이름": "정류소명", "x": "x_5179", "y": "y_5179"})
fin.to_csv(f"{OUT}/MCLP_v3_최종승차정류소.csv", index=False, encoding="utf-8-sig")
print(f"\n[5] 최종 승차 정류소 ({len(fin)}곳, 배정 합계 {fin.배정_70세이상_명.sum():,}명)")
print(fin[["ID", "정류소명", "동", "유형", "배정_70세이상_명", "v2와"]].to_string(index=False))

# ---------------------------------------------------------------- 지도
cl = bnd[bnd.ADM_CD.isin(["11180570", *DONGS])]
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
SEL, OLD, COV = "#2a78d6", "#eb6834", "#86b6ef"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
ext = np.r_[cl.total_bounds[:2] - 200, cl.total_bounds[2:] + 200]


def base(ax, title):
    ax.set_facecolor(SURF)
    cl.boundary.plot(ax=ax, color="#c3c2b7", lw=1, zorder=1)
    for r in cl.itertuples():
        pt = r.geometry.representative_point()
        ax.text(pt.x - 250, pt.y + 150, r.ADM_NM, fontsize=10, color=MUTED, ha="center", zorder=2,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)


def demand_layer(ax, cov, scale=1.5):
    ax.scatter(dem.x_5179[~cov], dem.y_5179[~cov], s=W[~cov] * scale, color=MUTED, alpha=0.7, lw=0, zorder=3)
    ax.scatter(dem.x_5179[cov], dem.y_5179[cov], s=W[cov] * scale, color=COV, alpha=0.9, lw=0, zorder=3)


fig, ax = plt.subplots(figsize=(9, 10), dpi=160, facecolor=SURF)
base(ax, f"보행 {BASE_R}m 기준 승차 정류소 {len(fin)}곳 (v3, 수요 = 100m 총인구 x 주민등록 70세 이상) — 커버율 {W[cov_new].sum() / TOT:.1%}")
demand_layer(ax, cov_new)
ax.scatter(v2xy[:, 0], v2xy[:, 1], s=70, facecolor="none", edgecolor=OLD, lw=1.6, zorder=5)
ax.scatter(fin.x_5179, fin.y_5179, s=70, color=SEL, edgecolor=SURF, lw=1.5, zorder=6)
for r in fin.itertuples():
    ax.annotate(r.ID, (r.x_5179, r.y_5179), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=9, label=f"v3 승차 정류소 ({len(fin)}곳)"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=OLD, ms=9, label="v2 정류소 (14곳)"),
                   Line2D([], [], marker="o", ls="", color=COV, ms=7, label=f"수요점: 보행 {BASE_R}m 커버 (크기=70세 이상)"),
                   Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="수요점: 미커버")], frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "보행 도로거리: OSM 보행망(경사 미반영). 정류소 번호는 MCLP_v3_최종승차정류소.csv 참조. © OpenStreetMap contributors", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/MCLP_v3_거점지도_보행400.png", facecolor=SURF)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), dpi=140, facecolor=SURF)
for ax, r in zip(axes, RADII):
    sel_r = res[r]["sel"]
    cov_r = res[r]["A"][:, sel_r].any(axis=1)
    base(ax, f"보행 {r}m: {len(sel_r)}곳, 커버율 {W[cov_r].sum() / TOT:.1%}")
    demand_layer(ax, cov_r, 1.0)
    in400 = c.후보ID.iloc[sel_r].isin(base_ids).to_numpy()
    xy = cxy[sel_r]
    ax.scatter(xy[in400, 0], xy[in400, 1], s=55, color=SEL, edgecolor=SURF, lw=1.2, zorder=6)
    ax.scatter(xy[~in400, 0], xy[~in400, 1], s=55, color=OLD, edgecolor=SURF, lw=1.2, zorder=6)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=8, label="400m안에도 있는 정류소"),
                        Line2D([], [], marker="o", ls="", color=OLD, ms=8, label="이 반경에만 있는 정류소"),
                        Line2D([], [], marker="o", ls="", color=COV, ms=6, label="커버된 수요점"),
                        Line2D([], [], marker="o", ls="", color=MUTED, ms=6, label="미커버 수요점")], frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "개수 기준: 커버율 95% 최초 달성 p와 추가 커버 2%p 미만 직전 p 중 큰 값. 보행 도로거리(경사 미반영). © OpenStreetMap contributors",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/MCLP_v3_반경비교지도.png", facecolor=SURF)
plt.close(fig)
print("\n저장: MCLP_v3_수요점_100m보정.csv, MCLP_v3_수요점_동별비교.csv, MCLP_v3_보행_p곡선.csv, MCLP_v3_반경민감도.csv, "
      "MCLP_v3_vs_v2_비교.csv, MCLP_v3_vs_v2_정류소변화.csv, MCLP_v3_최종승차정류소.csv, MCLP_v3_거점지도_보행400.png, MCLP_v3_반경비교지도.png")
