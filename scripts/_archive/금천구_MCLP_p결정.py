"""반경 300m 고정, 승차 거점 수 p를 기준으로 결정 (선행: scripts/금천구_MCLP.py 의 수요점·후보·고정목적지 출력).

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP_p결정.py`
기준
  A. 커버율 95% 이상을 처음 달성하는 p
  B. 다음 거점 1곳의 추가 커버가 2%p 미만으로 떨어지기 직전의 p (= 추가 커버 >= 2%p인 마지막 p)
  -> 둘 중 큰 p로 거점 표·지도 저장
미커버 수요 묶음: 미커버 수요점 중 서로 150m 이내(100m 하위격자 이웃)끼리 연결한 덩어리, 인원 많은 순
순회 거리: 선정 거점(+고정 목적지 정류소 3곳)을 한 번씩 도는 최단 순회의 직선거리 근사(최근접 이웃 + 2-opt).
  실제 도로 거리는 이보다 길다(보통 1.2~1.4배). 5km 초과 여부를 경고로 출력
"""
import glob
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pulp
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

RAW, OUT = "data/raw", "outputs"
R = 300
P_MAX = 15
TARGET_COVER = 0.95
MIN_GAIN = 0.02
CHECK_P = 8           # 미커버 수요를 볼 p
LINK_M = 150          # 미커버 수요점 묶음 연결 거리
TSP_WARN_M = 5000
CLUSTER_DONGS = {"11180570": "시흥1동", "11180590": "시흥3동", "11180600": "시흥4동", "11180610": "시흥5동"}

dem = pd.read_csv(f"{OUT}/MCLP_수요점_시흥345_70세.csv", encoding="utf-8-sig")
cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
dest = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.시흥클러스터 == "내부")]
bus = fac[fac.유형 == "버스정류소(연계)"]
W = dem.가중치_70세이상.to_numpy()
TOT = W.sum()
D = np.hypot(dem.x_5179.values[:, None] - cand.x.values[None, :], dem.y_5179.values[:, None] - cand.y.values[None, :])
A = D <= R


def mclp(p):
    m = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(len(cand)), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
    m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
    for i in range(len(W)):
        js = np.flatnonzero(A[i])
        m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
    m += pulp.lpSum(x.values()) == p
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    sel = [j for j in range(len(cand)) if x[j].value() > 0.5]
    return pulp.LpStatus[m.status], sel


def tour_len(xy):
    """최근접 이웃으로 시작해 2-opt로 개선한 닫힌 순회 길이(m). 모든 시작점 중 최솟값."""
    n = len(xy)
    if n < 2:
        return 0.0
    dm = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    best = np.inf
    for s in range(n):
        route, left = [s], set(range(n)) - {s}
        while left:
            nxt = min(left, key=lambda k: dm[route[-1], k])
            route.append(nxt)
            left.remove(nxt)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    a, b, c, d = route[i - 1], route[i], route[k], route[(k + 1) % n]
                    if dm[a, c] + dm[b, d] < dm[a, b] + dm[c, d] - 1e-9:
                        route[i:k + 1] = route[i:k + 1][::-1]
                        improved = True
        best = min(best, sum(dm[route[i], route[(i + 1) % n]] for i in range(n)))
    return best


fxy = fixed[["x", "y"]].to_numpy()
rows, sols = [], {}
for p in range(1, P_MAX + 1):
    status, sel = mclp(p)
    sols[p] = sel
    cw = W[A[:, sel].any(axis=1)].sum()
    sxy = cand.loc[sel, ["x", "y"]].to_numpy()
    rows.append({"p": p, "커버_명": cw, "커버율": cw / TOT, "상태": status,
                 "순회_승차거점만_km": tour_len(sxy) / 1000, "순회_승차+목적지3곳_km": tour_len(np.vstack([sxy, fxy])) / 1000})
t = pd.DataFrame(rows)
t["추가커버_명"] = t.커버_명.diff().fillna(t.커버_명)
t["추가커버_%p"] = t.커버율.diff().fillna(t.커버율) * 100
t["다음 p 추가커버_%p"] = t["추가커버_%p"].shift(-1)

pA = int(t.loc[t.커버율 >= TARGET_COVER, "p"].min())
pB = int(t.loc[t["다음 p 추가커버_%p"] < MIN_GAIN * 100, "p"].min())
P_SEL = max(pA, pB)
out = t.assign(커버_명=t.커버_명.round(0), 커버율_pct=(t.커버율 * 100).round(1), 추가커버_명=t.추가커버_명.round(0),
               **{"추가커버_%p": t["추가커버_%p"].round(1), "순회_승차거점만_km": t.순회_승차거점만_km.round(2),
                  "순회_승차+목적지3곳_km": t["순회_승차+목적지3곳_km"].round(2)})
out = out[["p", "커버_명", "커버율_pct", "추가커버_명", "추가커버_%p", "순회_승차거점만_km", "순회_승차+목적지3곳_km", "상태"]]
out.to_csv(f"{OUT}/MCLP_r300_p1-15.csv", index=False, encoding="utf-8-sig")
print(f"[1] 반경 {R}m, 수요 {TOT:,.0f}명")
print(out.to_string(index=False))
print(f"\n[2] 기준 A(커버율 {TARGET_COVER:.0%} 최초) p = {pA} | 기준 B(다음 추가커버 {MIN_GAIN:.0%}p 미만 직전) p = {pB} -> 채택 p = {P_SEL}")
over = t[t["순회_승차+목적지3곳_km"] > TSP_WARN_M / 1000]
print(f"    순회 {TSP_WARN_M / 1000:.0f}km 초과(승차+목적지 기준) 처음 나오는 p:", int(over.p.min()) if len(over) else "없음",
      "| 승차거점만 기준:", int(t.loc[t.순회_승차거점만_km > TSP_WARN_M / 1000, "p"].min()) if (t.순회_승차거점만_km > TSP_WARN_M / 1000).any() else "없음")

# ---- [3] p=8 미커버 수요 묶음
unc = ~A[:, sols[CHECK_P]].any(axis=1)
u = dem[unc].reset_index(drop=True)
uxy = u[["x_5179", "y_5179"]].to_numpy()
lab = -np.ones(len(u), int)
c = 0
for i in range(len(u)):
    if lab[i] >= 0:
        continue
    stack, lab[i] = [i], c
    while stack:
        k = stack.pop()
        nb = np.flatnonzero((np.hypot(*(uxy - uxy[k]).T) <= LINK_M) & (lab < 0))
        lab[nb] = c
        stack.extend(nb)
    c += 1
u["묶음"] = lab
sxy8 = cand.loc[sols[CHECK_P], ["x", "y"]].to_numpy()
bxy = bus[["x_5179", "y_5179"]].to_numpy()
grp = []
for g, gg in u.groupby("묶음"):
    w = gg.가중치_70세이상
    cx, cy = np.average(gg.x_5179, weights=w), np.average(gg.y_5179, weights=w)
    nb = int(np.argmin(np.hypot(*(bxy - (cx, cy)).T)))
    grp.append({"묶음": g, "인원_명": w.sum(), "수요점수": len(gg), "동": "/".join(gg.동.value_counts().index),
                "근처_버스정류소": bus.iloc[nb].시설명, "근처정류소까지_m": np.hypot(*(bxy[nb] - (cx, cy))),
                "가장가까운_선정거점까지_m": np.hypot(*(sxy8 - (cx, cy)).T).min(), "x": cx, "y": cy})
grp = pd.DataFrame(grp).sort_values("인원_명", ascending=False).reset_index(drop=True)
grp["누적비중_%"] = (grp.인원_명.cumsum() / grp.인원_명.sum() * 100).round(1)
gl = gpd.GeoSeries(gpd.points_from_xy(grp.x, grp.y), crs=5179).to_crs(4326)
grp["위도"], grp["경도"] = gl.y.round(6), gl.x.round(6)
grp_out = grp.assign(인원_명=grp.인원_명.round(0), 근처정류소까지_m=grp.근처정류소까지_m.round(0),
                     가장가까운_선정거점까지_m=grp.가장가까운_선정거점까지_m.round(0)).drop(columns=["x", "y"])
grp_out.to_csv(f"{OUT}/MCLP_r300_p{CHECK_P}_미커버묶음.csv", index=False, encoding="utf-8-sig")
print(f"\n[3] p={CHECK_P} 미커버 {W[unc].sum():,.0f}명 ({unc.sum()}점) / 동별:", u.groupby("동").가중치_70세이상.sum().round(0).to_dict())
print(grp_out.head(8).to_string(index=False))

# ---- [4] 채택 p 거점 표·지도
sel = sols[P_SEL]
dxy = dest[["x_5179", "y_5179"]].to_numpy()
cov_by = A[:, sel]
srows = []
for k, j in enumerate(sel):
    cc = cand.iloc[j]
    fd = np.hypot(fixed.x - cc.x, fixed.y - cc.y)
    others = np.delete(cov_by, k, axis=1).any(axis=1)
    srows.append({"후보ID": cc.후보ID, "후보유형": cc.후보유형, "이름": cc.이름, "동": cc.동, "위도": cc.위도, "경도": cc.경도,
                  "단독커버_명": round(W[cov_by[:, k]].sum()), "고유커버_명(이 거점만 커버)": round(W[cov_by[:, k] & ~others].sum()),
                  "반경내_목적지수": int((np.hypot(dxy[:, 0] - cc.x, dxy[:, 1] - cc.y) <= R).sum()),
                  "최근접_목적지정류소": fixed.이름.iloc[int(np.argmin(fd))], "최근접_목적지정류소_거리_m": round(fd.min())})
sdf = pd.DataFrame(srows).sort_values(["동", "단독커버_명"], ascending=[True, False])
sdf.to_csv(f"{OUT}/MCLP_r300_p{P_SEL}_선정거점.csv", index=False, encoding="utf-8-sig")
print(f"\n[4] p={P_SEL} 선정 거점 (커버율 {t.loc[t.p == P_SEL, '커버율'].iloc[0]:.1%})")
print(sdf.drop(columns=["후보ID", "위도", "경도"]).to_string(index=False))

# 지도
bnd = gpd.read_file(f"zip://{RAW}/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(CLUSTER_DONGS)]
osm = json.load(open(glob.glob(f"{RAW}/OSM_금천권역_산림공원_*.json")[0], encoding="utf-8"))
gpolys = []
for e in osm["elements"]:
    if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
        gpolys.append(Polygon([(q["lon"], q["lat"]) for q in e["geometry"]]).buffer(0))
    elif e["type"] == "relation":
        ring = lambda role: [LineString([(q["lon"], q["lat"]) for q in m["geometry"]])
                             for m in e["members"] if m.get("role") == role and m.get("geometry")]
        outer = unary_union(list(polygonize(unary_union(ring("outer")))))
        inner = ring("inner")
        gpolys.append((outer.difference(unary_union(list(polygonize(unary_union(inner))))) if inner else outer).buffer(0))
green = gpd.GeoSeries(gpolys, crs=4326).to_crs(5179).union_all()

SEL, INK, INK2, MUTED, SURF, FIX = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#eb6834"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
covered = A[:, sel].any(axis=1)
fig, ax = plt.subplots(figsize=(9, 9), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
gpd.GeoSeries([green.intersection(cl.union_all().buffer(400))], crs=5179).plot(ax=ax, color="#e3efe3", zorder=0)
cl.boundary.plot(ax=ax, color=INK2, lw=1, zorder=1)
shift = {"시흥4동": (420, 80), "시흥3동": (330, 330), "시흥5동": (330, -120)}
for r in cl.itertuples():
    pt = r.geometry.representative_point()
    dx, dy = shift.get(r.ADM_NM, (0, 0))
    ax.text(pt.x + dx, pt.y - 120 + dy, r.ADM_NM, fontsize=12, color=INK2, ha="center", zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
ax.scatter(dem.x_5179[~covered], dem.y_5179[~covered], s=W[~covered] * 1.2, color=MUTED, alpha=0.7, lw=0, zorder=3)
ax.scatter(dem.x_5179[covered], dem.y_5179[covered], s=W[covered] * 1.2, color="#86b6ef", alpha=0.9, lw=0, zorder=3)
ss = cand.loc[sel]
for c_ in ss.itertuples():
    ax.add_patch(Circle((c_.x, c_.y), R, facecolor=SEL, alpha=0.08, edgecolor=SEL, lw=1.1, zorder=4))
ax.scatter(ss.x, ss.y, s=80, color=SEL, edgecolor=SURF, linewidth=2, zorder=7)
for c_ in ss.itertuples():
    # 오른쪽 350m 안에 다른 거점이 있으면 라벨을 왼쪽에 둔다(겹침 방지)
    left = ((ss.x > c_.x) & (np.hypot(ss.x - c_.x, ss.y - c_.y) < 350)).any()
    ax.annotate(("격자점 " if c_.후보유형 == "격자점" else "") + c_.이름, (c_.x, c_.y), xytext=(-6, 5) if left else (6, 5),
                textcoords="offset points", ha="right" if left else "left", fontsize=7.5, color=INK, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
ax.scatter(fixed.x, fixed.y, s=170, marker="*", color=FIX, edgecolor=SURF, linewidth=1, zorder=7)
for f in fixed.itertuples():
    ax.annotate(f.이름, (f.x, f.y), xytext=(7, -12), textcoords="offset points", fontsize=8, color=INK, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
xmin, ymin, xmax, ymax = cl.total_bounds
ax.set_xlim(xmin - 250, xmax + 250)
ax.set_ylim(ymin - 250, ymax + 250)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
cv = t.loc[t.p == P_SEL, "커버율"].iloc[0]
ax.set_title(f"DRT 승차 거점 {P_SEL}곳 (반경 {R}m) — 시흥3·4·5동 70세 이상 수요 {cv:.1%} 커버", loc="left", fontsize=12)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=9, label=f"선정 승차 거점 (반경 {R}m 원)"),
                   Line2D([], [], marker="*", ls="", color=FIX, ms=13, label="고정 목적지 정류소"),
                   Line2D([], [], marker="o", ls="", color="#86b6ef", ms=7, label="수요점 (커버됨, 크기=70세 이상 인구)"),
                   Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="수요점 (미커버)"),
                   Patch(color="#e3efe3", label="녹지(OSM 산림·공원, 수요 제외)")], frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "수요: SGIS 2024 1km 격자 70세 이상을 녹지 제외 100m 하위격자로 배분. 거리: 직선거리. 배경 녹지: © OpenStreetMap contributors",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/MCLP_거점지도_p{P_SEL}_r{R}.png", facecolor=SURF)
plt.close(fig)
print(f"\n저장: {OUT}/MCLP_r300_p1-15.csv, MCLP_r300_p{CHECK_P}_미커버묶음.csv, MCLP_r300_p{P_SEL}_선정거점.csv, MCLP_거점지도_p{P_SEL}_r{R}.png")
