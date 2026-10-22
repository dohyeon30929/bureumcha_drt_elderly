"""MCLP v4: 경로당·주민센터 정차 위치 보정 + 성긴 지역 후보 추가 + 동별 최소 커버율 제약 + 기존 정류소만 비교 + 목적지 보행 400m.

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP_v4.py`  (TSP 없음. 수요점은 v3: MCLP_v3_수요점_100m보정.csv)
1. 경로당·주민센터: 시설에서 '걸어서 가장 가까운' 정차 가능 도로(residential 이상, 주행망 최대 강연결) 지점으로 이동.
   아파트 단지 안이면 단지 내 보행로를 따라 나가는 출입구 쪽 도로가 선택됨. 50m 초과 이동 표시
2. 동별 상한 진단: 후보 전부 선택 시 보행 400m 밖 수요를
   '도로가 닿지 않음'(정차 도로까지 보행 400m 초과) / '후보 성김'(도로는 400m 안인데 후보 없음)으로 분류.
   후보 성김 지점에서 직선 500m 안의 정차 가능 도로에 100m 간격 후보 추가(기존 후보 50m 이내 제외)
   동별 최소 커버율: 각 동 95% 이상, 상한이 95% 미만이면 상한의 98% 이상. 기준 p에서 못 맞추면 제약을 넣고 가능한 최소 p로
3. 기존 버스정류소 97곳만 후보로 같은 개수 MCLP + 97곳 전체 커버율
4. 목적지 정류소: 시흥1동 병의원·사회복지시설 중 목적지 정류소에서 보행 400m 안 비율, 70% 미만이면 탐욕적 추가(신규 10개 이상)
반경 300·500m 민감도는 같은 후보·같은 기준(제약 없이)으로 다시 계산
"""
import sys

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from mclp_common import build_candidates, load_demand, p_curve, solve_mclp, stop_name  # noqa: E402
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "outputs"
R0, RADII = 400, [300, 400, 500]
DONGS = ["시흥3동", "시흥4동", "시흥5동"]
ADD_SPACING, ADD_RADIUS, ADD_MIN_SEP = 100, 500, 50
FLOOR, FLOOR_UB = 0.95, 0.98
DEST_MIN_GAIN, DEST_TARGET = 10, 0.70
pd.set_option("display.width", 250)

roads = Roads()
dem = load_demand()
W = dem.가중치_70세이상.to_numpy()
TOT = W.sum()
dxy = dem[["x_5179", "y_5179"]].to_numpy()
dong = dem.동.to_numpy()
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
snap_nodes = {n for s in roads.snap_segs for n in s[:2]}

# ---------------------------------------------------------------- 1. 후보 (경로당·주민센터 이동)
c = build_candidates(roads)
lm = c[c.후보유형.isin(["경로당", "주민센터"])].copy()
lm["50m초과"] = np.where(lm.이동_직선m > 50, "Y", "")
lm[["후보ID", "후보유형", "이름", "동", "이동방식", "이동_직선m", "이동_보행m", "직선최근접과_차이m", "도로명", "50m초과"]].round(1).to_csv(
    f"{OUT}/MCLP_v4_경로당주민센터_정차위치.csv", index=False, encoding="utf-8-sig")
print(f"[1] 경로당·주민센터 {len(lm)}곳 이동: 직선 중앙값 {lm.이동_직선m.median():.0f}m, 50m 초과 {(lm.이동_직선m > 50).sum()}곳")


def road_walk(points):
    """수요점 -> 가장 가까운 정차 가능 도로 노드까지 보행거리"""
    dn, dc = roads.nearest_walk_node(points)
    out = []
    for n0, c0 in zip(dn, dc):
        dd = nx.single_source_dijkstra_path_length(roads.W, n0, cutoff=800, weight="w")
        out.append(c0 + min([v for k, v in dd.items() if k in snap_nodes], default=np.inf))
    return np.array(out)


# ---------------------------------------------------------------- 2. 상한 진단 + 후보 추가
WD0 = roads.walk_dist(c[["x", "y"]].to_numpy(), dxy, max(RADII))
ub0 = (WD0 <= R0).any(axis=1)
out_ub = ~ub0
rw = road_walk(dxy[out_ub])
diag = dem[out_ub].assign(정차도로까지_보행m=rw.round(0), 원인=np.where(rw > R0, "도로가 닿지 않음", "후보 성김"))
sparse = diag[diag.원인 == "후보 성김"]
add = []
for s in roads.snap_segs:
    u, v, L = s[0], s[1], s[2]
    (x1, y1), (x2, y2) = roads.XY[u], roads.XY[v]
    for t in np.arange(0, 1, ADD_SPACING / L if L > ADD_SPACING else 1.0):
        add.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, s[5]))
add = pd.DataFrame(add, columns=["x", "y", "도로명"]).drop_duplicates(["x", "y"])
sxy = sparse[["x_5179", "y_5179"]].to_numpy()
near_sparse = np.array([np.hypot(*(sxy - (x, y)).T).min() <= ADD_RADIUS for x, y in zip(add.x, add.y)]) if len(sxy) else np.zeros(len(add), bool)
add = add[near_sparse]
keep = []
pool = c[["x", "y"]].to_numpy()
for r in add.itertuples():  # 기존 후보·이미 추가한 점과 50m 이상 떨어진 것만
    if np.hypot(*(pool - (r.x, r.y)).T).min() >= ADD_MIN_SEP:
        keep.append(r)
        pool = np.vstack([pool, (r.x, r.y)])
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
addf = pd.DataFrame(keep)
if len(addf):
    pts = gpd.GeoDataFrame(addf, geometry=gpd.points_from_xy(addf.x, addf.y), crs=5179)
    addf["동"] = gpd.sjoin(pts, bnd[["ADM_NM", "geometry"]], how="left").drop_duplicates(subset=None).ADM_NM.values[:len(pts)]  # 클러스터 밖이면 실제 동 이름
    addf = pd.DataFrame({"후보ID": [f"R{i:03d}" for i in range(len(addf))], "후보유형": "추가 도로지점", "이름": "", "동": addf.동,
                         "x": addf.x, "y": addf.y, "도로명": addf.도로명, "이동방식": "성긴 지역 추가(정차 가능 도로 위)"})
    c = pd.concat([c, addf], ignore_index=True)
WD = np.hstack([WD0, roads.walk_dist(addf[["x", "y"]].to_numpy(), dxy, max(RADII))]) if len(addf) else WD0
cxy = c[["x", "y"]].to_numpy()
A = WD <= R0
ub = A.any(axis=1)
diag["후보 추가 후 커버 가능"] = ub[out_ub]
diag.to_csv(f"{OUT}/MCLP_v4_상한밖수요_진단.csv", index=False, encoding="utf-8-sig")
ub_d = {d: W[(dong == d) & ub].sum() / W[dong == d].sum() for d in DONGS}
ub0_d = {d: W[(dong == d) & ub0].sum() / W[dong == d].sum() for d in DONGS}
print(f"\n[2] 상한 밖 수요(후보 추가 전) {W[out_ub].sum():.0f}명: " +
      ", ".join(f"{k} {v:.0f}명" for k, v in diag.groupby("원인").가중치_70세이상.sum().items()) +
      f" | 성긴 지역에 도로 후보 {len(addf)}곳 추가 -> 후보 {len(c)}곳")
for d in DONGS:
    print(f"    {d}: 상한 {ub0_d[d]:.1%} -> {ub_d[d]:.1%}")

# ---------------------------------------------------------------- 400m 기본안 + 동별 제약
res = {}
for r in RADII:
    Ar = WD <= r
    t, sols, pA, pB, p = p_curve(W, Ar, 35 if r == 300 else 25)
    res[r] = dict(A=Ar, t=t, sols=sols, pA=pA, pB=pB, p=p, sel=sols[p])
    print(f"    보행 {r}m: 기준A {pA} / 기준B {pB} -> p = {p} ({t.loc[t.p == p, '커버율'].iloc[0]:.1%})")
floors_rate = {d: (FLOOR if ub_d[d] >= FLOOR else FLOOR_UB * ub_d[d]) for d in DONGS}
floors = {d: floors_rate[d] * W[dong == d].sum() for d in DONGS}


def dong_rates(sel, Ar):
    cov = Ar[:, sel].any(axis=1)
    return {d: W[(dong == d) & cov].sum() / W[dong == d].sum() for d in DONGS}


p0 = res[R0]["p"]
rates0 = dong_rates(res[R0]["sel"], A)
need = any(rates0[d] < floors_rate[d] - 1e-9 for d in DONGS)
print(f"\n    동별 최소 기준: " + ", ".join(f"{d} {floors_rate[d]:.1%}" for d in DONGS))
print(f"    기준 p={p0} 무제약안 동별: " + ", ".join(f"{d} {rates0[d]:.1%}" for d in DONGS) + f" -> 제약 {'필요' if need else '불필요'}")
p_fin, sel_fin = p0, res[R0]["sel"]
if need:
    for p in range(p0, 40):
        st, sel = solve_mclp(W, A, p, groups=dong, floors=floors)
        if st == "Optimal":
            p_fin, sel_fin = p, sel
            break
    print(f"    동별 제약 적용: p {p0} -> {p_fin} (+{p_fin - p0}곳), 동별 " +
          ", ".join(f"{d} {v:.1%}" for d, v in dong_rates(sel_fin, A).items()))
cov_fin = A[:, sel_fin].any(axis=1)
# 형평(동별) 제약 전후 비교: 무제약 p0 해 vs 최종 해
u_ids, f_ids = set(c.후보ID.iloc[res[R0]["sel"]]), set(c.후보ID.iloc[sel_fin])
eq = []
for lab, ids in [("무제약만", u_ids - f_ids), ("제약 후만", f_ids - u_ids), ("공통", u_ids & f_ids)]:
    for i in sorted(ids):
        r = c[c.후보ID == i].iloc[0]
        j = c.index[c.후보ID == i][0]
        eq.append({"구분": lab, "후보ID": i, "이름": stop_name(r, bus), "동": r.동, "후보유형": r.후보유형,
                   "단독커버_명": round(W[A[:, j]].sum()), **{f"{d}_단독커버": round(W[A[:, j] & (dong == d)].sum()) for d in DONGS}})
pd.DataFrame(eq).to_csv(f"{OUT}/MCLP_v4_형평제약_전후비교.csv", index=False, encoding="utf-8-sig")
print("\n[형평 제약 전후]", pd.DataFrame(eq).query("구분 != '공통'").to_string(index=False))
print("무제약 12곳 동별:", {d: f"{v:.1%}" for d, v in rates0.items()})
# 근거표: 제약 전후 동별 커버율·커버 인원·정류소 수
evid = []
for lab, p_, sel_ in [("무제약(기준 p)", p0, res[R0]["sel"]), ("동별 제약 후(최종)", p_fin, sel_fin)]:
    cov_ = A[:, sel_].any(axis=1)
    row = {"안": lab, "정류소_수": p_, "후보_수": len(c), "수요점_수": len(W), "수요_70세이상_명": round(W.sum()), "커버_명": round(W[cov_].sum()),
           "커버율_%": round(W[cov_].sum() / W.sum() * 100, 1)}
    for d in DONGS:
        row[f"{d}_정류소"] = int((c.동.iloc[sel_] == d).sum())
        row[f"{d}_커버율_%"] = round(W[(dong == d) & cov_].sum() / W[dong == d].sum() * 100, 1)
        row[f"{d}_기준_%"] = round(floors_rate[d] * 100, 1)
    row["기타동_정류소"] = int((~c.동.iloc[sel_].isin(DONGS)).sum())
    evid.append(row)
pd.DataFrame(evid).to_csv(f"{OUT}/MCLP_v4_근거표_제약전후.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(evid).T.to_string())
curve = pd.concat([v["t"].assign(반경_m=r) for r, v in res.items()])
curve.assign(커버율_pct=(curve.커버율 * 100).round(1), **{"추가커버_%p": curve["추가커버_%p"].round(1)}).round({"커버_명": 0, "추가커버_명": 0})[
    ["반경_m", "p", "커버_명", "커버율_pct", "추가커버_명", "추가커버_%p", "상태"]].to_csv(f"{OUT}/MCLP_v4_보행_p곡선.csv", index=False, encoding="utf-8-sig")
print("\n보행 400m p=1~20:")
t4 = res[R0]["t"]
print(pd.DataFrame({"p": t4.p, "커버_명": t4.커버_명.round(0), "커버율_%": (t4.커버율 * 100).round(1), "추가커버_명": t4.추가커버_명.round(0),
                    "추가커버_%p": t4["추가커버_%p"].round(1)})[t4.p <= 20].to_string(index=False))

# ---------------------------------------------------------------- 반경 민감도 (최종 400m안 기준 겹침)
fin_ids = set(c.후보ID.iloc[sel_fin])
rows = []
for r in RADII:
    sel = sel_fin if r == R0 else res[r]["sel"]
    Ar = res[r]["A"]
    cov = Ar[:, sel].any(axis=1)
    row = {"반경_m": r, "정류소수": len(sel), "기준A": res[r]["pA"], "기준B": res[r]["pB"],
           "비고": f"동별 제약 +{p_fin - p0}" if (r == R0 and p_fin > p0) else "", "커버율_%": round(W[cov].sum() / TOT * 100, 1),
           "보행400m로 잰 커버율_%": round(W[(WD[:, sel] <= R0).any(axis=1)].sum() / TOT * 100, 1),
           "후보전체 상한_%": round(W[Ar.any(axis=1)].sum() / TOT * 100, 1)}
    for d in DONGS:
        row[f"{d}_정류소"] = int((c.동.iloc[sel] == d).sum())
    row["기타동_정류소"] = int((~c.동.iloc[sel].isin(DONGS)).sum())
    for d, v in dong_rates(sel, Ar).items():
        row[f"{d}_커버율_%"] = round(v * 100, 1)
    for d in DONGS:
        row[f"{d}_미커버_명"] = round(W[(dong == d) & ~cov].sum())
    row["400m최종안과 겹침"] = len(set(c.후보ID.iloc[sel]) & fin_ids)
    rows.append(row)
sens = pd.DataFrame(rows)
sens.to_csv(f"{OUT}/MCLP_v4_반경민감도.csv", index=False, encoding="utf-8-sig")
print("\n반경 민감도:")
print(sens.T.to_string())

# ---------------------------------------------------------------- 3. 기존 버스정류소만
bi = np.flatnonzero((c.후보유형 == "기존 버스정류소").to_numpy())
st, sb = solve_mclp(W, A[:, bi], p_fin)
sel_bus = list(bi[sb])
cov_bus = A[:, sel_bus].any(axis=1)
cov_all97 = A[:, bi].any(axis=1)
cmp_rows = []
for lab, sel, cov in [(f"최종안 ({p_fin}곳)", sel_fin, cov_fin), (f"기존 버스정류소만 ({p_fin}곳)", sel_bus, cov_bus),
                      (f"참고: 기존 버스정류소 {len(bi)}곳 전부", list(bi), cov_all97)]:
    row = {"안": lab, "정류소수": len(sel), "보행400m_커버율_%": round(W[cov].sum() / TOT * 100, 1), "미커버_명": round(W[~cov].sum())}
    for d in DONGS:
        row[f"{d}_커버율_%"] = round(W[(dong == d) & cov].sum() / W[dong == d].sum() * 100, 1)
    for d in DONGS:
        row[f"{d}_정류소"] = int((c.동.iloc[sel] == d).sum())
    cmp_rows.append(row)
cmp_bus = pd.DataFrame(cmp_rows)
cmp_bus.to_csv(f"{OUT}/MCLP_v4_최종안_vs_기존정류소.csv", index=False, encoding="utf-8-sig")
print("\n[3] 기존 정류소 비교:")
print(cmp_bus.to_string(index=False))

# ---------------------------------------------------------------- 4. 목적지 정류소 보행 400m
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig")
mu = fac[(fac.시설명 == "무지개아파트") & (fac.행정동 == "시흥1동")].iloc[0]
dst = pd.concat([fixed[["이름", "x", "y"]], pd.DataFrame([{"이름": "무지개아파트", "x": mu.x_5179, "y": mu.y_5179}])], ignore_index=True)
d1 = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.행정동 == "시흥1동")]
d1xy = d1[["x_5179", "y_5179"]].to_numpy()
DW = roads.walk_dist(dst[["x", "y"]].to_numpy(), d1xy, R0)
covd = np.isfinite(DW).any(axis=1)
rows = [{"단계": "현재 4곳", "정류소": "", "신규": "", "누적_%": round(covd.mean() * 100, 1)}]
per = {n: int(np.isfinite(DW[:, k]).sum()) for k, n in enumerate(dst.이름)}
s1b = bus[bus.행정동 == "시흥1동"]
BW = roads.walk_dist(s1b[["x_5179", "y_5179"]].to_numpy(), d1xy, R0)
added_dest = []
while covd.mean() < DEST_TARGET:
    gain = (np.isfinite(BW) & ~covd[:, None]).sum(axis=0)
    k = int(np.argmax(gain))
    if gain[k] < DEST_MIN_GAIN:
        break
    covd |= np.isfinite(BW[:, k])
    added_dest.append(s1b.iloc[k].시설명)
    rows.append({"단계": f"+{len(added_dest)}", "정류소": s1b.iloc[k].시설명, "신규": int(gain[k]), "누적_%": round(covd.mean() * 100, 1)})
dtab = pd.DataFrame(rows)
dtab.to_csv(f"{OUT}/MCLP_v4_목적지정류소_보행400.csv", index=False, encoding="utf-8-sig")
print(f"\n[4] 시흥1동 병의원·사회복지 {len(d1)}개 중 목적지 정류소 보행 400m 안 (정류소별 {per}):")
print(dtab.to_string(index=False))

# ---------------------------------------------------------------- 5·6. 최종 목록
s = c.iloc[sel_fin].copy()
s["표시이름"] = [stop_name(r, bus) for r in s.itertuples()]
s["표시이름"] = s.표시이름.replace({"석수역": "석수역 버스정류소(승차)"})
WS = np.where(WD[:, sel_fin] <= R0, WD[:, sel_fin], np.inf)
assign = np.where(np.isfinite(WS).any(axis=1), WS.argmin(axis=1), -1)
s["배정_70세이상_명"] = [round(W[assign == k].sum()) for k in range(len(sel_fin))]
s["배정_주민_동구성"] = [", ".join(f"{d} {W[(assign == k) & (dong == d)].sum():.0f}" for d in DONGS if W[(assign == k) & (dong == d)].sum() >= 1)
                   for k in range(len(sel_fin))]
TYPE = {"기존 버스정류소": "기존 버스정류소", "격자점": "신규 지점", "추가 도로지점": "신규 지점", "경로당": "경로당·주민센터", "주민센터": "경로당·주민센터"}
s["유형"] = s.후보유형.map(TYPE)
v3 = pd.read_csv(f"{OUT}/MCLP_v3_최종승차정류소.csv", encoding="utf-8-sig")
s["v3와"] = np.where(s.후보ID.isin(set(v3.후보ID)), "유지", "신규")
ll = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), crs=5179, index=s.index).to_crs(4326)
s["위도"], s["경도"] = ll.y.round(6), ll.x.round(6)
s = s.sort_values(["동", "배정_70세이상_명"], ascending=[True, False]).reset_index(drop=True)
s.insert(0, "ID", [f"B{i + 1:02d}" for i in range(len(s))])
fin = s[["ID", "표시이름", "동", "유형", "배정_70세이상_명", "배정_주민_동구성", "v3와", "이동방식", "후보ID", "위도", "경도", "x", "y"]].rename(
    columns={"표시이름": "정류소명", "x": "x_5179", "y": "y_5179"})
fin.to_csv(f"{OUT}/MCLP_v4_최종승차정류소.csv", index=False, encoding="utf-8-sig")
print(f"\n[5] 최종 승차 정류소 {len(fin)}곳 (보행 400m 커버율 {W[cov_fin].sum() / TOT:.1%}, 배정 합계 {fin.배정_70세이상_명.sum():,}명)")
print(fin[["ID", "정류소명", "동", "유형", "배정_70세이상_명", "배정_주민_동구성", "v3와"]].to_string(index=False))
print("※ 동은 정류소 위치 기준이며, 다른 동 주민이 가장 가까운 정류소로 배정될 수 있음")
print("v3 대비 빠진 곳:", v3[~v3.후보ID.isin(set(s.후보ID))].정류소명.tolist())

# ---------------------------------------------------------------- 지도
SURF, INK, MUTED, SEL, OLD, COV, ADD = "#fcfcfb", "#0b0b0b", "#898781", "#2a78d6", "#eb6834", "#86b6ef", "#1baf7a"
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


def demand(ax, cov, scale=1.5):
    ax.scatter(dxy[~cov, 0], dxy[~cov, 1], s=W[~cov] * scale, color=MUTED, alpha=0.7, lw=0, zorder=3)
    ax.scatter(dxy[cov, 0], dxy[cov, 1], s=W[cov] * scale, color=COV, alpha=0.9, lw=0, zorder=3)


# (a) 최종안
fig, ax = plt.subplots(figsize=(9, 10), dpi=160, facecolor=SURF)
base(ax, f"보행 {R0}m 승차 정류소 {len(fin)}곳 (v4) — 커버율 {W[cov_fin].sum() / TOT:.1%}")
demand(ax, cov_fin)
ax.scatter(v3.x_5179, v3.y_5179, s=70, facecolor="none", edgecolor=OLD, lw=1.6, zorder=5)
ax.scatter(fin.x_5179, fin.y_5179, s=70, color=SEL, edgecolor=SURF, lw=1.5, zorder=6)
for r in fin.itertuples():
    ax.annotate(r.ID, (r.x_5179, r.y_5179), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=9, label=f"v4 승차 정류소 ({len(fin)}곳)"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=OLD, ms=9, label=f"v3 정류소 ({len(v3)}곳)"),
                   Line2D([], [], marker="o", ls="", color=COV, ms=7, label=f"수요점: 보행 {R0}m 커버 (크기=70세 이상)"),
                   Line2D([], [], marker="o", ls="", color=MUTED, ms=7, label="수요점: 미커버")], frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "보행 도로거리: OSM 보행망(경사 미반영). 번호는 MCLP_v4_최종승차정류소.csv. © OpenStreetMap contributors", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/MCLP_v4_거점지도_보행400.png", facecolor=SURF)
plt.close(fig)

# (b) 상한 진단
fig, ax = plt.subplots(figsize=(9, 10), dpi=160, facecolor=SURF)
base(ax, "후보 전부 선택해도 보행 400m 밖인 수요 (후보 추가 전)과 추가 후보")
ax.scatter(dxy[:, 0], dxy[:, 1], s=W * 1.0, color="#e6e5e0", lw=0, zorder=2)
ax.scatter(cxy[c.후보유형 != "추가 도로지점", 0], cxy[c.후보유형 != "추가 도로지점", 1], s=8, color="#c3c2b7", zorder=3)
for cause, col in [("도로가 닿지 않음", OLD), ("후보 성김", SEL)]:
    g = diag[diag.원인 == cause]
    ax.scatter(g.x_5179, g.y_5179, s=g.가중치_70세이상 * 2 + 20, color=col, edgecolor=SURF, lw=1, zorder=5,
               label=f"{cause} ({g.가중치_70세이상.sum():.0f}명)")
if len(addf):
    ax.scatter(addf.x, addf.y, s=22, marker="^", color=ADD, zorder=6, label=f"추가 후보 (정차 가능 도로, {len(addf)}곳)")
ax.scatter([], [], s=8, color="#c3c2b7", label="기존 후보")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/MCLP_v4_상한진단지도.png", facecolor=SURF)
plt.close(fig)

# (c) 반경 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), dpi=140, facecolor=SURF)
for ax, r in zip(axes, RADII):
    sel = sel_fin if r == R0 else res[r]["sel"]
    cov = res[r]["A"][:, sel].any(axis=1)
    base(ax, f"보행 {r}m: {len(sel)}곳, 커버율 {W[cov].sum() / TOT:.1%}")
    demand(ax, cov, 1.0)
    inn = c.후보ID.iloc[sel].isin(fin_ids).to_numpy()
    xy = cxy[sel]
    ax.scatter(xy[inn, 0], xy[inn, 1], s=55, color=SEL, edgecolor=SURF, lw=1.2, zorder=6)
    ax.scatter(xy[~inn, 0], xy[~inn, 1], s=55, color=OLD, edgecolor=SURF, lw=1.2, zorder=6)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=8, label="400m 최종안에도 있는 정류소"),
                        Line2D([], [], marker="o", ls="", color=OLD, ms=8, label="이 반경에만 있는 정류소"),
                        Line2D([], [], marker="o", ls="", color=COV, ms=6, label="커버된 수요점"),
                        Line2D([], [], marker="o", ls="", color=MUTED, ms=6, label="미커버 수요점")], frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/MCLP_v4_반경비교지도.png", facecolor=SURF)
plt.close(fig)
print("\n저장: MCLP_v4_경로당주민센터_정차위치.csv, MCLP_v4_상한밖수요_진단.csv, MCLP_v4_보행_p곡선.csv, MCLP_v4_반경민감도.csv, "
      "MCLP_v4_최종안_vs_기존정류소.csv, MCLP_v4_목적지정류소_보행400.csv, MCLP_v4_최종승차정류소.csv, 지도 3종")
