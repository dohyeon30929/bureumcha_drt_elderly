"""승차 정류소 수를 보행 도로거리 400m 기준으로 재결정 (직선 300m 11곳안과 비교).

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP_보행.py`  (선행: 금천구_MCLP_최종.py)
- 후보: 금천구_MCLP_최종.py 와 같은 집합(격자점은 정차 가능 도로 위로 스냅, 버스정류소 100m 이내 격자점 제거)
- 커버: 수요점-후보 보행 도로거리 <= 400m (OSM 보행망, 양 끝은 최근접 보행 노드에 직선 연결)
- p = 1~20 MCLP. 기준 A: 커버율 95% 최초 p / 기준 B: 다음 추가 커버 2%p 미만 직전 p (= 추가 >= 2%p 마지막 p)
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

OUT = "outputs"
WALK_R, EUC_R, NEAR_BUS = 400, 300, 100
P_RANGE = range(1, 21)
SHOW_FROM = 11
TARGET, MIN_GAIN = 0.95, 0.02

dem = pd.read_csv(f"{OUT}/MCLP_수요점_시흥345_70세.csv", encoding="utf-8-sig")
cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
raw_bus = pd.read_csv("data/interim/금천구_목적지후보_상세.csv", encoding="utf-8-sig")
raw_bus = raw_bus[(raw_bus.유형 == "버스정류소(연계)") & raw_bus.시흥클러스터.notna()]
fin11 = pd.read_csv(f"{OUT}/MCLP_최종_p11_승차정류소.csv", encoding="utf-8-sig")
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
WD = roads.walk_dist(cxy, dxy, WALK_R)
A = np.isfinite(WD)
ub = W[A.any(axis=1)].sum() / TOT
print(f"후보 {len(c)}개, 수요 {len(W)}점 {TOT:,.0f}명. 후보 전부 선택 시 보행 {WALK_R}m 커버 상한 {ub:.1%} "
      f"(어떤 후보에서도 {WALK_R}m 안에 못 닿는 수요 {W[~A.any(axis=1)].sum():,.0f}명)")


def mclp(p):
    m = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(len(c)), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
    m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
    for i in range(len(W)):
        js = np.flatnonzero(A[i])
        m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
    m += pulp.lpSum(x.values()) == p
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    return pulp.LpStatus[m.status], [j for j in range(len(c)) if x[j].value() > 0.5]


rows, sols = [], {}
for p in P_RANGE:
    st, sel = mclp(p)
    sols[p] = sel
    rows.append({"p": p, "커버_명": W[A[:, sel].any(axis=1)].sum(), "상태": st})
t = pd.DataFrame(rows)
t["커버율"] = t.커버_명 / TOT
t["추가커버_명"] = t.커버_명.diff().fillna(t.커버_명)
t["추가커버_%p"] = t.커버율.diff().fillna(t.커버율) * 100
t["다음p_추가_%p"] = t["추가커버_%p"].shift(-1)
reachA = t[t.커버율 >= TARGET]
pA = int(reachA.p.min()) if len(reachA) else None
pB = int(t.loc[t["다음p_추가_%p"] < MIN_GAIN * 100, "p"].min())
p_sel = max(x for x in [pA, pB] if x is not None)
out = pd.DataFrame({"p": t.p, "보행400m_커버_명": t.커버_명.round(0), "커버율_%": (t.커버율 * 100).round(1),
                    "추가커버_명": t.추가커버_명.round(0), "추가커버_%p": t["추가커버_%p"].round(1), "상태": t.상태})
out.to_csv(f"{OUT}/MCLP_보행400_p1-20.csv", index=False, encoding="utf-8-sig")
print(out[out.p >= SHOW_FROM - 1].to_string(index=False))
print(f"\n기준 A(커버율 {TARGET:.0%} 최초): {pA if pA else f'p<=20 에서 미달성(상한 {ub:.1%})'} | "
      f"기준 B(다음 추가 {MIN_GAIN:.0%}p 미만 직전): {pB} -> 채택 p = {p_sel}")

# ---- 직선 300m 11곳안 vs 보행 400m 안
def metrics(xy, label):
    e = (np.hypot(dxy[:, None, 0] - xy[None, :, 0], dxy[:, None, 1] - xy[None, :, 1]) <= EUC_R).any(axis=1)
    w = np.isfinite(roads.walk_dist(xy, dxy, WALK_R)).any(axis=1)
    r = {"안": label, "정류소수": len(xy), "직선300m_커버율_%": round(W[e].sum() / TOT * 100, 1),
         "보행400m_커버율_%": round(W[w].sum() / TOT * 100, 1)}
    for d in ["시흥3동", "시흥4동", "시흥5동"]:
        r[f"{d}_보행400m_미커버_명"] = round(W[~w & (dem.동 == d).to_numpy()].sum())
    r["보행400m_미커버_합계_명"] = round(W[~w].sum())
    return r


wsel = sols[p_sel]
cmp_ = pd.DataFrame([metrics(fin11[["x_5179", "y_5179"]].to_numpy(), "직선 300m 기준 (p=11, 최종)"),
                     metrics(cxy[sols[11]], "보행 400m 기준 p=11 (참고: 같은 개수)"),
                     metrics(cxy[wsel], f"보행 400m 기준 (p={p_sel})")])
cmp_.to_csv(f"{OUT}/MCLP_직선300_vs_보행400_비교.csv", index=False, encoding="utf-8-sig")
print("\n비교:")
print(cmp_.to_string(index=False))

# ---- 보행 400m 채택안 정류소
fb = bus[["x_5179", "y_5179"]].to_numpy()


def nm(r):
    if r.후보유형 != "격자점":
        return r.이름
    k = int(np.argmin(np.hypot(*(fb - (r.x, r.y)).T)))
    d = np.hypot(*(fb[k] - (r.x, r.y)))
    parts = [r.도로명 if isinstance(r.도로명, str) else None, f"{bus.iloc[k].시설명} 정류소 {d:.0f}m" if d <= 250 else None]
    return " · ".join(p for p in parts if p) or f"{r.동} 무명도로"


s = c.loc[wsel].copy()
s["표시이름"] = [nm(r) for r in s.itertuples()]
Ws = WD[:, wsel]
cov = np.isfinite(Ws).any(axis=1)
assign = np.where(cov, np.where(np.isfinite(Ws), Ws, 1e18).argmin(axis=1), -1)
s["배정_70세이상_명(보행)"] = [round(W[assign == k].sum()) for k in range(len(wsel))]
s["보행400m_단독커버_명"] = [round(W[np.isfinite(Ws[:, k])].sum()) for k in range(len(wsel))]
s["직선300m안에도_있음"] = np.where(s.후보ID.isin(set(fin11.후보ID)), "Y", "")
ll = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), crs=5179, index=s.index).to_crs(4326)
s["위도"], s["경도"] = ll.y.round(6), ll.x.round(6)
s = s.sort_values(["동", "배정_70세이상_명(보행)"], ascending=[True, False])
s[["후보ID", "후보유형", "표시이름", "이름", "동", "위도", "경도", "x", "y", "배정_70세이상_명(보행)", "보행400m_단독커버_명", "직선300m안에도_있음"]].rename(
    columns={"x": "x_5179", "y": "y_5179", "이름": "원이름"}).to_csv(f"{OUT}/MCLP_보행400_p{p_sel}_승차정류소.csv", index=False, encoding="utf-8-sig")
print(f"\n보행 400m 기준 p={p_sel} 정류소 (직선 300m 11곳안과 겹침 {(s['직선300m안에도_있음'] == 'Y').sum()}곳):")
print(s[["후보유형", "표시이름", "동", "배정_70세이상_명(보행)", "보행400m_단독커버_명", "직선300m안에도_있음"]].to_string(index=False))
print(f"\n저장: {OUT}/MCLP_보행400_p1-20.csv, MCLP_직선300_vs_보행400_비교.csv, MCLP_보행400_p{p_sel}_승차정류소.csv")
