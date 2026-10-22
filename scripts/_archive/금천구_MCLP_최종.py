"""1단계 MCLP 마무리: 후보 격자점을 정차 가능 도로 위로 먼저 스냅한 뒤 반경 300m·p=11 재최적화.

실행: 프로젝트 루트에서 `python scripts/금천구_MCLP_최종.py`  (선행: 금천구_MCLP.py, 금천구_MCLP_p결정.py, 금천구_DRT_도로망.py)
- 후보: 기존 버스정류소·경로당·주민센터는 위치 그대로, 격자점은 osm_roads.Roads.snap 위치로 이동.
  스냅 후 기존 버스정류소 100m 이내로 들어온 격자점은 이전과 같은 규칙으로 중복 제거
- 커버 판정: 직선 300m(최적화 기준) + 보행 도로거리 400m(OSM 보행망, 수요점·정류소는 최근접 보행 노드에 직선 연결)
- 배정 고령인구: 커버된 수요점을 가장 가까운(직선) 선정 거점에 배정 -> 3단계 호출 확률 가중치
"""
import sys

import numpy as np
import pandas as pd
import pulp
import geopandas as gpd

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

OUT = "outputs"
R, P, WALK_R, NEAR_BUS = 300, 11, 400, 100

dem = pd.read_csv(f"{OUT}/MCLP_수요점_시흥345_70세.csv", encoding="utf-8-sig")
cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
raw_bus = pd.read_csv("data/interim/금천구_목적지후보_상세.csv", encoding="utf-8-sig")
raw_bus = raw_bus[(raw_bus.유형 == "버스정류소(연계)") & raw_bus.시흥클러스터.notna()]
prev = pd.read_csv(f"{OUT}/MCLP_r300_p11_선정거점.csv", encoding="utf-8-sig")
prev_snap = pd.read_csv(f"{OUT}/DRT_정류소14_스냅.csv", encoding="utf-8-sig")
roads = Roads()

# ---- 후보 격자점 스냅
c = cand.copy()
c["원x"], c["원y"], c["스냅거리_m"], c["도로명"] = c.x, c.y, 0.0, None
g = c.후보유형 == "격자점"
for i in c.index[g]:
    s = roads.snap(c.at[i, "x"], c.at[i, "y"])
    c.loc[i, ["x", "y", "스냅거리_m", "도로명"]] = [s["sx"], s["sy"], s["스냅거리_m"], s["도로명"]]
bxy = raw_bus[["x_5179", "y_5179"]].to_numpy()
near = np.array([np.hypot(*(bxy - (x, y)).T).min() <= NEAR_BUS for x, y in zip(c.x, c.y)]) & g
print(f"격자점 {g.sum()}개 스냅: 이동거리 중앙값 {c.loc[g, '스냅거리_m'].median():.0f}m, 최대 {c.loc[g, '스냅거리_m'].max():.0f}m, "
      f"50m 초과 {(c.loc[g, '스냅거리_m'] > 50).sum()}개 | 스냅 후 버스정류소 {NEAR_BUS}m 이내라 제거 {near.sum()}개")
c = c[~near].reset_index(drop=True)

# ---- MCLP
W = dem.가중치_70세이상.to_numpy()
TOT = W.sum()
dxy = dem[["x_5179", "y_5179"]].to_numpy()
A = np.hypot(dxy[:, None, 0] - c.x.values[None, :], dxy[:, None, 1] - c.y.values[None, :]) <= R
m = pulp.LpProblem("MCLP", pulp.LpMaximize)
x = pulp.LpVariable.dicts("x", range(len(c)), cat="Binary")
y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
for i in range(len(W)):
    js = np.flatnonzero(A[i])
    m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
m += pulp.lpSum(x.values()) == P
m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
sel = [j for j in range(len(c)) if x[j].value() > 0.5]
s = c.loc[sel].copy()
cov = A[:, sel].any(axis=1)
print(f"MCLP p={P} r={R}m: {pulp.LpStatus[m.status]}, 커버율 {W[cov].sum() / TOT:.1%} ({W[cov].sum():,.0f}/{TOT:,.0f}명)")

# ---- 이름: 기존 정류소·경로당은 원래 이름, 격자점은 도로명 + 최근접 버스정류소
fb = bus[["x_5179", "y_5179"]].to_numpy()


def nm(r):
    if r.후보유형 != "격자점":
        return r.이름
    k = int(np.argmin(np.hypot(*(fb - (r.x, r.y)).T)))
    d = np.hypot(*(fb[k] - (r.x, r.y)))
    parts = [r.도로명 if isinstance(r.도로명, str) else None, f"{bus.iloc[k].시설명} 정류소 {d:.0f}m" if d <= 250 else None]
    return " · ".join(p for p in parts if p) or f"{r.동} 무명도로"


s["표시이름"] = [nm(r) for r in s.itertuples()]

# ---- 보행 400m 커버 (최종) + 이전 두 안 비교
def walk_cov(xy):
    return roads.walk_within(xy, dxy, WALK_R).any(axis=1)


def euc_cov(xy, r=R):
    return (np.hypot(dxy[:, None, 0] - xy[None, :, 0], dxy[:, None, 1] - xy[None, :, 1]) <= r).any(axis=1)


prev_xy0 = prev.merge(cand[["후보ID", "x", "y"]], on="후보ID")[["x", "y"]].to_numpy()
pb = prev_snap[prev_snap.구분 == "승차"]
prev_xy1 = np.where((pb.후보유형 == "격자점").values[:, None], pb[["x_5179_스냅", "y_5179_스냅"]].values, pb[["x_5179_원", "y_5179_원"]].values)
fin_xy = s[["x", "y"]].to_numpy()
cmp_rows = []
for label, xy in [("이전: 스냅 전(격자 원위치)", prev_xy0), ("이전: 선정 후 스냅", prev_xy1), ("최종: 스냅 후 재최적화", fin_xy)]:
    e, w = euc_cov(xy), walk_cov(xy)
    cmp_rows.append({"안": label, "직선300m_커버율_%": round(W[e].sum() / TOT * 100, 1), "보행400m_커버율_%": round(W[w].sum() / TOT * 100, 1),
                     "직선300m_명": round(W[e].sum()), "보행400m_명": round(W[w].sum())})
cmp_ = pd.DataFrame(cmp_rows)
cmp_.to_csv(f"{OUT}/MCLP_최종_커버율비교.csv", index=False, encoding="utf-8-sig")
print("\n커버율 비교:")
print(cmp_.to_string(index=False))
wf = walk_cov(fin_xy)
print(f"최종안: 직선300m 커버 & 보행400m 미커버 {W[cov & ~wf].sum():,.0f}명 / 직선 미커버 & 보행 커버 {W[~cov & wf].sum():,.0f}명")
print("보행400m 미커버 동별:", dem[~wf].groupby("동").가중치_70세이상.sum().round(0).to_dict())

# ---- 배정 고령인구 + 선정 거점 비교
Dsel = np.hypot(dxy[:, None, 0] - fin_xy[None, :, 0], dxy[:, None, 1] - fin_xy[None, :, 1])
assign = np.where(cov, Dsel.argmin(axis=1), -1)
s["배정_70세이상_명"] = [round(W[assign == k].sum()) for k in range(len(s))]
s["단독커버_명"] = [round(W[A[:, j]].sum()) for j in sel]
s["보행400m_단독커버_명"] = [round(W[roads.walk_within(fin_xy[k:k + 1], dxy, WALK_R)[:, 0]].sum()) for k in range(len(s))]
ll = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), crs=5179, index=s.index).to_crs(4326)
s["위도"], s["경도"] = ll.y.round(6), ll.x.round(6)
prev_ids = set(prev.후보ID)
s["이전p11과"] = np.where(s.후보ID.isin(prev_ids), "유지", "신규")
s = s.sort_values(["동", "배정_70세이상_명"], ascending=[True, False])
cols = ["후보ID", "후보유형", "표시이름", "이름", "동", "위도", "경도", "x", "y", "스냅거리_m", "배정_70세이상_명", "단독커버_명",
        "보행400m_단독커버_명", "이전p11과"]
s[cols].rename(columns={"x": "x_5179", "y": "y_5179", "이름": "원이름"}).round({"스냅거리_m": 1}).to_csv(
    f"{OUT}/MCLP_최종_p11_승차정류소.csv", index=False, encoding="utf-8-sig")
dropped = prev[~prev.후보ID.isin(set(s.후보ID))]
print(f"\n최종 승차 정류소 11곳 (이전 p=11 대비 유지 {(s.이전p11과 == '유지').sum()}, 신규 {(s.이전p11과 == '신규').sum()}; "
      f"빠진 곳: {dropped.이름.tolist()})")
print(s[["후보유형", "표시이름", "동", "스냅거리_m", "배정_70세이상_명", "단독커버_명", "보행400m_단독커버_명", "이전p11과"]].round(1).to_string(index=False))
print(f"\n저장: {OUT}/MCLP_최종_p11_승차정류소.csv, {OUT}/MCLP_최종_커버율비교.csv")
