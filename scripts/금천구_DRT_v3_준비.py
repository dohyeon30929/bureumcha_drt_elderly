"""DRT v3 준비: MCLP v4 승차 13곳 + 목적지 5곳 + 권역 A 대기 장소, 비대칭 도로거리 행렬, 고정노선 순환, 목적지 추첨 비중.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_v3_준비.py` -> data/interim/DRT_v3_입력.pkl (+ outputs/DRT_v3_정류소.csv 등)
- 차량 도로망: data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json (관악구 난향동까지 확장, B01 경로 계산용)
- 승차 13곳: outputs/MCLP_v4_최종승차정류소.csv. 권역 A = 시흥3동 5곳, B = 시흥4·5동 7곳 + B01(관악구 난향동, 시흥4동 주민 배정)
- 목적지 5곳: 금천구청역, 시흥사거리, 현대시장.시흥4동주민센터, 무지개아파트(이상 시흥1동) + 석수역 환승(석수역 버스정류소 하차, 1번 출입구 도보 안내)
- 대기 장소: 단일안·권역 B = 시흥사거리, 권역 A = 시흥3동 안 정차 가능 도로 노드 중 A 승차 5곳 왕복 도로거리 합 최소 지점
- 목적지 추첨 비중(승차 정류소별):
    석수역 환승 = O-D(2026-08-31, 60세 이상) 출발 동별 '안양시행 ÷ (시흥1동행 + 안양시행)' (B01은 시흥4동 값)
    나머지 = 시흥1동 4곳에 '정류소 보행 400m 안 병의원·사회복지시설 수' 비례
    승차 정류소에서 보행 400m 안에 있는 목적지는 그 정류소의 후보에서 제외 후 재정규화
- 고정노선: 승차 13 + 목적지 5를 한 번씩 도는 최단 비대칭 순회(MTZ, 시흥사거리 출발)
"""
import pickle
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

OUT, INTERIM = "outputs", "data/interim"
ROAD = "data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json"
SPEED_MPM = 20000 / 60  # m/분
DWELL = 1.0
ZONE = {"시흥3동": "A", "시흥4동": "B", "시흥5동": "B", "난향동": "B"}
RES_DONG = {"난향동": "시흥4동"}  # B01 이용자는 시흥4동 주민
SIHEUNG3 = "11180590"
DESTS = ["금천구청역", "시흥사거리", "현대시장.시흥4동주민센터", "무지개아파트", "석수역 환승"]

roads = Roads(path=ROAD)
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
b = pd.read_csv(f"{OUT}/MCLP_v4_최종승차정류소.csv", encoding="utf-8-sig")
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig").set_index("이름")
mu = fac[(fac.시설명 == "무지개아파트") & (fac.행정동 == "시흥1동")].iloc[0]
ss = fac[(fac.시설명 == "석수역") & (fac.유형 == "버스정류소(연계)") & (fac.행정동 == "시흥3동")].iloc[0]
dxy = {"금천구청역": (fixed.loc["금천구청역", "x"], fixed.loc["금천구청역", "y"]), "시흥사거리": (fixed.loc["시흥사거리", "x"], fixed.loc["시흥사거리", "y"]),
       "현대시장.시흥4동주민센터": (fixed.loc["현대시장.시흥4동주민센터", "x"], fixed.loc["현대시장.시흥4동주민센터", "y"]),
       "무지개아파트": (mu.x_5179, mu.y_5179), "석수역 환승": (ss.x_5179, ss.y_5179)}
stops = pd.concat([
    pd.DataFrame({"ID": b.ID, "구분": "승차", "이름": b.정류소명, "동": b.동, "권역": b.동.map(ZONE), "유형": b.유형,
                  "배정_70세이상_명": b.배정_70세이상_명, "x": b.x_5179, "y": b.y_5179}),
    pd.DataFrame({"ID": [f"D{i + 1}" for i in range(5)], "구분": "목적지", "이름": DESTS, "동": ["시흥1동"] * 4 + ["시흥3동(안양 석수역 도보 환승)"],
                  "권역": "", "유형": ["지하철역", "기존 버스정류소", "기존 버스정류소", "기존 버스정류소", "기존 버스정류소(석수역 버스정류소)"],
                  "배정_70세이상_명": np.nan, "x": [dxy[d][0] for d in DESTS], "y": [dxy[d][1] for d in DESTS]})], ignore_index=True)
snaps = [roads.snap(x, y) for x, y in zip(stops.x, stops.y)]
stops["스냅거리_m"] = [round(s["스냅거리_m"], 1) for s in snaps]
print("스냅 50m 초과:", stops.loc[stops.스냅거리_m > 50, ["ID", "이름", "스냅거리_m"]].to_dict("records"))

# ---- 권역 A 대기 장소
A_idx = list(stops.index[stops.권역 == "A"])
snap_nodes = {n for s in roads.snap_segs for n in s[:2]}
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
s3 = bnd[bnd.ADM_CD == SIHEUNG3].geometry.iloc[0]
cand = [n for n in snap_nodes if s3.contains(gpd.points_from_xy([roads.XY[n][0]], [roads.XY[n][1]])[0])]
GR = roads.G.reverse(copy=False)


def ends(s):
    u, v, L, two, *_ = roads.snap_segs[s["seg"]]
    ex, en = [(v, (1 - s["t"]) * L)], [(u, s["t"] * L)]
    if two:
        ex.append((u, s["t"] * L))
        en.append((v, (1 - s["t"]) * L))
    return ex, en


cost = dict.fromkeys(cand, 0.0)
for i in A_idx:
    ex, en = ends(snaps[i])
    to_s, fr_s = {}, {}
    for node, c in en:
        for n_, d in nx.single_source_dijkstra_path_length(GR, node, weight="w").items():
            to_s[n_] = min(to_s.get(n_, np.inf), d + c)
    for node, c in ex:
        for n_, d in nx.single_source_dijkstra_path_length(roads.G, node, weight="w").items():
            fr_s[n_] = min(fr_s.get(n_, np.inf), d + c)
    for n in cand:
        cost[n] += to_s.get(n, np.inf) + fr_s.get(n, np.inf)
best = min(cost, key=cost.get)
dep = roads.snap(*roads.XY[best])
bus = fac[fac.유형 == "버스정류소(연계)"]
bxy = bus[["x_5179", "y_5179"]].to_numpy()
k = int(np.argmin(np.hypot(*(bxy - (dep["sx"], dep["sy"])).T)))
dep_name = f"{dep['도로명'] or '무명도로'} · {bus.iloc[k].시설명} 정류소 {np.hypot(*(bxy[k] - (dep['sx'], dep['sy']))):.0f}m"
stops = pd.concat([stops, pd.DataFrame([{"ID": "DEP_A", "구분": "대기장소", "이름": f"권역A 대기: {dep_name}", "동": "시흥3동", "권역": "A",
                                         "유형": "대기 지점", "x": dep["sx"], "y": dep["sy"], "스냅거리_m": 0.0}])], ignore_index=True)
snaps.append(dep)
M = roads.drive_matrix(snaps)
assert np.isfinite(M[~np.eye(len(M), dtype=bool)]).all(), "도달 불가 칸 있음"
T = M / SPEED_MPM  # 분
ids = stops.ID.tolist()
pd.DataFrame(np.round(M), index=ids, columns=ids).to_csv(f"{OUT}/DRT_v3_도로거리행렬_m.csv", encoding="utf-8-sig")
I = {n: i for i, n in enumerate(stops.ID)}
D_SH = I["D2"]
sumA = lambda d: sum(M[d, i] + M[i, d] for i in A_idx)
print(f"권역 A 대기: {dep_name} (A 5곳 왕복 합 {sumA(I['DEP_A']) / 1000:.2f}km vs 시흥사거리 {sumA(D_SH) / 1000:.2f}km)")

# ---- 목적지 추첨 비중
BI = list(stops.index[stops.구분 == "승차"])
DI = [I[f"D{i + 1}"] for i in range(5)]
d1 = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.행정동 == "시흥1동")][["x_5179", "y_5179"]].to_numpy()
FW = roads.walk_dist(stops.loc[DI[:4], ["x", "y"]].to_numpy(), d1, 400)
fac_cnt = dict(zip(DESTS[:4], np.isfinite(FW).sum(axis=0)))
od = pd.read_csv(f"{OUT}/OD검증_도착지묶음_60세이상.csv", encoding="utf-8-sig", index_col=0)
p_ss = {d: od.loc["⑧ 안양시", f"{d}_%"] / (od.loc["⑧ 안양시", f"{d}_%"] + od.loc["③ 시흥1동", f"{d}_%"]) for d in ["시흥3동", "시흥4동", "시흥5동"]}
WB = roads.walk_dist(stops.loc[BI, ["x", "y"]].to_numpy(), stops.loc[DI, ["x", "y"]].to_numpy(), 400)  # [dest, board]
P = np.zeros((len(BI), 5))
excl = []
for r, bi in enumerate(BI):
    rd = RES_DONG.get(stops.동[bi], stops.동[bi])
    w = np.array([(1 - p_ss[rd]) * fac_cnt[d] / sum(fac_cnt.values()) for d in DESTS[:4]] + [p_ss[rd]])
    near = np.isfinite(WB[:, r])
    w[near] = 0
    excl.append(", ".join(np.array(DESTS)[near]))
    P[r] = w / w.sum()
dw = pd.DataFrame(P, columns=[f"P_{d}" for d in DESTS]).round(3)
dw.insert(0, "승차", stops.ID[BI].values)
dw["보행400m내_제외"] = excl
dw.to_csv(f"{OUT}/DRT_v3_목적지추첨비중.csv", index=False, encoding="utf-8-sig")
print("시흥1동 목적지 시설 수(보행 400m):", fac_cnt, "| 석수역 비중:", {k: round(v, 3) for k, v in p_ss.items()})
print(dw.to_string(index=False))

# ---- 고정노선 (승차 13 + 목적지 5, 최단 비대칭 순회)
nodes = [D_SH] + [i for i in BI + DI if i != D_SH]
n = len(nodes)
m = pulp.LpProblem("ATSP", pulp.LpMinimize)
x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(n) for j in range(n) if i != j}
u = {i: pulp.LpVariable(f"u_{i}", 1, n - 1) for i in range(1, n)}
m += pulp.lpSum(M[nodes[i], nodes[j]] * v for (i, j), v in x.items())
for i in range(n):
    m += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
    m += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            m += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2
m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=900))
nxt = {i: j for (i, j), v in x.items() if v.value() > 0.5}
seq, cur = [0], 0
while (cur := nxt[cur]) != 0:
    seq.append(cur)
loop = [nodes[k] for k in seq]
loop_m = sum(M[a, c] for a, c in zip(loop, loop[1:] + loop[:1]))
cycle = loop_m / SPEED_MPM + DWELL * len(loop)
pd.DataFrame({"순서": range(1, len(loop) + 1), "ID": stops.ID[loop].values, "이름": stops.이름[loop].values,
              "다음까지_m": [round(M[a, c]) for a, c in zip(loop, loop[1:] + loop[:1])]}).to_csv(
    f"{OUT}/DRT_v3_고정노선_순환.csv", index=False, encoding="utf-8-sig")
print(f"고정노선 순환({pulp.LpStatus[m.status]}): {loop_m / 1000:.2f}km, 순환시간 {cycle:.1f}분(정류소당 {DWELL:.0f}분 정차 포함)")

# ---- O-D 시간대 분포
hr = pd.read_csv(f"{OUT}/OD검증_시간대별_시흥1동_왕복_60세이상.csv", encoding="utf-8-sig", index_col=0)
go = hr["가는_3·4·5동→시흥1동"].loc[8:18]
back = hr["오는_시흥1동→3·4·5동"]

stops.drop(columns=[]).to_csv(f"{OUT}/DRT_v3_정류소.csv", index=False, encoding="utf-8-sig")
pickle.dump({"stops": stops, "snaps": snaps, "M": M, "T": T, "I": I, "BI": BI, "DI": DI, "P": P, "loop": loop, "cycle": cycle,
             "loop_m": loop_m, "go_hour": go, "back_hour": back, "fac_cnt": fac_cnt, "p_ss": p_ss, "dep_name": dep_name},
            open(f"{INTERIM}/DRT_v3_입력.pkl", "wb"))
print(f"저장: {INTERIM}/DRT_v3_입력.pkl, {OUT}/DRT_v3_정류소.csv, DRT_v3_도로거리행렬_m.csv, DRT_v3_목적지추첨비중.csv, DRT_v3_고정노선_순환.csv")
