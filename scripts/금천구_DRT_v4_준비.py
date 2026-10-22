"""DRT v4 준비: v3 입력(정류소·행렬·목적지 비중)을 그대로 쓰고 권역 재구성·권역별 대기 장소·B01 제외 고정노선 순환을 더한다.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_v4_준비.py`  (선행: 금천구_DRT_v3_준비.py)
권역 재구성: 동 단위(같은 동 정류소는 같은 권역)로, 서로 맞닿은 동끼리만 묶는 2분할 중 배정 고령인구 차이가 최소인 안.
  B01(관악구 난향동)은 이용자가 시흥4동 주민이라 시흥4동과 같은 권역. 인접 판정: 행정동 경계 폴리곤 접촉(10m 버퍼)
대기 장소: 각 권역 소속 동 안의 정차 가능 도로 노드 중 권역 승차 정류소 왕복 도로거리 합 최소 지점
고정노선: 승차 12(B01 제외) + 목적지 5 최단 비대칭 순회(기본안), v3의 B01 포함 순회는 참고
"""
import itertools
import pickle
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

INTERIM, OUT = "data/interim", "outputs"
ROAD = "data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json"
SPEED_MPM = 20000 / 60
DONG_CD = {"시흥3동": "11180590", "시흥4동": "11180600", "시흥5동": "11180610"}
D = pickle.load(open(f"{INTERIM}/DRT_v3_입력.pkl", "rb"))
st, M, I, BI, DI = D["stops"], D["M"], D["I"], D["BI"], D["DI"]
roads = Roads(path=ROAD)

# ---- 권역 재구성 (동 단위, 인접한 동끼리)
home = {i: ("시흥4동" if st.동[i] == "난향동" else st.동[i]) for i in BI}
pop = {d: st.loc[[i for i in BI if home[i] == d], "배정_70세이상_명"].sum() for d in DONG_CD}
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
poly = {d: bnd[bnd.ADM_CD == c].geometry.iloc[0] for d, c in DONG_CD.items()}
adj = {(a, b): poly[a].buffer(10).intersects(poly[b]) for a, b in itertools.combinations(DONG_CD, 2)}
print("동 인접:", {f"{a}-{b}": v for (a, b), v in adj.items()}, "| 동별 배정:", pop)
best = None
for k in (1, 2):
    for grp in itertools.combinations(DONG_CD, k):
        rest = [d for d in DONG_CD if d not in grp]
        conn = lambda g: len(g) == 1 or all(adj.get((a, b), adj.get((b, a))) for a, b in itertools.combinations(g, 2))
        if not (conn(grp) and conn(rest)):
            continue
        diff = abs(sum(pop[d] for d in grp) - sum(pop[d] for d in rest))
        print(f"  후보 {list(grp)} / {rest}: {sum(pop[d] for d in grp):.0f} : {sum(pop[d] for d in rest):.0f} (차이 {diff:.0f})")
        if best is None or diff < best[0]:
            best = (diff, list(grp), rest)
gA, gB = (best[1], best[2]) if "시흥3동" in best[1] else (best[2], best[1])
ZONE4 = {i: ("A" if home[i] in gA else "B") for i in BI}
print(f"채택: 권역 A = {gA} ({sum(pop[d] for d in gA):.0f}명), 권역 B = {gB} + B01 ({sum(pop[d] for d in gB):.0f}명)")

# ---- 권역별 대기 장소
snaps = list(D["snaps"])
snap_nodes = {n for s in roads.snap_segs for n in s[:2]}
GR = roads.G.reverse(copy=False)


def ends(s):
    u, v, L, two, *_ = roads.snap_segs[s["seg"]]
    ex, en = [(v, (1 - s["t"]) * L)], [(u, s["t"] * L)]
    if two:
        ex.append((u, s["t"] * L))
        en.append((v, (1 - s["t"]) * L))
    return ex, en


fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
bxy = bus[["x_5179", "y_5179"]].to_numpy()
deps = {}
for z, dongs in [("A", gA), ("B", gB)]:
    area = gpd.GeoSeries([poly[d] for d in dongs]).union_all()
    cand = [n for n in snap_nodes if area.contains(gpd.points_from_xy([roads.XY[n][0]], [roads.XY[n][1]])[0])]
    cost = dict.fromkeys(cand, 0.0)
    for i in [i for i in BI if ZONE4[i] == z]:
        ex, en = ends(snaps[i])
        to_s, fr_s = {}, {}
        for node, c in en:
            for n_, dd in nx.single_source_dijkstra_path_length(GR, node, weight="w").items():
                to_s[n_] = min(to_s.get(n_, np.inf), dd + c)
        for node, c in ex:
            for n_, dd in nx.single_source_dijkstra_path_length(roads.G, node, weight="w").items():
                fr_s[n_] = min(fr_s.get(n_, np.inf), dd + c)
        for n in cand:
            cost[n] += to_s.get(n, np.inf) + fr_s.get(n, np.inf)
    nb = min(cost, key=cost.get)
    s = roads.snap(*roads.XY[nb])
    k = int(np.argmin(np.hypot(*(bxy - (s["sx"], s["sy"])).T)))
    deps[z] = (s, f"권역{z} 대기: {s['도로명'] or '무명도로'} · {bus.iloc[k].시설명} 정류소 {np.hypot(*(bxy[k] - (s['sx'], s['sy']))):.0f}m", cost[nb])
# 행렬 확장: 기존 정류소 + 새 대기 장소 2곳
names = list(st.ID) + ["DEP4_A", "DEP4_B"]
snaps += [deps["A"][0], deps["B"][0]]
M4 = roads.drive_matrix(snaps)
assert np.isfinite(M4[~np.eye(len(M4), dtype=bool)]).all()
I4 = {n: k for k, n in enumerate(names)}
for z in "AB":
    zi = [i for i in BI if ZONE4[i] == z]
    rt = lambda d: sum(M4[d, i] + M4[i, d] for i in zi)
    print(f"{deps[z][1]} | 권역 {z} 정류소 {len(zi)}곳 왕복 합 {rt(I4[f'DEP4_{z}']) / 1000:.2f}km (시흥사거리 대기 시 {rt(I4['D2']) / 1000:.2f}km)")

# ---- 고정노선: B01 제외 순환
def atsp(nodes):
    n = len(nodes)
    m = pulp.LpProblem("ATSP", pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(n) for j in range(n) if i != j}
    u = {i: pulp.LpVariable(f"u_{i}", 1, n - 1) for i in range(1, n)}
    m += pulp.lpSum(M4[nodes[i], nodes[j]] * v for (i, j), v in x.items())
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
    return [nodes[k] for k in seq]


b01 = I["B01"]
loop0 = atsp([I["D2"]] + [i for i in BI + DI if i not in (I["D2"], b01)])
loop0_m = sum(M4[a, c] for a, c in zip(loop0, loop0[1:] + loop0[:1]))
cycle0 = loop0_m / SPEED_MPM + 1.0 * len(loop0)
pd.DataFrame({"순서": range(1, len(loop0) + 1), "ID": [names[i] for i in loop0], "이름": st.이름[loop0].values,
              "다음까지_m": [round(M4[a, c]) for a, c in zip(loop0, loop0[1:] + loop0[:1])]}).to_csv(
    f"{OUT}/DRT_v4_고정노선_순환_B01제외.csv", index=False, encoding="utf-8-sig")
print(f"고정노선 B01 제외: {loop0_m / 1000:.2f}km, {cycle0:.1f}분 (참고 B01 포함 {D['loop_m'] / 1000:.2f}km, {D['cycle']:.1f}분)")

zt = pd.DataFrame({"ID": st.ID[BI].values, "이름": st.이름[BI].values, "동": st.동[BI].values, "배정_70세이상_명": st.배정_70세이상_명[BI].values,
                   "권역_v3": st.권역[BI].values, "권역_v4": [ZONE4[i] for i in BI]})
zt.to_csv(f"{OUT}/DRT_v4_권역구성.csv", index=False, encoding="utf-8-sig")
print(zt.groupby("권역_v4").agg(정류소=("ID", "count"), 배정=("배정_70세이상_명", "sum")).to_string())
D4 = dict(D)
D4.update({"M": M4, "T": M4 / SPEED_MPM, "I": I4, "names": names, "snaps": snaps, "ZONE4": ZONE4, "dep_names": {z: deps[z][1] for z in "AB"},
           "loop0": loop0, "cycle0": cycle0, "loop0_m": loop0_m, "gA": gA, "gB": gB})
pickle.dump(D4, open(f"{INTERIM}/DRT_v4_입력.pkl", "wb"))
print(f"저장: {INTERIM}/DRT_v4_입력.pkl, {OUT}/DRT_v4_권역구성.csv, {OUT}/DRT_v4_고정노선_순환_B01제외.csv")
