"""DRT 노선 설계 v2: 석수역(1호선)은 S06(석수역 버스정류소) 하차 후 도보 환승, 권역 A 차량은 시흥3동 안에서 대기.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_TSP_v2.py`  (선행: 금천구_MCLP_보행.py, 금천구_DRT_TSP.py(v1 결과 *_v1 비교용))
v1 대비 변경
  1. 목적지 '석수역(1호선)'을 빼고 'S06 석수역 버스정류소' 하차 + 도보 환승으로 바꾼다(S06 -> 석수역 출입구 보행 경로 확인).
     S06이 목적지인 운행에서 S06 호출 승객은 차량을 타지 않는 것으로 처리한다(같은 정류소 승하차).
  2. 권역 A 대기 장소: 시흥3동 안 정차 가능 도로 지점 중 권역 A 승차 정류소까지 왕복 도로거리 합이 최소인 곳.
     단일안(1대)·권역 B는 시흥사거리 대기 유지
그 외 가정·정의는 v1과 같다: 20km/h, 정차 1분(호출 승차 정류소·하차 목적지), k=2·3·4 각 1,000회(seed 42),
  고령인구 비례 비복원 추첨, 편도, 목적지는 300m 내 병의원·복지시설 수 비례 추첨, 분리안은 같은 호출을 권역별로 나눠 2대가 동시 처리,
  고정노선은 담당 승차 정류소를 전체 순회 순서로 전부 지나간 뒤 목적지로(호출 정류소에서만 정차).
"""
import itertools
import json
import sys

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "outputs"
SPEED, DWELL = 20 / 3.6, 60.0
KS, NREP, SEED = [2, 3, 4], 1000, 42
DEST_R = 300
ZONE = {"시흥3동": "A", "시흥4동": "B", "시흥5동": "B"}
S06_NAME = "석수역 버스정류소(승차)"
TRANSFER = "석수역 환승(S06 하차)"
DESTS = ["금천구청역", "시흥사거리", "현대시장.시흥4동주민센터", "무지개아파트"]
ZONE_DEST = {"A": ["시흥사거리", TRANSFER], "B": ["현대시장.시흥4동주민센터", "시흥사거리"]}
SIHEUNG3 = "11180590"

roads = Roads()
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
dest_fac = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.시흥클러스터 == "내부")][["x_5179", "y_5179"]].to_numpy()
n300 = lambda x, y: int((np.hypot(*(dest_fac - (x, y)).T) <= DEST_R).sum())

# ---------------------------------------------------------------- 정류소
b = pd.read_csv(f"{OUT}/MCLP_보행400_p14_승차정류소.csv", encoding="utf-8-sig")
b["표시이름"] = b.표시이름.replace({"석수역": S06_NAME})
b["권역"] = b.동.map(ZONE)
b = b.sort_values(["권역", "동", "배정_70세이상_명(보행)"], ascending=[True, True, False]).reset_index(drop=True)
TYPE = {"기존 버스정류소": "기존 버스정류소", "격자점": "신규 지점", "경로당": "경로당·주민센터", "주민센터": "경로당·주민센터"}
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig").set_index("이름")
mu = fac[(fac.시설명 == "무지개아파트") & (fac.행정동 == "시흥1동")].iloc[0]
dxy = {n: (fixed.loc[n, "x"], fixed.loc[n, "y"]) for n in fixed.index} | {"무지개아파트": (mu.x_5179, mu.y_5179)}
dtype = {"금천구청역": "지하철역", "시흥사거리": "기존 버스정류소", "현대시장.시흥4동주민센터": "기존 버스정류소", "무지개아파트": "기존 버스정류소"}
stops = pd.concat([
    pd.DataFrame({"구분": "승차", "이름": b.표시이름, "동": b.동, "권역": b.권역, "유형": b.후보유형.map(TYPE), "x": b.x_5179, "y": b.y_5179,
                  "배정_70세이상_명": b["배정_70세이상_명(보행)"]}),
    pd.DataFrame({"구분": "목적지", "이름": DESTS, "동": "시흥1동", "권역": "", "유형": [dtype[d] for d in DESTS],
                  "x": [dxy[d][0] for d in DESTS], "y": [dxy[d][1] for d in DESTS], "배정_70세이상_명": np.nan})], ignore_index=True)

# ---------------------------------------------------------------- 1. S06 -> 석수역 출입구 보행 경로
osm_st = json.load(open("data/raw/OSM_석수역_출입구_20261010.json", encoding="utf-8"))
ents = {f"{e['tags']['ref']}번 출입구": (e["lon"], e["lat"]) for e in osm_st["elements"] if e.get("tags", {}).get("railway") == "subway_entrance"}
ep = gpd.GeoSeries(gpd.points_from_xy([v[0] for v in ents.values()], [v[1] for v in ents.values()]), crs=4326).to_crs(5179)
i06 = int(stops.index[stops.이름 == S06_NAME][0])
(sn,), (sc,) = roads.nearest_walk_node([(stops.x[i06], stops.y[i06])])
walk_rows = []
for (name, _), p in zip(ents.items(), ep):
    (en,), (ec,) = roads.nearest_walk_node([(p.x, p.y)])
    path = nx.shortest_path(roads.W, sn, en, weight="w")
    edges = [roads.W[a][c] for a, c in zip(path[:-1], path[1:])]
    L = sc + sum(e["w"] for e in edges) + ec
    kinds = []
    for e in edges:
        k = ("육교" if e.get("bridge") == "yes" and e["highway"] in ("footway", "path", "steps") else
             "지하도" if e.get("tunnel") == "yes" else
             "계단" if e["highway"] == "steps" else
             f"횡단보도({e.get('crossing')})" if e.get("footway") == "crossing" else None)
        if k:
            kinds.append((k, e["w"]))
    agg = {}
    for k, w in kinds:
        agg[k] = agg.get(k, 0) + w
    walk_rows.append({"출입구": name, "직선_m": round(np.hypot(p.x - stops.x[i06], p.y - stops.y[i06])), "보행_m": round(L),
                      "300m초과": "Y" if L > 300 else "", "철로 횡단·특이구간": ", ".join(f"{k} {v:.0f}m" for k, v in agg.items()) or "없음"})
walk = pd.DataFrame(walk_rows).sort_values("출입구")
walk.to_csv(f"{OUT}/DRT_S06_석수역_보행경로.csv", index=False, encoding="utf-8-sig")
print("[1] S06 -> 석수역 출입구 보행 경로 (OSM 보행망; 보도가 따로 그려지지 않은 구간은 도로 중심선으로 근사)")
print(walk.to_string(index=False))

# ---------------------------------------------------------------- 2. 권역 A 대기 장소
snaps = [roads.snap(x, y) for x, y in zip(stops.x, stops.y)]
A_idx = list(stops.index[stops.권역 == "A"])


def ends(s):
    u, v, L, two, *_ = roads.snap_segs[s["seg"]]
    ex, en = [(v, (1 - s["t"]) * L)], [(u, s["t"] * L)]
    if two:
        ex.append((u, s["t"] * L))
        en.append((v, (1 - s["t"]) * L))
    return ex, en


GR = roads.G.reverse(copy=False)
out_cost, in_cost = [], []  # 노드 -> 정류소, 정류소 -> 노드
for i in A_idx:
    ex, en = ends(snaps[i])
    to_stop = {}
    for node, c in en:   # 노드 -> (진입 노드) -> 정류소
        for n_, d in nx.single_source_dijkstra_path_length(GR, node, weight="w").items():
            to_stop[n_] = min(to_stop.get(n_, np.inf), d + c)
    from_stop = {}
    for node, c in ex:   # 정류소 -> (진출 노드) -> 노드
        for n_, d in nx.single_source_dijkstra_path_length(roads.G, node, weight="w").items():
            from_stop[n_] = min(from_stop.get(n_, np.inf), d + c)
    out_cost.append(to_stop)
    in_cost.append(from_stop)
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
s3 = bnd[bnd.ADM_CD == SIHEUNG3].geometry.iloc[0]
cand_nodes = {n for s in roads.snap_segs for n in s[:2]}
cand_nodes = [n for n in cand_nodes if s3.contains(gpd.points_from_xy([roads.XY[n][0]], [roads.XY[n][1]])[0])]
cost = {n: sum(o.get(n, np.inf) + i_.get(n, np.inf) for o, i_ in zip(out_cost, in_cost)) for n in cand_nodes}
best = min(cost, key=cost.get)
dep_snap = roads.snap(*roads.XY[best])
bus = fac[fac.유형 == "버스정류소(연계)"]
bxy = bus[["x_5179", "y_5179"]].to_numpy()
k = int(np.argmin(np.hypot(*(bxy - (dep_snap["sx"], dep_snap["sy"])).T)))
dep_name = f"권역A 대기: {dep_snap['도로명'] or '무명도로'} · {bus.iloc[k].시설명} 정류소 {np.hypot(*(bxy[k] - (dep_snap['sx'], dep_snap['sy']))):.0f}m"
print(f"\n[2] 권역 A 대기 장소 후보 {len(cand_nodes)}개 노드 중 최소: {dep_name} (A 승차 6곳 왕복 도로거리 합 {cost[best] / 1000:.2f}km; "
      f"시흥사거리 대기 시 비교값은 아래 행렬로 계산)")
stops = pd.concat([stops, pd.DataFrame([{"구분": "대기장소", "이름": dep_name, "동": "시흥3동", "권역": "A", "유형": "대기 지점",
                                         "x": dep_snap["sx"], "y": dep_snap["sy"], "배정_70세이상_명": np.nan}])], ignore_index=True)
snaps.append(dep_snap)
stops.insert(0, "정류소ID", [f"S{i + 1:02d}" for i in range(len(stops))])
stops["스냅거리_m"] = [round(s["스냅거리_m"], 1) for s in snaps]
ll = gpd.GeoSeries(gpd.points_from_xy([s["sx"] for s in snaps], [s["sy"] for s in snaps]), crs=5179).to_crs(4326)
stops["위도"], stops["경도"] = ll.y.round(6), ll.x.round(6)
stops["반경300m_목적지수"] = [n300(s["sx"], s["sy"]) if g != "승차" or i == i06 else np.nan
                         for i, (g, s) in enumerate(zip(stops.구분, snaps))]
M = roads.drive_matrix(snaps)
ids = stops.정류소ID.tolist()
pd.DataFrame(np.round(M), index=ids, columns=ids).to_csv(f"{OUT}/DRT_도로거리행렬_v2_m.csv", encoding="utf-8-sig")
idx = {n: i for i, n in enumerate(stops.이름)}
idx[TRANSFER] = i06
DEP_SINGLE, DEP_A = idx["시흥사거리"], len(stops) - 1
sum_a = lambda d: sum(M[d, i] + M[i, d] for i in A_idx)
print(f"    A 승차 6곳 왕복 도로거리 합: 새 대기 지점 {sum_a(DEP_A) / 1000:.2f}km vs 시흥사거리 {sum_a(DEP_SINGLE) / 1000:.2f}km")
BI = stops.index[stops.구분 == "승차"].to_numpy()
w_dest = {d: (stops.반경300m_목적지수[idx[d]]) for d in DESTS + [TRANSFER]}

# ---------------------------------------------------------------- 최종 정류소 목록 (4번)
final = stops[stops.구분 != "대기장소"].copy()
final.loc[final.index == i06, "유형"] = "기존 버스정류소(석수역 환승 하차 겸용)"
final_tbl = final[["정류소ID", "이름", "구분", "동", "권역", "유형", "배정_70세이상_명", "위도", "경도"]].rename(
    columns={"정류소ID": "ID", "이름": "정류소명", "배정_70세이상_명": "배정 고령인구(70세 이상)"})
final_tbl.to_csv(f"{OUT}/DRT_최종정류소목록.csv", index=False, encoding="utf-8-sig")
stops.drop(columns=["x", "y"]).to_csv(f"{OUT}/DRT_정류소_v2.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n[4] 최종 정류소 목록")
print(final_tbl.drop(columns=["위도", "경도"]).to_string(index=False))


# ---------------------------------------------------------------- 3. 전체 순회
def atsp(nodes, start):
    nodes = [start] + [n for n in dict.fromkeys(nodes) if n != start]
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
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=600))
    nxt = {i: j for (i, j), v in x.items() if v.value() > 0.5}
    seq, cur = [0], 0
    while True:
        cur = nxt[cur]
        if cur == 0:
            break
        seq.append(cur)
    order = [nodes[k] for k in seq] + [nodes[0]]
    return pulp.LpStatus[m.status], order, sum(M[a, c] for a, c in zip(order[:-1], order[1:]))


plans = {"단일(1대)": (DEP_SINGLE, list(BI), [idx[d] for d in DESTS + [TRANSFER]]),
         "분리 A": (DEP_A, A_idx, [idx[d] for d in ZONE_DEST["A"]]),
         "분리 B": (DEP_SINGLE, list(stops.index[stops.권역 == "B"].drop(DEP_A, errors="ignore")), [idx[d] for d in ZONE_DEST["B"]])}
plans["분리 B"] = (DEP_SINGLE, [i for i in BI if stops.권역[i] == "B"], plans["분리 B"][2])
tours, trow = {}, []
for key, (dep, bs_, ds_) in plans.items():
    st, order, dist = atsp(bs_ + ds_, dep)
    tours[key] = order
    nstop = len(order) - 2
    trow.append({"안": key, "대기장소": stops.이름[dep], "방문 정류소": nstop, "거리_km": round(dist / 1000, 2),
                 "소요시간_분": round(dist / SPEED / 60 + nstop * DWELL / 60, 1), "상태": st,
                 "순서": " → ".join(stops.정류소ID[o] for o in order)})
tdf = pd.DataFrame(trow)
sp = tdf[tdf.안.str.startswith("분리")]
tdf = pd.concat([tdf, pd.DataFrame([{"안": "분리(2대) 합계/최대", "대기장소": "", "방문 정류소": sp["방문 정류소"].sum(),
                                     "거리_km": round(sp.거리_km.sum(), 2), "소요시간_분": sp.소요시간_분.max(), "상태": "", "순서": ""}])], ignore_index=True)
tdf.to_csv(f"{OUT}/DRT_전체순회_v2.csv", index=False, encoding="utf-8-sig")
print("\n[3-1] 전체 순회 (v2)")
print(tdf.drop(columns="순서").to_string(index=False))


# ---------------------------------------------------------------- 3. 호출 시뮬레이션
def evaluate(seq, calls):
    t, depart, dist = 0.0, {}, 0.0
    for a, c in zip(seq[:-1], seq[1:]):
        dist += M[a, c]
        t += M[a, c] / SPEED
        if c in calls and c not in depart:
            t += DWELL
            depart[c] = t
    arrive = t
    return dist, t + DWELL, [arrive - depart[c] for c in calls]


def trip(dep, calls, dest, fixed_order=None):
    calls = [c for c in calls if c != dest]  # 목적지 정류소에서 호출한 승객은 차량 불필요
    if not calls:
        return None
    if fixed_order is not None:
        return evaluate([dep, *fixed_order, dest], set(calls))
    best = min(itertools.permutations(calls), key=lambda p: sum(M[a, c] for a, c in zip([dep, *p, dest][:-1], [dep, *p, dest][1:])))
    return evaluate([dep, *best, dest], set(calls))


forder = {k: [o for o in tours[k][1:-1] if stops.구분[o] == "승차"] for k in plans}
w_all = stops.loc[BI, "배정_70세이상_명"].to_numpy(float)
pick = lambda names: rng.choice([idx[n] for n in names], p=np.array([w_dest[n] for n in names], float) / sum(w_dest[n] for n in names))
rng = np.random.default_rng(SEED)
recs = []
for k in KS:
    for rep in range(NREP):
        calls = list(rng.choice(BI, size=k, replace=False, p=w_all / w_all.sum()))
        d1 = pick(DESTS + [TRANSFER])
        dz = {z: pick(ZONE_DEST[z]) for z in "AB"}
        for mode in ["DRT", "고정노선"]:
            r = trip(DEP_SINGLE, calls, d1, forder["단일(1대)"] if mode == "고정노선" else None)
            if r:
                recs.append({"k": k, "rep": rep, "안": "단일(1대)", "방식": mode, "운행거리_km": r[0] / 1000, "소요시간_분": r[1] / 60,
                             "1인당승차시간_분": np.mean(r[2]) / 60, "운행차량": 1})
            D_, T_, R_, used = 0.0, 0.0, [], 0
            for z, key, dep in [("A", "분리 A", DEP_A), ("B", "분리 B", DEP_SINGLE)]:
                cz = [c for c in calls if stops.권역[c] == z]
                r = trip(dep, cz, dz[z], forder[key] if mode == "고정노선" else None) if cz else None
                if r:
                    used += 1
                    D_, T_, R_ = D_ + r[0], max(T_, r[1]), R_ + r[2]
            if used:
                recs.append({"k": k, "rep": rep, "안": "분리(2대)", "방식": mode, "운행거리_km": D_ / 1000, "소요시간_분": T_ / 60,
                             "1인당승차시간_분": np.mean(R_) / 60, "운행차량": used})
sim = pd.DataFrame(recs)
sim.to_csv("data/interim/DRT_시뮬레이션_원자료_v2.csv", index=False, encoding="utf-8-sig")
q90 = lambda s: s.quantile(0.9)
agg = sim.groupby(["k", "안", "방식"]).agg(
    운행거리_평균=("운행거리_km", "mean"), 운행거리_중앙=("운행거리_km", "median"), 운행거리_p90=("운행거리_km", q90),
    소요시간_평균=("소요시간_분", "mean"), 소요시간_중앙=("소요시간_분", "median"), 소요시간_p90=("소요시간_분", q90),
    승차시간_평균=("1인당승차시간_분", "mean"), 승차시간_중앙=("1인당승차시간_분", "median"), 승차시간_p90=("1인당승차시간_분", q90),
    평균운행차량=("운행차량", "mean"), 운행수=("rep", "count")).round(2).reset_index()
agg.to_csv(f"{OUT}/DRT_호출시뮬레이션_요약_v2.csv", index=False, encoding="utf-8-sig")
print(f"\n[3-2] 호출 시뮬레이션 v2 (k별 {NREP}회; 목적지=호출 정류소인 승객만 있는 운행은 제외되어 운행수 표시)")
print(agg.to_string(index=False))

# ---------------------------------------------------------------- v1 vs v2 비교
v1t = pd.read_csv(f"{OUT}/DRT_전체순회_v1.csv", encoding="utf-8-sig").set_index("안")
v1s = pd.read_csv(f"{OUT}/DRT_호출시뮬레이션_요약_v1.csv", encoding="utf-8-sig")
tt = tdf.set_index("안")
cmp_t = pd.DataFrame({"v1_거리_km": v1t.거리_km, "v2_거리_km": tt.거리_km, "v1_시간_분": v1t.소요시간_분, "v2_시간_분": tt.소요시간_분})
cmp_t["거리차_km"] = (cmp_t.v2_거리_km - cmp_t.v1_거리_km).round(2)
cmp_t["시간차_분"] = (cmp_t.v2_시간_분 - cmp_t.v1_시간_분).round(1)
cmp_t.to_csv(f"{OUT}/DRT_v1_vs_v2_전체순회.csv", encoding="utf-8-sig")
cols = ["운행거리_평균", "소요시간_평균", "승차시간_평균", "운행거리_p90", "소요시간_p90", "승차시간_p90"]
cmp_s = v1s.set_index(["k", "안", "방식"])[cols].join(agg.set_index(["k", "안", "방식"])[cols], lsuffix="_v1", rsuffix="_v2")
for c in cols:
    cmp_s[f"{c}_차이"] = (cmp_s[f"{c}_v2"] - cmp_s[f"{c}_v1"]).round(2)
cmp_s = cmp_s[[f"{c}_{s}" for c in cols[:3] for s in ("v1", "v2", "차이")]].reset_index()
cmp_s.to_csv(f"{OUT}/DRT_v1_vs_v2_시뮬레이션.csv", index=False, encoding="utf-8-sig")
print("\n[비교] 전체 순회 v1 -> v2")
print(cmp_t.to_string())
print("\n[비교] 시뮬레이션 평균 v1 -> v2 (거리 km, 시간 분)")
print(cmp_s[cmp_s.방식 == "DRT"].to_string(index=False))
print(cmp_s[cmp_s.방식 == "고정노선"].to_string(index=False))
piv = agg.pivot_table(index=["k", "안"], columns="방식", values=["운행거리_평균", "소요시간_평균", "승차시간_평균"])
eff = pd.DataFrame({f"{m}_DRT감소율_%": ((1 - piv[(m, "DRT")] / piv[(m, "고정노선")]) * 100).round(1)
                    for m in ["운행거리_평균", "소요시간_평균", "승차시간_평균"]}).reset_index()
eff.to_csv(f"{OUT}/DRT_vs_고정노선_효율_v2.csv", index=False, encoding="utf-8-sig")
print("\n[DRT가 고정노선보다 줄이는 비율 v2]")
print(eff.to_string(index=False))

# ---------------------------------------------------------------- 지도
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
SURF, INK, MUTED, FIX = "#fcfcfb", "#0b0b0b", "#898781", "#eb6834"
CA, CB = "#2a78d6", "#1baf7a"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sx = np.array([s["sx"] for s in snaps])
sy = np.array([s["sy"] for s in snaps])
PATHS = []


def draw(ax, order, color):
    for a, c in zip(order[:-1], order[1:]):
        pts = np.array(roads.drive_path(snaps[a], snaps[c]))
        PATHS.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.2, alpha=0.85, zorder=3, solid_capstyle="round")
        mid = len(pts) // 2
        if len(pts) > 2:
            ax.annotate("", xy=pts[min(mid + 1, len(pts) - 1)], xytext=pts[mid],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=11), zorder=4)


def base(ax, title):
    ax.set_facecolor(SURF)
    cl.boundary.plot(ax=ax, color="#c3c2b7", lw=1, zorder=1)
    for r in cl.itertuples():
        pt = r.geometry.representative_point()
        ax.text(pt.x - 250, pt.y + 150, r.ADM_NM, fontsize=10, color=MUTED, ha="center", zorder=2,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)


def layer(ax, colors):
    for i in BI:
        ax.scatter(sx[i], sy[i], s=60, color=colors[stops.권역[i]], edgecolor=SURF, lw=1.5, zorder=6)
        ax.annotate(stops.정류소ID[i] + (" (석수역 환승)" if i == i06 else ""), (sx[i], sy[i]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
    for d in DESTS:
        i = idx[d]
        ax.scatter(sx[i], sy[i], s=170, marker="s" if d == "시흥사거리" else "*", color=FIX, edgecolor=SURF, lw=1, zorder=6)
        ax.annotate(d.replace(".시흥4동주민센터", ""), (sx[i], sy[i]), xytext=(7, -11), textcoords="offset points", fontsize=8,
                    color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])


t2 = tdf.set_index("안")
fig, axes = plt.subplots(1, 2, figsize=(14, 9), dpi=150, facecolor=SURF)
base(axes[0], f"단일 노선 전체 순회 (v2): {t2.loc['단일(1대)', '거리_km']}km, {t2.loc['단일(1대)', '소요시간_분']}분\n"
              f"(시흥사거리 대기, 석수역은 S06 하차·도보 환승)")
draw(axes[0], tours["단일(1대)"], CA)
layer(axes[0], {"A": CA, "B": CA})
base(axes[1], f"권역 분리 전체 순회 (v2): A {t2.loc['분리 A', '거리_km']}km·{t2.loc['분리 A', '소요시간_분']}분 / "
              f"B {t2.loc['분리 B', '거리_km']}km·{t2.loc['분리 B', '소요시간_분']}분\n(A: 시흥3동 대기, 목적지 시흥사거리+S06 / B: 시흥사거리 대기, 목적지 현대시장+시흥사거리)")
draw(axes[1], tours["분리 A"], CA)
draw(axes[1], tours["분리 B"], CB)
layer(axes[1], {"A": CA, "B": CB})
axes[1].scatter(sx[DEP_A], sy[DEP_A], s=150, marker="s", color=CA, edgecolor=SURF, lw=1.5, zorder=8)
axes[1].annotate("A 대기", (sx[DEP_A], sy[DEP_A]), xytext=(-8, -13), textcoords="offset points", ha="right", fontsize=8.5, color=INK,
                 zorder=8, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
allp = np.vstack(PATHS + [np.c_[sx, sy]])
ext = np.r_[np.minimum(cl.total_bounds[:2], allp.min(0)) - 200, np.maximum(cl.total_bounds[2:], allp.max(0)) + 200]
for ax in axes:
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
axes[1].legend(handles=[Line2D([], [], color=CA, lw=2.5, label="권역 A 차량 (시흥3동)"), Line2D([], [], color=CB, lw=2.5, label="권역 B 차량 (시흥4·5동)"),
                        Line2D([], [], marker="s", ls="", color=CA, ms=9, label="권역 A 대기 지점"),
                        Line2D([], [], marker="s", ls="", color=FIX, ms=9, label="시흥사거리 (단일안·B 대기, 목적지)"),
                        Line2D([], [], marker="*", ls="", color=FIX, ms=12, label="목적지 정류소")], frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "도로거리: OSM 주행망(일방통행 반영). 가정: 20km/h, 정류소당 1분 정차. 정류소 번호는 DRT_최종정류소목록.csv 참조. © OpenStreetMap contributors",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/DRT_노선지도_v2.png", facecolor=SURF)
plt.close(fig)
print(f"\n저장: DRT_S06_석수역_보행경로.csv, DRT_최종정류소목록.csv, DRT_정류소_v2.csv, DRT_도로거리행렬_v2_m.csv, DRT_전체순회_v2.csv, "
      f"DRT_호출시뮬레이션_요약_v2.csv, DRT_v1_vs_v2_전체순회.csv, DRT_v1_vs_v2_시뮬레이션.csv, DRT_vs_고정노선_효율_v2.csv, DRT_노선지도_v2.png")
