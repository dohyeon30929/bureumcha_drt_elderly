"""DRT 정류소 14곳(승차 11 + 목적지 3) 도로 스냅 + 도로거리 행렬 (TSP 입력).

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_도로망.py`  (선행: scripts/금천구_MCLP_p결정.py)
도로망: data/raw/OSM_시흥클러스터_도로망_*.json (Overpass, highway=* 전체, © OpenStreetMap contributors)
  - 주행 그래프: 차량 통행 도로(DRIVE). 일방통행(oneway=yes/-1, motorway)을 반영한 방향 그래프이며, 행렬은 비대칭일 수 있다.
  - 정차(스냅) 허용 도로: SNAP_OK 등급만. living_street(보행 우선 골목)·service(단지 내부·주차 통로)·고속·도시고속 제외
    OSM에 폭(width) 정보가 없어 '골목 폭'은 도로 등급으로만 판단한다
  - 스냅 대상 구간은 주행 그래프의 최대 강연결 요소에 속한 것만(고립 구간에 정차하지 않게)
신규 격자점 8곳은 스냅 위치로 옮겨 커버율(직선 300m, 최적화와 같은 기준)을 다시 계산하고,
스냅 지점의 도로명 + 가장 가까운 버스정류소명으로 임시 이름을 붙인다.
기존 정류소·경로당·목적지 정류소도 같은 규칙으로 도로 위 지점에 스냅해서 행렬을 만든다(위치 자체는 바꾸지 않음).
"""
import glob
import heapq
import json
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

RAW, OUT = "data/raw", "outputs"
R = 300
P_SEL = 11
FAR_M = 50
DRIVE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
         "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"}
SNAP_OK = {"primary", "secondary", "tertiary", "unclassified", "residential"}
SKIP_SERVICE = {"driveway", "parking_aisle", "drive-through"}

# ---------------------------------------------------------------- 도로 그래프
osm = json.load(open(glob.glob(f"{RAW}/OSM_시흥클러스터_도로망_*.json")[0], encoding="utf-8"))
ways = [e for e in osm["elements"] if e["type"] == "way" and e.get("tags", {}).get("highway") in DRIVE]
ways = [w for w in ways if not (w["tags"]["highway"] == "service" and w["tags"].get("service") in SKIP_SERVICE)]
coords = {}
for w in ways:
    for nid, g in zip(w["nodes"], w["geometry"]):
        coords[nid] = (g["lon"], g["lat"])
nid_list = list(coords)
xy = gpd.GeoSeries(gpd.points_from_xy([coords[n][0] for n in nid_list], [coords[n][1] for n in nid_list]), crs=4326).to_crs(5179)
XY = {n: (p.x, p.y) for n, p in zip(nid_list, xy)}

G = nx.DiGraph()
segs = []  # (u, v, 길이, 양방향여부, highway, name, way_id)
for w in ways:
    t = w["tags"]
    ow = t.get("oneway")
    oneway = ow in ("yes", "1", "true", "-1") or t["highway"] in ("motorway", "motorway_link") or t.get("junction") == "roundabout"
    ns = w["nodes"][::-1] if ow == "-1" else w["nodes"]
    for u, v in zip(ns[:-1], ns[1:]):
        L = float(np.hypot(XY[u][0] - XY[v][0], XY[u][1] - XY[v][1]))
        G.add_edge(u, v, w=L)
        if not oneway:
            G.add_edge(v, u, w=L)
        segs.append((u, v, L, not oneway, t["highway"], t.get("name") or t.get("ref"), w["id"]))
scc = max(nx.strongly_connected_components(G), key=len)
print(f"주행 그래프: 노드 {G.number_of_nodes():,}, 방향 간선 {G.number_of_edges():,}, 최대 강연결 요소 {len(scc):,}노드")

snap_segs = [s for s in segs if s[4] in SNAP_OK and s[0] in scc and s[1] in scc]
snap_geom = [LineString([XY[s[0]], XY[s[1]]]) for s in snap_segs]
tree = STRtree(snap_geom)
named = [(g, s[5]) for g, s in zip(snap_geom, snap_segs) if s[5]]
named_tree = STRtree([g for g, _ in named])


def snap(x, y):
    p = Point(x, y)
    k = int(tree.nearest(p))
    g, s = snap_geom[k], snap_segs[k]
    t = g.project(p) / g.length if g.length else 0.0
    q = g.interpolate(t, normalized=True)
    name = s[5]
    if not name:  # 이름 없는 골목이면 100m 안의 가장 가까운 이름 있는 도로
        j = int(named_tree.nearest(q))
        if named[j][0].distance(q) <= 100:
            name = named[j][1]
    return {"sx": q.x, "sy": q.y, "스냅거리_m": p.distance(q), "seg": k, "t": t, "도로등급": s[4], "도로명": name}


# ---------------------------------------------------------------- 정류소 14곳
sel = pd.read_csv(f"{OUT}/MCLP_r300_p{P_SEL}_선정거점.csv", encoding="utf-8-sig")
cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig").set_index("후보ID")
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig")
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
bus = fac[fac.유형 == "버스정류소(연계)"]
stops = pd.concat([
    pd.DataFrame({"구분": "승차", "후보ID": sel.후보ID, "후보유형": sel.후보유형, "원이름": sel.이름, "동": sel.동,
                  "x": cand.loc[sel.후보ID, "x"].values, "y": cand.loc[sel.후보ID, "y"].values}),
    pd.DataFrame({"구분": "목적지", "후보ID": "", "후보유형": "고정 목적지 정류소", "원이름": fixed.이름, "동": "시흥1동",
                  "x": fixed.x, "y": fixed.y})], ignore_index=True)
sn = pd.DataFrame([snap(r.x, r.y) for r in stops.itertuples()])
stops = pd.concat([stops, sn], axis=1)
grid = stops.후보유형 == "격자점"
bxy = bus[["x_5179", "y_5179"]].to_numpy()


def temp_name(r):
    k = int(np.argmin(np.hypot(*(bxy - (r.sx, r.sy)).T)))
    d = np.hypot(*(bxy[k] - (r.sx, r.sy)))
    lm = f"{bus.iloc[k].시설명} 정류소 {d:.0f}m" if d <= 250 else None
    return " · ".join(x for x in [r.도로명, lm] if isinstance(x, str) and x) or f"{r.동} 무명도로"


stops["표시이름"] = stops.원이름
stops.loc[grid, "표시이름"] = [temp_name(r) for r in stops[grid].itertuples()]
stops["정류소ID"] = [f"S{i + 1:02d}" for i in range(len(stops))]
stops[f"스냅_{FAR_M}m초과"] = np.where(stops.스냅거리_m > FAR_M, "Y", "")
ll = gpd.GeoSeries(gpd.points_from_xy(stops.sx, stops.sy), crs=5179).to_crs(4326)
stops["위도"], stops["경도"] = ll.y.round(6), ll.x.round(6)

# ---------------------------------------------------------------- 커버율 (스냅 전/후)
dem = pd.read_csv(f"{OUT}/MCLP_수요점_시흥345_70세.csv", encoding="utf-8-sig")
W = dem.가중치_70세이상.to_numpy()
dx, dy = dem.x_5179.to_numpy()[:, None], dem.y_5179.to_numpy()[:, None]
b = stops[stops.구분 == "승차"]
cov0 = (np.hypot(dx - b.x.values, dy - b.y.values) <= R).any(axis=1)
nx_, ny_ = np.where(b.후보유형 == "격자점", b.sx, b.x), np.where(b.후보유형 == "격자점", b.sy, b.y)
cov1 = (np.hypot(dx - nx_, dy - ny_) <= R).any(axis=1)
c0, c1 = W[cov0].sum() / W.sum(), W[cov1].sum() / W.sum()
print(f"\n커버율(직선 {R}m): 스냅 전 {c0:.1%} -> 신규 격자점 8곳 스냅 후 {c1:.1%} ({(c1 - c0) * 100:+.1f}%p, {W[cov1].sum() - W[cov0].sum():+,.0f}명)")
lost = W[cov0 & ~cov1].sum()
gain = W[~cov0 & cov1].sum()
print(f"    새로 빠진 수요 {lost:,.0f}명 / 새로 커버된 수요 {gain:,.0f}명")

# ---------------------------------------------------------------- 도로거리 행렬 (방향 고려)
def ends(r):
    """스냅 지점에서 나가는 노드(exits)와 들어오는 노드(entries), 각 비용."""
    u, v, L, two, *_ = snap_segs[r.seg]
    a, bb = r.t * L, (1 - r.t) * L
    exits, entries = [(v, bb)], [(u, a)]
    if two:
        exits.append((u, a))
        entries.append((v, bb))
    return exits, entries


EX = [ends(r) for r in stops.itertuples()]
sources = {n for ex, _ in EX for n, _ in ex}
SP = {s: nx.single_source_dijkstra_path_length(G, s, weight="w") for s in sources}
n = len(stops)
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        best = np.inf
        ri, rj = stops.iloc[i], stops.iloc[j]
        if ri.seg == rj.seg:  # 같은 구간 위: 진행 방향(또는 양방향)으로 바로
            u, v, L, two, *_ = snap_segs[ri.seg]
            if rj.t >= ri.t or two:
                best = abs(rj.t - ri.t) * L
        for a, ca in EX[i][0]:
            for bnode, cb in EX[j][1]:
                d = SP[a].get(bnode)
                if d is not None:
                    best = min(best, ca + d + cb)
        M[i, j] = best
ids = stops.정류소ID.tolist()
Mdf = pd.DataFrame(np.round(M), index=ids, columns=ids)
Mdf.to_csv(f"{OUT}/DRT_도로거리행렬_14x14_m.csv", encoding="utf-8-sig")
E = np.hypot(stops.sx.values[:, None] - stops.sx.values[None, :], stops.sy.values[:, None] - stops.sy.values[None, :])
off = ~np.eye(n, dtype=bool)
ratio = M[off] / E[off]
asym = np.abs(M - M.T)[np.triu_indices(n, 1)]
print(f"\n도로거리 행렬 {n}x{n}: 도달 불가 {np.isinf(M[off]).sum()}칸 | 도로/직선 비율 중앙값 {np.median(ratio):.2f} "
      f"(범위 {ratio.min():.2f}~{ratio.max():.2f}) | 왕복 비대칭 >100m {(asym > 100).sum()}쌍 (최대 {asym.max():.0f}m)")

cols = ["정류소ID", "구분", "표시이름", "원이름", "후보유형", "후보ID", "동", "위도", "경도", "sx", "sy", "x", "y",
        "스냅거리_m", f"스냅_{FAR_M}m초과", "도로등급", "도로명"]
st_out = stops[cols].rename(columns={"sx": "x_5179_스냅", "sy": "y_5179_스냅", "x": "x_5179_원", "y": "y_5179_원"})
st_out["스냅거리_m"] = st_out.스냅거리_m.round(1)
st_out.to_csv(f"{OUT}/DRT_정류소14_스냅.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n정류소 14곳:")
print(st_out[["정류소ID", "구분", "표시이름", "원이름", "동", "스냅거리_m", f"스냅_{FAR_M}m초과", "도로등급"]].to_string(index=False))
print("\n도로거리 행렬(m):")
print(Mdf.astype(int).to_string())
print(f"\n저장: {OUT}/DRT_정류소14_스냅.csv, {OUT}/DRT_도로거리행렬_14x14_m.csv")
