"""OSM 도로망 공통 처리: 주행 그래프(방향), 정차 가능 도로 스냅, 비대칭 도로거리 행렬, 보행 그래프.

원본: data/raw/OSM_시흥클러스터_도로망_*.json (Overpass highway=* 전체, © OpenStreetMap contributors). 좌표는 EPSG:5179.
- 주행: DRIVE 등급. 일방통행(oneway=yes/-1, motorway, roundabout) 반영
- 정차(스냅): SNAP_OK 등급(residential 이상). living_street(보행 우선 골목)·service(단지 내부·주차 통로)·고속·도시고속 제외.
  OSM에 폭(width) 정보가 없어 골목 폭은 도로 등급으로만 판단. 주행 그래프 최대 강연결 요소 위 구간에만 스냅
- 보행: motorway·motorway_link·construction 을 뺀 모든 highway, 방향 무시
"""
import glob
import json

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

DRIVE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
         "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"}
SNAP_OK = {"primary", "secondary", "tertiary", "unclassified", "residential"}
SKIP_SERVICE = {"driveway", "parking_aisle", "drive-through"}
NO_WALK = {"motorway", "motorway_link", "construction", "proposed", "raceway"}


class Roads:
    def __init__(self, raw="data/raw", path=None):
        """path 미지정 시 MCLP에 쓴 시흥클러스터 도로망, TSP v3는 관악 확장판(OSM_도로망_시흥클러스터_관악확장_*.json)을 지정"""
        osm = json.load(open(path or glob.glob(f"{raw}/OSM_시흥클러스터_도로망_*.json")[0], encoding="utf-8"))
        allw = [e for e in osm["elements"] if e["type"] == "way" and "highway" in e.get("tags", {})]
        coords = {}
        for w in allw:
            for nid, g in zip(w["nodes"], w["geometry"]):
                coords[nid] = (g["lon"], g["lat"])
        ids = list(coords)
        pts = gpd.GeoSeries(gpd.points_from_xy([coords[n][0] for n in ids], [coords[n][1] for n in ids]), crs=4326).to_crs(5179)
        self.XY = {n: (p.x, p.y) for n, p in zip(ids, pts)}
        L = lambda u, v: float(np.hypot(self.XY[u][0] - self.XY[v][0], self.XY[u][1] - self.XY[v][1]))

        self.G = nx.DiGraph()   # 주행
        self.W = nx.Graph()     # 보행
        self.segs = []          # (u, v, 길이, 양방향, highway, name)
        for w in allw:
            t = w["tags"]
            hw = t["highway"]
            if hw not in NO_WALK:
                info = {k: t.get(k) for k in ("highway", "name", "bridge", "tunnel", "footway", "crossing", "layer", "covered")}
                for u, v in zip(w["nodes"][:-1], w["nodes"][1:]):
                    self.W.add_edge(u, v, w=L(u, v), way=w["id"], **info)
            if hw not in DRIVE or (hw == "service" and t.get("service") in SKIP_SERVICE):
                continue
            ow = t.get("oneway")
            oneway = ow in ("yes", "1", "true", "-1") or hw in ("motorway", "motorway_link") or t.get("junction") == "roundabout"
            ns = w["nodes"][::-1] if ow == "-1" else w["nodes"]
            for u, v in zip(ns[:-1], ns[1:]):
                self.G.add_edge(u, v, w=L(u, v))
                if not oneway:
                    self.G.add_edge(v, u, w=L(u, v))
                self.segs.append((u, v, L(u, v), not oneway, hw, t.get("name") or t.get("ref")))
        scc = max(nx.strongly_connected_components(self.G), key=len)
        self.snap_segs = [s for s in self.segs if s[4] in SNAP_OK and s[0] in scc and s[1] in scc]
        self.snap_geom = [LineString([self.XY[s[0]], self.XY[s[1]]]) for s in self.snap_segs]
        self.tree = STRtree(self.snap_geom)
        named = [(g, s[5]) for g, s in zip(self.snap_geom, self.snap_segs) if s[5]]
        self.named_geom, self.named_name = [g for g, _ in named], [n for _, n in named]
        self.named_tree = STRtree(self.named_geom)
        wn = list(self.W.nodes)
        self.wnodes = np.array(wn)
        self.wxy = np.array([self.XY[n] for n in wn])
        self.wtree = STRtree([Point(p) for p in self.wxy])

    def snap(self, x, y):
        """정차 가능 도로 위 최근접 지점. 반환: sx, sy, 스냅거리_m, seg, t, 도로등급, 도로명"""
        p = Point(x, y)
        k = int(self.tree.nearest(p))
        g, s = self.snap_geom[k], self.snap_segs[k]
        t = g.project(p) / g.length if g.length else 0.0
        q = g.interpolate(t, normalized=True)
        name = s[5]
        if not name:
            j = int(self.named_tree.nearest(q))
            if self.named_geom[j].distance(q) <= 100:
                name = self.named_name[j]
        return {"sx": q.x, "sy": q.y, "스냅거리_m": p.distance(q), "seg": k, "t": t, "도로등급": s[4], "도로명": name}

    def drive_matrix(self, snaps):
        """snap() 결과 리스트 -> 비대칭 도로거리 행렬(m). 행=출발, 열=도착."""
        def ends(r):
            u, v, L, two, *_ = self.snap_segs[r["seg"]]
            a, b = r["t"] * L, (1 - r["t"]) * L
            ex, en = [(v, b)], [(u, a)]
            if two:
                ex.append((u, a))
                en.append((v, b))
            return ex, en
        E = [ends(r) for r in snaps]
        SP = {s: nx.single_source_dijkstra_path_length(self.G, s, weight="w") for s in {n for ex, _ in E for n, _ in ex}}
        n = len(snaps)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                best = np.inf
                if snaps[i]["seg"] == snaps[j]["seg"]:
                    u, v, L, two, *_ = self.snap_segs[snaps[i]["seg"]]
                    if snaps[j]["t"] >= snaps[i]["t"] or two:
                        best = abs(snaps[j]["t"] - snaps[i]["t"]) * L
                for a, ca in E[i][0]:
                    for b, cb in E[j][1]:
                        d = SP[a].get(b)
                        if d is not None:
                            best = min(best, ca + d + cb)
                M[i, j] = best
        return M

    def drive_path(self, si, sj):
        """스냅 지점 si -> sj 최단 주행 경로 좌표 [(x, y), ...] (지도용)."""
        def ends(r):
            u, v, L, two, *_ = self.snap_segs[r["seg"]]
            ex, en = [(v, (1 - r["t"]) * L)], [(u, r["t"] * L)]
            if two:
                ex.append((u, r["t"] * L))
                en.append((v, (1 - r["t"]) * L))
            return ex, en
        (ex, _), (_, en) = ends(si), ends(sj)
        best, path = np.inf, None
        for a, ca in ex:
            for b, cb in en:
                try:
                    d, p = nx.single_source_dijkstra(self.G, a, b, weight="w")
                except nx.NetworkXNoPath:
                    continue
                if ca + d + cb < best:
                    best, path = ca + d + cb, p
        pts = [(si["sx"], si["sy"])] + [self.XY[n] for n in (path or [])] + [(sj["sx"], sj["sy"])]
        return pts

    def nearest_walk_node(self, xy):
        """점 배열 -> (보행 노드, 연결 직선거리)"""
        idx = np.array([int(self.wtree.nearest(Point(p))) for p in xy])
        return self.wnodes[idx], np.hypot(*(self.wxy[idx] - np.asarray(xy)).T)

    def walk_dist(self, src_xy, dst_xy, cutoff):
        """보행 도로거리(m, 양 끝 연결 직선 포함) 행렬 [len(dst), len(src)]. cutoff 초과는 inf."""
        dn, dc = self.nearest_walk_node(dst_xy)
        sn, sc = self.nearest_walk_node(src_xy)
        out = np.full((len(dst_xy), len(src_xy)), np.inf)
        for j, (s, c0) in enumerate(zip(sn, sc)):
            if c0 > cutoff:
                continue
            d = nx.single_source_dijkstra_path_length(self.W, s, cutoff=cutoff - c0, weight="w")
            v = np.array([c0 + d.get(n, np.inf) + c for n, c in zip(dn, dc)])
            out[:, j] = np.where(v <= cutoff, v, np.inf)
        return out

    def walk_within(self, src_xy, dst_xy, limit):
        """각 출발점에서 보행 도로거리(연결 직선 포함) limit 이내인 도착점 여부 행렬 [len(dst), len(src)]."""
        return np.isfinite(self.walk_dist(src_xy, dst_xy, limit))
