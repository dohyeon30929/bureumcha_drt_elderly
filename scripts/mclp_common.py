"""MCLP v4 공통: 후보지 구성(경로당·주민센터 정차 위치 보정 포함), MCLP 풀이, 이름 붙이기."""
import networkx as nx
import numpy as np
import pandas as pd
import pulp

OUT = "outputs"
NEAR_BUS = 100
LM_SEARCH_M = 300  # 경로당·주민센터에서 정차 가능 도로를 찾는 최대 보행거리


def load_demand():
    return pd.read_csv(f"{OUT}/MCLP_v3_수요점_100m보정.csv", encoding="utf-8-sig")


def build_candidates(roads):
    """v3 후보(격자점 스냅, 버스정류소 100m 이내 격자점 제거) + 경로당·주민센터를 '걸어서 가장 가까운 정차 가능 도로 지점'으로 이동.
    아파트 단지 안 시설은 단지 내 보행로를 따라 나가는 출입구 쪽 도로 지점이 선택된다."""
    cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
    raw_bus = pd.read_csv("data/interim/금천구_목적지후보_상세.csv", encoding="utf-8-sig")
    raw_bus = raw_bus[(raw_bus.유형 == "버스정류소(연계)") & raw_bus.시흥클러스터.notna()]
    c = cand.copy()
    c["원x"], c["원y"], c["도로명"], c["이동_직선m"], c["이동_보행m"], c["이동방식"] = c.x, c.y, None, 0.0, 0.0, ""
    g = c.후보유형 == "격자점"
    for i in c.index[g]:
        s = roads.snap(c.at[i, "x"], c.at[i, "y"])
        c.loc[i, ["x", "y", "도로명", "이동_직선m"]] = [s["sx"], s["sy"], s["도로명"], s["스냅거리_m"]]
        c.at[i, "이동방식"] = "도로 스냅(직선 최근접)"
    bxy = raw_bus[["x_5179", "y_5179"]].to_numpy()
    c = c[~(np.array([np.hypot(*(bxy - (x, y)).T).min() <= NEAR_BUS for x, y in zip(c.x, c.y)]) & g)].reset_index(drop=True)

    snap_nodes = {n for s in roads.snap_segs for n in s[:2]}
    lm = c.후보유형.isin(["경로당", "주민센터"])
    for i in c.index[lm]:
        x0, y0 = c.at[i, "x"], c.at[i, "y"]
        (n0,), (c0,) = roads.nearest_walk_node([(x0, y0)])
        d = nx.single_source_dijkstra_path_length(roads.W, n0, cutoff=LM_SEARCH_M, weight="w")
        reach = [(dd, n) for n, dd in d.items() if n in snap_nodes]
        straight = roads.snap(x0, y0)
        if reach:
            dd, n = min(reach)
            s = roads.snap(*roads.XY[n])  # 노드가 속한 정차 가능 구간 위 지점
            c.loc[i, ["x", "y", "도로명"]] = [s["sx"], s["sy"], s["도로명"]]
            c.at[i, "이동_보행m"] = c0 + dd
            c.at[i, "이동방식"] = "보행 최근접 정차 도로(출입구 방향)"
        else:
            c.loc[i, ["x", "y", "도로명"]] = [straight["sx"], straight["sy"], straight["도로명"]]
            c.at[i, "이동방식"] = "도로 스냅(보행 경로 없음)"
        c.at[i, "이동_직선m"] = np.hypot(c.at[i, "x"] - x0, c.at[i, "y"] - y0)
        c.at[i, "직선최근접과_차이m"] = np.hypot(c.at[i, "x"] - straight["sx"], c.at[i, "y"] - straight["sy"])
    return c


def solve_mclp(W, A, p, groups=None, floors=None, time_limit=300):
    """max sum W*y. groups: 수요점별 그룹 라벨 배열, floors: {그룹: 최소 커버 인원}."""
    m = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(A.shape[1]), cat="Binary")
    y = pulp.LpVariable.dicts("y", range(len(W)), cat="Binary")
    m += pulp.lpSum(W[i] * y[i] for i in range(len(W)))
    for i in range(len(W)):
        js = np.flatnonzero(A[i])
        m += y[i] <= pulp.lpSum(x[j] for j in js) if len(js) else y[i] == 0
    m += pulp.lpSum(x.values()) == p
    if floors:
        for gname, need in floors.items():
            idx = np.flatnonzero(groups == gname)
            m += pulp.lpSum(W[i] * y[i] for i in idx) >= need
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    st = pulp.LpStatus[m.status]
    sel = [j for j in range(A.shape[1]) if (x[j].value() or 0) > 0.5] if st == "Optimal" else []
    return st, sel


def p_curve(W, A, p_max=30, target=0.95, min_gain=0.02):
    tot = W.sum()
    rows, sols = [], {}
    for p in range(1, p_max + 1):
        st, sel = solve_mclp(W, A, p)
        sols[p] = sel
        rows.append({"p": p, "커버_명": W[A[:, sel].any(axis=1)].sum(), "상태": st})
    t = pd.DataFrame(rows)
    t["커버율"] = t.커버_명 / tot
    t["추가커버_명"] = t.커버_명.diff().fillna(t.커버_명)
    t["추가커버_%p"] = t.커버율.diff().fillna(t.커버율) * 100
    nxt = t["추가커버_%p"].shift(-1)
    pa = t.loc[t.커버율 >= target, "p"]
    pA = int(pa.min()) if len(pa) else None
    pB = int(t.loc[nxt < min_gain * 100, "p"].min())
    return t, sols, pA, pB, max(v for v in (pA, pB) if v is not None)


def stop_name(r, bus):
    if r.후보유형 not in ("격자점", "추가 도로지점"):
        return str(r.이름).strip()
    fb = bus[["x_5179", "y_5179"]].to_numpy()
    k = int(np.argmin(np.hypot(*(fb - (r.x, r.y)).T)))
    d = np.hypot(*(fb[k] - (r.x, r.y)))
    parts = [r.도로명 if isinstance(r.도로명, str) else None, f"{bus.iloc[k].시설명} 정류소 {d:.0f}m" if d <= 250 else None]
    return " · ".join(p for p in parts if p) or f"{r.동 if isinstance(r.동, str) else ''} 무명도로 지점".strip()
