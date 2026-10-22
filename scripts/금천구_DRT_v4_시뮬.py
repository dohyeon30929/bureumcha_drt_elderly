"""DRT v4 시뮬레이션: v3 엔진에 가정 설정(cfg)을 붙여 v4 가정과 가정별 영향(하나씩 바꾸기)을 함께 계산.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_v4_시뮬.py`  (선행: 금천구_DRT_v4_준비.py)
v4 가정 (v3 대비 변경)
  - 최대 승차시간: min(직행 주행시간 + 15분, 40분)  (v3: min(직행 x 2, 30분))
  - 가는 호출 접수 8~17시(O-D '가는 방향' 8~16시 분포 사용), 귀가는 끝까지 처리  (v3: 8~19시)
  - 2대 권역: A = 시흥3·5동 8곳, B = 시흥4동 4곳 + B01, 대기 장소 권역별 재선정  (v3: A = 시흥3동 5곳, B = 나머지 8곳)
  - 고정노선 기본안: B01 제외 순환(11.61km, 51.8분), B01 이용자는 미서비스로 별도 집계. B01 포함 순환(18.38km, 73.2분)은 참고
그대로인 가정: 20km/h, 정원 12, 승하차 1인당 1분, 최대 대기 30분, 이용률 1/3/5% 하루 1회 왕복, 귀가 U(1,3)시간 후,
  목적지 비중·삽입 휴리스틱·권역 고정 배정·빈 차량은 마지막 하차 지점 대기, 시나리오별 300회(시드 20261011)
추가 지표: 차량 평균 탑승 인원 = (주행 중 탑승 인원 x 주행시간) 합 / 주행시간 합, 승차시간 / 직행시간
보정·탄소용 지표(금천구_DRT_v4_보정.py가 사용, 기존 결과에는 영향 없음): 빈 차/탑승 km, 인-km, 거절 편도 직행시간, 시간대별 호출·거절,
  차량 투입·철수 시각(shifts), 고정노선 시간표 기반 탑승 km·구간 최대 탑승
체크포인트: 시나리오 하나가 끝날 때마다 data/interim/DRT_v4_runs/ 에 저장, 재실행 시 완료분 건너뜀
"""
import heapq
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd

INTERIM, OUT = "data/interim", "outputs"
POP70, CAP, SVC, MAXWAIT, OPEN = 8817, 12, 1.0, 30.0, 480.0
NREP, SEED = 300, 20261011
RATES = [0.01, 0.03, 0.05]
SCAN = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
FLEET = range(3, 13)
fz = lambda d: tuple(sorted(d.items()))  # 설정을 사전 키로 쓰기 위해 튜플로
V3 = fz({"ride": "v3", "close": 1140.0, "zones": "v3"})
V4 = fz({"ride": "v4", "close": 1020.0, "zones": "v4"})
D = None


def init():
    global D, T, M, BI, DI, P, W, ZONE, DEPS, LOOPRIDE, LOOPDIST, LOOPGEO, B01, HOURS, ST
    D = pickle.load(open(f"{INTERIM}/DRT_v4_입력.pkl", "rb"))
    ST = D["stops"]
    T, M = D["T"].tolist(), D["M"].tolist()
    BI, DI, P = D["BI"], D["DI"], D["P"]
    W = ST.loc[BI, "배정_70세이상_명"].to_numpy(float)
    W = W / W.sum()
    I = D["I"]
    ZONE = {"v3": {i: ST.권역[i] for i in BI}, "v4": D["ZONE4"]}
    DEPS = {"single": I["D2"], "v3": {"A": I["DEP_A"], "B": I["D2"]}, "v4": {"A": I["DEP4_A"], "B": I["DEP4_B"]}}
    B01 = I["B01"]
    HOURS = D["go_hour"]  # 8~18시
    LOOPRIDE, LOOPDIST, LOOPGEO = {}, {}, {}
    for key, loop in [("B01", D["loop"]), ("noB01", D["loop0"])]:
        n = len(loop)
        pos = {s: k for k, s in enumerate(loop)}
        seg = [T[loop[k]][loop[(k + 1) % n]] for k in range(n)]
        segm = [M[loop[k]][loop[(k + 1) % n]] for k in range(n)]
        LOOPGEO[key] = (pos, segm, np.r_[0.0, np.cumsum([s + SVC for s in seg])[:-1]])  # 정류소 위치, 구간 거리(m), 순환 시작부터 도착까지 시각
        for o in loop:
            for d in loop:
                if o != d:
                    k, t, m = pos[o], 0.0, 0.0
                    while loop[(k + 1) % n] != d:
                        t += seg[k] + SVC
                        m += segm[k]
                        k = (k + 1) % n
                    LOOPRIDE[key, o, d] = t + seg[k]
                    LOOPDIST[key, o, d] = m + segm[k]


def maxride(cfg, o, d):
    return min(2 * T[o][d], 30.0) if cfg["ride"] == "v3" else min(T[o][d] + 15.0, 40.0)


def users(rate, rep, cfg):
    cfg = dict(cfg)
    rng = np.random.default_rng([SEED, int(rate * 10000), rep])
    n = int(round(POP70 * rate))
    h = HOURS[HOURS.index < cfg["close"] / 60]
    o = rng.choice(BI, size=n, p=W)
    d = np.array([DI[rng.choice(5, p=P[BI.index(x)])] for x in o])
    hr = rng.choice(h.index.to_numpy(), size=n, p=(h / h.sum()).to_numpy())
    t = hr * 60 + rng.uniform(0, 60, n)
    stay = rng.uniform(60, 180, n)
    return sorted((float(t[k]), int(o[k]), int(d[k]), float(stay[k])) for k in range(n))


def users_n(n, rep, b01=False, close=1020.0, hours=None):
    """하루 이용자 n명 표본(반납자 시나리오용, 이용률 대신 인원으로 시드). b01=False면 B01 배정 인원을 빼고 나머지 12곳 비중으로 추첨
    따라서 DRT와 고정노선(B01 제외)에 같은 승객·같은 목적지를 넣을 수 있다."""
    rng = np.random.default_rng([SEED, 7001, n, rep, int(b01)])
    w = W.copy()
    if not b01:
        w[BI.index(B01)] = 0.0
        w = w / w.sum()
    h = (HOURS if hours is None else hours)  # hours: 가는 호출 시간대 분포(민감도용), 기본 = O-D 8/31 분포
    h = h[h.index < close / 60]
    o = rng.choice(BI, size=n, p=w)
    d = np.array([DI[rng.choice(5, p=P[BI.index(x)])] for x in o])
    hr = rng.choice(h.index.to_numpy(), size=n, p=(h / h.sum()).to_numpy())
    t = hr * 60 + rng.uniform(0, 60, n)
    stay = rng.uniform(60, 180, n)
    return sorted((float(t[k]), int(o[k]), int(d[k]), float(stay[k])) for k in range(n))


def run_drt(us, nveh, cfg, shifts=None, pool=False, maxwait=MAXWAIT, detail=False, idle_dep=None):
    """idle_dep: 정류소 번호를 주면 계획이 빈 차량이 그 지점(차고지)으로 돌아가 대기(민감도). None = 기본(직전 하차 위치에서 대기)
    detail=True면 편도별 기록(legs: 이용자 번호, 편도, 출발, 도착, 상태, 대기, 차내)을 함께 반환한다(기본 결과에는 영향 없음).
    shifts: 차량별 (투입 시각, 철수 시각) 목록(분). None이면 전 차량 종일. 철수 시각 뒤에 끝나는 계획은 받지 않는다.
    pool=True면 2대도 공동 운영(권역 구분 없음). 기본은 1대 단일, 2대 권역 고정, 3대 이상 공동 운영"""
    cfg = dict(cfg)
    zmap = ZONE[cfg["zones"]]
    if nveh == 1:
        vz = [("single", DEPS["single"])]
    elif nveh == 2 and not pool:
        vz = [(z, DEPS[cfg["zones"]][z]) for z in "AB"]
    else:  # 공동 운영: 권역 A·B 대기 장소 교대
        vz = [("pool", DEPS["v4"]["A" if k % 2 else "B"]) for k in range(nveh)]
    sh = shifts or [(OPEN, 1e9)] * nveh
    veh = [{"z": z, "node": dp, "t": on, "plan": [], "load": 0, "dist": 0.0, "busy": 0.0, "dep": dp, "ln": 0.0, "mv": 0.0,
            "on": on, "off": off, "ke": 0.0, "kl": 0.0, "pkm": 0.0} for (z, dp), (on, off) in zip(vz, sh)]
    req, heap = [], []
    for k, (t, o, d, stay) in enumerate(us):
        heapq.heappush(heap, (t, len(req)))
        req.append({"t": t, "o": o, "d": d, "stay": stay, "leg": "가는", "mr": maxride(cfg, o, d), "z": zmap[o], "st": "대기", "u": k})

    def advance(v, t, idle=True):
        while v["plan"] and v["t"] <= t:
            n_, kind, rid = v["plan"].pop(0)
            tt, mm = T[v["node"]][n_], M[v["node"]][n_]
            v["dist"] += mm
            v["kl" if v["load"] else "ke"] += mm  # 탑승 승객 있음/빈 차 주행거리
            v["pkm"] += v["load"] * mm
            v["busy"] += tt + SVC
            v["ln"] += v["load"] * tt
            v["mv"] += tt
            if v["load"] > 10:  # 진단: 10명 초과로 달린 시간
                v["t10"] = v.get("t10", 0.0) + tt
            a = v["t"] + tt
            if kind == 0:  # 차고지 복귀(빈 차 이동, 정차 시간 없음)
                v["node"], v["t"] = n_, a
                continue
            r = req[rid]
            if kind == 1:
                r["ap"], r["sp"] = a, a + SVC
                v["load"] += 1
                v["maxon"] = max(v.get("maxon", 0), v["load"])  # 진단: 동시 탑승 최대 인원
            else:
                r["ad"], r["st"] = a, "완료"
                v["load"] -= 1
                if r["leg"] == "가는":
                    q = {"t": a + r["stay"], "o": r["d"], "d": r["o"], "stay": 0, "leg": "귀가", "mr": maxride(cfg, r["d"], r["o"]),
                         "z": r["z"], "st": "대기", "u": r["u"]}
                    heapq.heappush(heap, (q["t"], len(req)))
                    req.append(q)
            v["node"], v["t"] = n_, a + SVC
            if idle_dep is not None and not v["plan"] and v["node"] != idle_dep:
                v["plan"].append((idle_dep, 0, -1))
        if idle and not v["plan"] and v["t"] < t:
            v["t"] = t

    def evaluate(v, plan):
        node, t, load, sp = v["node"], v["t"], v["load"], {}
        for n_, kind, rid in plan:
            a = t + T[node][n_]
            if kind == 0:
                node, t = n_, a
                continue
            r = req[rid]
            if kind == 1:
                if a - r["t"] > maxwait:
                    return None
                load += 1
                if load > CAP:
                    return None
                sp[rid] = a + SVC
            else:
                if a - sp.get(rid, r.get("sp", a)) > r["mr"]:
                    return None
                load -= 1
            node, t = n_, a + SVC
        return t

    while True:
        if not heap:
            # 호출이 다 끝나도 차량에 남은 '가는' 하차가 귀가 호출을 만들 수 있음 -> 다음 이벤트까지 진행 후 다시 확인
            busy = [v for v in veh if v["plan"]]
            if not busy:
                break
            tn = min(v["t"] for v in busy)
            for v in veh:
                advance(v, tn)
            continue
        t, rid = heapq.heappop(heap)
        r = req[rid]
        if r["leg"] == "가는" and t >= cfg["close"]:
            r["st"] = "접수마감"
            continue
        for v in veh:
            advance(v, t)
        cands = [v for v in veh if v["z"] == r["z"]] if (nveh == 2 and not pool) else veh
        best = None
        for v in cands:
            if v["on"] > t:  # 아직 투입 전
                continue
            base = evaluate(v, v["plan"]) if v["plan"] else v["t"]
            pl = v["plan"]
            Pk, Dk = (r["o"], 1, rid), (r["d"], -1, rid)
            for i in range(len(pl) + 1):
                for j in range(i, len(pl) + 1):
                    new = pl[:i] + [Pk] + pl[i:j] + [Dk] + pl[j:]
                    e = evaluate(v, new)
                    if e is not None and e <= v["off"] and (best is None or e - base < best[0]):
                        best = (e - base, v, new)
        if best is None:
            r["st"] = "거절"
        else:
            bv = best[1]
            if all(k_ == 0 for _, k_, _ in bv["plan"]):  # 빈 차(승객 계획 없음)가 받은 호출: 차량 위치 -> 승차 정류소 주행시간
                r["idle_app"] = T[bv["node"]][r["o"]]
            bv["plan"] = best[2]
    end = []
    for v in veh:
        advance(v, 1e9, idle=False)
        v["dist"] += M[v["node"]][v["dep"]]
        v["ke"] += M[v["node"]][v["dep"]]
        v["t"] += T[v["node"]][v["dep"]]
        end.append(v["t"])
    pax = [r for r in req if r["st"] == "완료"]
    rej = [r for r in req if r["st"] == "거절"]
    hreq = [r for r in req if r["st"] != "접수마감"]
    out = {"요청": sum(r["st"] != "접수마감" for r in req), "거절": sum(r["st"] == "거절" for r in req),
            "wait": [r["ap"] - r["t"] for r in pax], "ride": [r["ad"] - r["sp"] for r in pax], "dir": [T[r["o"]][r["d"]] for r in pax],
            # 보정용: 거절 호출의 직행시간, 가는 호출이 거절돼 생기지 않은 귀가 편도의 직행시간
            "rej_dir": [T[r["o"]][r["d"]] for r in rej], "rej_ret_dir": [T[r["d"]][r["o"]] for r in rej if r["leg"] == "가는"], "n_user": len(us),
            "km_empty": sum(v["ke"] for v in veh) / 1000, "km_loaded": sum(v["kl"] for v in veh) / 1000, "pkm": sum(v["pkm"] for v in veh) / 1000,
            "hour_req": np.bincount([int(r["t"] // 60) for r in hreq], minlength=24).tolist(),
            "hour_rej": np.bincount([int(r["t"] // 60) for r in rej], minlength=24).tolist(),
            "km": sum(v["dist"] for v in veh) / 1000, "util": [v["busy"] / max(1e-9, v["t"] - v["on"]) for v in veh], "end": max(end),
            "occ": sum(v["ln"] for v in veh) / max(1e-9, sum(v["mv"] for v in veh)),
            "zone_req": {z: sum(r["z"] == z and r["st"] != "접수마감" for r in req) for z in "AB"},
            "zone_rej": {z: sum(r["z"] == z and r["st"] == "거절" for r in req) for z in "AB"},
            "zone_util": {v["z"]: v["busy"] / (v["t"] - OPEN) for v in veh} if (nveh == 2 and not pool) else {},
            "b01": sum(r["o"] == B01 or r["d"] == B01 for r in req), "미서비스": 0}
    out["max_onboard"] = max(v.get("maxon", 0) for v in veh)  # 진단 기록(결과 계산에는 쓰지 않음)
    out["t_over10"] = sum(v.get("t10", 0.0) for v in veh)
    out["t_move"] = sum(v["mv"] for v in veh)
    out["idle_app"] = [r["idle_app"] for r in req if "idle_app" in r]
    out["wait_idle"] = [r["ap"] - r["t"] for r in pax if "idle_app" in r]  # 빈 차가 받은 호출의 대기
    out["wait_busy"] = [r["ap"] - r["t"] for r in pax if "idle_app" not in r]  # 승객을 태운(계획이 있는) 차량에 끼워 넣은 호출의 대기
    out["wait_leg"] = {lg: [r["ap"] - r["t"] for r in pax if r["leg"] == lg] for lg in ("가는", "귀가")}
    if detail:
        out["legs"] = [(r["u"], r["leg"], r["o"], r["d"], r["st"], r.get("ap", np.nan) - r["t"], r.get("ad", np.nan) - r.get("sp", np.nan))
                       for r in req]
    return out


def run_fixed(us, nveh, loopkey):
    cyc = D["cycle"] if loopkey == "B01" else D["cycle0"]
    lm = D["loop_m"] if loopkey == "B01" else D["loop0_m"]
    H = cyc / nveh
    pos, segm, toff = LOOPGEO[loopkey]
    n = len(pos)
    wait, ride, dr, last, miss, miss_dir, pkm, used = [], [], [], 0.0, 0, [], 0.0, {}

    def board(t, o, d):
        """탄소 지표용 시간표: 차량 j는 OPEN + j*H에 순환 첫 정류소 출발. t 이후 o에 처음 오는 차량에 타서 d까지 지나는 구간 기록"""
        k = pos[o]
        best = None
        for j in range(nveh):
            m = max(0, int(np.ceil((t - OPEN - j * H - toff[k]) / cyc)))
            a = OPEN + j * H + m * cyc + toff[k]
            if best is None or a < best[0]:
                best = (a, j, m)
        _, j, m = best
        while True:
            used[j, m, k] = used.get((j, m, k), 0) + 1
            k += 1
            if k == n:
                k, m = 0, m + 1
            if k == pos[d]:
                break

    for t, o, d, stay in us:
        if (loopkey, o, d) not in LOOPRIDE:  # B01 제외 순환에서 B01 이용자
            miss += 2
            miss_dir += [T[o][d], T[d][o]]
            continue
        r1 = LOOPRIDE[loopkey, o, d]
        bt = t + H / 2 + r1 + stay
        r2 = LOOPRIDE[loopkey, d, o]
        wait += [H / 2, H / 2]
        ride += [r1, r2]
        dr += [T[o][d], T[d][o]]
        pkm += (LOOPDIST[loopkey, o, d] + LOOPDIST[loopkey, d, o]) / 1000
        board(t, o, d)
        board(bt, d, o)
        last = max(last, bt + H / 2 + r2)
    km = nveh * lm / 1000 * (last - OPEN) / cyc
    kl = min(km, sum(segm[k] for (_, _, k) in used) / 1000)
    return {"요청": 2 * len(us) - miss, "거절": 0, "wait": wait, "ride": ride, "dir": dr, "km": km,
            "util": [1.0] * nveh, "end": last, "occ": np.nan, "zone_req": {}, "zone_rej": {}, "zone_util": {},
            "b01": 0, "미서비스": miss, "miss_dir": miss_dir, "rej_dir": [], "rej_ret_dir": [], "n_user": len(us),
            "km_empty": km - kl, "km_loaded": kl, "pkm": pkm, "maxload": max(used.values(), default=0)}


def run_fixed_cap(us, nveh, loopkey, cap=CAP):
    """정원 반영 고정노선: 시간표대로 도는 차량이 정류소마다 하차 후 대기 승객을 도착 순서대로 태움(정원까지).
    만차면 남은 승객은 다음 차량을 기다림(대기시간에 반영). 대기 = 실제 승차 시각 - 도착 시각(배차간격/2 가정 아님).
    귀가 호출 = 목적지 하차 시각 + 체류시간. 차량 j는 OPEN + j x 배차간격에 순환 첫 정류소 출발, 정류소마다 SVC분 정차"""
    import bisect
    cyc = D["cycle"] if loopkey == "B01" else D["cycle0"]
    loop = D["loop"] if loopkey == "B01" else D["loop0"]
    pos, segm, _ = LOOPGEO[loopkey]
    n = len(loop)
    segt = [T[loop[k]][loop[(k + 1) % n]] for k in range(n)]
    H = cyc / nveh
    q = {k: [] for k in range(n)}  # 정류소별 대기 [도착 시각, 승객번호, 편도(0 가는/1 귀가), 목적 위치, 못 탄 횟수]
    miss_dir, pend = [], 0
    for u, (t, o, d, stay) in enumerate(us):
        if (loopkey, o, d) not in LOOPRIDE:
            miss_dir += [T[o][d], T[d][o]]
            continue
        q[pos[o]].append([t, u, 0, pos[d], 0])
        pend += 2
    for k in q:
        q[k].sort()
    on = [[] for _ in range(nveh)]  # 차량별 탑승 [목적 위치, 승객번호, 편도, 승차 시각, 도착 시각, 못 탄 횟수]
    ev = [(OPEN + j * H, j, 0) for j in range(nveh)]
    heapq.heapify(ev)
    wait, ride, dr, skips, ke, kl, pkm, last, maxload = [], [], [], [], 0.0, 0.0, 0.0, OPEN, 0
    while pend:
        tau, j, k = heapq.heappop(ev)
        keep = []
        for p in on[j]:
            if p[0] == k:
                dk, u, leg, tb, ta, sk = p
                wait.append(tb - ta)
                ride.append(tau - (tb + SVC))
                o_, d_ = (us[u][1], us[u][2]) if leg == 0 else (us[u][2], us[u][1])
                dr.append(T[o_][d_])
                skips.append(sk)
                pend -= 1
                last = max(last, tau)
                if leg == 0:
                    bisect.insort(q[k], [tau + us[u][3], u, 1, pos[us[u][1]], 0])
            else:
                keep.append(p)
        on[j] = keep
        wl = q[k]
        i = 0
        while i < len(wl) and wl[i][0] <= tau:
            if len(on[j]) < cap:
                ta, u, leg, dk, sk = wl.pop(i)
                on[j].append([dk, u, leg, tau, ta, sk])
            else:
                wl[i][4] += 1  # 이 차량 만차로 못 탐
                i += 1
        load = len(on[j])
        maxload = max(maxload, load)
        m = segm[k]
        if load:
            kl += m
        else:
            ke += m
        pkm += load * m
        heapq.heappush(ev, (tau + SVC + segt[k], j, (k + 1) % n))
    sk = np.array(skips)
    km = (ke + kl) / 1000
    return {"요청": len(wait) + len(miss_dir), "거절": 0, "wait": wait, "ride": ride, "dir": dr, "km": km, "util": [1.0] * nveh, "end": last,
            "occ": np.nan, "미서비스": len(miss_dir), "miss_dir": miss_dir, "rej_dir": [], "rej_ret_dir": [], "n_user": len(us),
            "km_empty": ke / 1000, "km_loaded": kl / 1000, "pkm": pkm / 1000, "maxload": maxload,
            "left1": int((sk >= 1).sum()), "left2": int((sk >= 2).sum()), "b01": 0, "zone_req": {}, "zone_rej": {}, "zone_util": {},
            "hour_req": [0] * 24, "hour_rej": [0] * 24}


def task(args):
    rate, rep, mode, nveh, cfg, loopkey = args
    us = users(rate, rep, cfg)
    return run_drt(us, nveh, cfg) if mode == "DRT" else run_fixed(us, nveh, loopkey)


def summarize(rs, rate):
    w, rd, dr = (np.concatenate([r[k] for r in rs]) for k in ("wait", "ride", "dir"))
    tot = w + rd
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    return {"이용자_명": int(round(POP70 * rate)), "호출_건": req / len(rs), "총시간_평균": tot.mean(), "총시간_p95": np.percentile(tot, 95),
            "대기_평균": w.mean(), "대기_p95": np.percentile(w, 95), "승차_평균": rd.mean(), "승차_p95": np.percentile(rd, 95),
            "직행_평균": dr.mean(), "승차/직행": rd.mean() / dr.mean(), "거절률_%": rej / req * 100,
            "미서비스_건": np.mean([r["미서비스"] for r in rs]), "평균탑승인원": np.nanmean([r["occ"] for r in rs]),
            "운행거리_km": np.mean([r["km"] for r in rs]), "가동률_%": np.mean([np.mean(r["util"]) for r in rs]) * 100,
            "종료_평균": np.mean([r["end"] for r in rs]), "종료_최대": np.max([r["end"] for r in rs])}


hm = lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"


def cfg_name(c):
    c = dict(c)
    return f"ride{c['ride']}_close{int(c['close'] // 60)}_zone{c['zones']}"


if __name__ == "__main__":
    init()
    RUNS = f"{INTERIM}/DRT_v4_runs"
    os.makedirs(RUNS, exist_ok=True)
    one = lambda **kw: fz({**dict(V3), **kw})
    scen = []  # (key, rate, mode, nveh, cfg, loopkey)
    for rate in RATES:
        scen += [("v4", rate, "DRT", 1, V4, None), ("v4", rate, "DRT", 2, V4, None),
                 ("v4", rate, "고정", 1, V4, "noB01"), ("v4", rate, "고정", 2, V4, "noB01"),
                 ("v4참고", rate, "고정", 1, V4, "B01"), ("v4참고", rate, "고정", 2, V4, "B01"),
                 ("v3재현", rate, "DRT", 1, V3, None), ("v3재현", rate, "DRT", 2, V3, None),
                 ("승차완화만", rate, "DRT", 1, one(ride="v4"), None), ("승차완화만", rate, "DRT", 2, one(ride="v4"), None),
                 ("17시마감만", rate, "DRT", 1, one(close=1020.0), None), ("17시마감만", rate, "DRT", 2, one(close=1020.0), None),
                 ("권역재구성만", rate, "DRT", 2, one(zones="v4"), None)]
    scen += [("v4", r, "DRT", nv, V4, None) for r in SCAN if r not in RATES for nv in (1, 2)]
    scen += [("v4", r, "DRT", nv, V4, None) for r in RATES for nv in FLEET]
    fname = lambda s: f"{RUNS}/{s[0]}_r{s[1] * 1000:05.1f}_{s[2]}{s[3]}대_{cfg_name(s[4])}_{s[5] or ''}.pkl"
    todo = [s for s in scen if not os.path.exists(fname(s))]
    print(f"시나리오 {len(scen)}개 중 완료 {len(scen) - len(todo)}개, 남은 {len(todo)}개 (시나리오당 {NREP}회)", flush=True)
    t0 = time.time()
    with mp.Pool(11, initializer=init) as pool:
        for n_, s in enumerate(todo, 1):
            t1 = time.time()
            rs = pool.map(task, [(s[1], rp, s[2], s[3], s[4], s[5]) for rp in range(NREP)], chunksize=5)
            pickle.dump(rs, open(fname(s) + ".tmp", "wb"))
            os.replace(fname(s) + ".tmp", fname(s))
            print(f"[{n_}/{len(todo)}] {s[0]} {s[1]:.1%} {s[2]} {s[3]}대 {s[5] or ''}: {time.time() - t1:.0f}초 (누적 {(time.time() - t0) / 60:.1f}분)", flush=True)
    R = {s: pickle.load(open(fname(s), "rb")) for s in scen}
    S = {s: summarize(R[s], s[1]) for s in scen}
    pd.set_option("display.width", 260)

    # ---- 1. v3 승차시간 진단
    rows = []
    for rate in RATES:
        for nv in (1, 2):
            a = S[("v3재현", rate, "DRT", nv, V3, None)]
            rows.append({"이용률_%": rate * 100, "안": f"DRT {nv}대", "직행_평균": a["직행_평균"], "승차_평균": a["승차_평균"],
                         "승차-직행_분": a["승차_평균"] - a["직행_평균"], "승차/직행": a["승차/직행"], "평균탑승인원": a["평균탑승인원"]})
    diag = pd.DataFrame(rows)
    diag.round(2).to_csv(f"{OUT}/DRT_v4_v3승차시간진단.csv", index=False, encoding="utf-8-sig")
    print("\n[1] v3 가정(최대 승차 min(직행x2, 30분))에서 승차시간 vs 직행")
    print(diag.round(2).to_string(index=False))

    # ---- 본 결과 v4
    def row(s, label):
        a = S[s]
        return {"이용률_%": s[1] * 100, "안": label, **a}
    main = []
    for rate in RATES:
        main += [row(("v4", rate, "DRT", 1, V4, None), "DRT 1대"), row(("v4", rate, "DRT", 2, V4, None), "DRT 2대(새 권역)"),
                 row(("v4", rate, "고정", 1, V4, "noB01"), "고정노선 1대(B01 제외)"), row(("v4", rate, "고정", 2, V4, "noB01"), "고정노선 2대(B01 제외)"),
                 row(("v4참고", rate, "고정", 1, V4, "B01"), "참고: 고정노선 1대(B01 포함)"), row(("v4참고", rate, "고정", 2, V4, "B01"), "참고: 고정노선 2대(B01 포함)")]
    main = pd.DataFrame(main)
    out = main.assign(종료_평균=main.종료_평균.map(hm), 종료_최대=main.종료_최대.map(hm))
    out.round(2).to_csv(f"{OUT}/DRT_v4_시뮬레이션_요약.csv", index=False, encoding="utf-8-sig")
    print("\n[v4 결과] 승객 총시간 기준 (분)")
    print(out[["이용률_%", "안", "이용자_명", "호출_건", "총시간_평균", "총시간_p95", "대기_평균", "승차_평균", "직행_평균", "거절률_%", "미서비스_건",
               "평균탑승인원", "운행거리_km", "가동률_%", "종료_평균", "종료_최대"]].round(1).to_string(index=False))

    # ---- 가정별 영향 (v3 재현 -> 하나씩 -> v4)
    eff = []
    steps = [("v3재현", V3, "v3 재현"), ("승차완화만", one(ride="v4"), "+ 최대 승차 완화만"), ("17시마감만", one(close=1020.0), "+ 17시 마감만"),
             ("권역재구성만", one(zones="v4"), "+ 권역 재구성만"), ("v4", V4, "v4 (모두 적용)")]
    for rate in RATES:
        for nv in (1, 2):
            for key, c, lab in steps:
                s = (key, rate, "DRT", nv, c, None)
                if s not in S:
                    continue
                a = S[s]
                eff.append({"이용률_%": rate * 100, "안": f"DRT {nv}대", "가정": lab, "거절률_%": a["거절률_%"], "대기_평균": a["대기_평균"],
                            "승차_평균": a["승차_평균"], "총시간_평균": a["총시간_평균"], "평균탑승인원": a["평균탑승인원"],
                            "운행거리_km": a["운행거리_km"], "종료_평균": hm(a["종료_평균"]), "종료_최대": hm(a["종료_최대"])})
    eff = pd.DataFrame(eff)
    eff.round(2).to_csv(f"{OUT}/DRT_v4_가정별영향.csv", index=False, encoding="utf-8-sig")
    print("\n[가정별 영향] DRT")
    print(eff.round(1).to_string(index=False))

    # ---- 전환 기준
    sc = []
    for r in sorted(set(SCAN) | set(RATES)):
        for nv in (1, 2):
            a = S[("v4", r, "DRT", nv, V4, None)]
            sc.append({"이용률_%": r * 100, "이용자_명": a["이용자_명"], "안": f"DRT {nv}대", "대기_평균": a["대기_평균"], "거절률_%": a["거절률_%"],
                       "총시간_평균": a["총시간_평균"], "평균탑승인원": a["평균탑승인원"]})
    sc = pd.DataFrame(sc)
    sc.round(2).to_csv(f"{OUT}/DRT_v4_전환기준_스캔.csv", index=False, encoding="utf-8-sig")
    print("\n[전환 기준 스캔 v4]")
    print(sc.pivot(index=["이용률_%", "이용자_명"], columns="안", values=["대기_평균", "거절률_%"]).round(1).to_string())
    for nv in (1, 2):
        g = sc[sc.안 == f"DRT {nv}대"].sort_values("이용률_%")
        bad = g[(g.대기_평균 > 15) | (g["거절률_%"] > 5)]
        okg = g[g["이용률_%"] < bad.iloc[0]["이용률_%"]] if len(bad) else g
        print(f"  DRT {nv}대 한계 이용자(평균 대기 15분·거절률 5% 이하를 지키는 마지막 지점):",
              f"{okg.iloc[-1]['이용률_%']:.1f}% (하루 {okg.iloc[-1].이용자_명}명), 대기 {okg.iloc[-1].대기_평균:.1f}분, 거절 {okg.iloc[-1]['거절률_%']:.1f}%" if len(okg) else "없음",
              f"/ 다음 지점 {bad.iloc[0].이용자_명}명에서 초과" if len(bad) else "")

    # ---- 필요 차량 수 (공동 운영)
    fl = []
    for r in RATES:
        for nv in [1, 2, *FLEET]:
            a = S[("v4", r, "DRT", nv, V4, None)]
            fl.append({"이용률_%": r * 100, "차량": nv, "운영": "단일" if nv == 1 else "권역 고정" if nv == 2 else "공동 운영", "거절률_%": a["거절률_%"],
                       "대기_평균": a["대기_평균"], "총시간_평균": a["총시간_평균"], "평균탑승인원": a["평균탑승인원"], "운행거리_km": a["운행거리_km"]})
    fl = pd.DataFrame(fl)
    fl.round(2).to_csv(f"{OUT}/DRT_v4_필요차량수.csv", index=False, encoding="utf-8-sig")
    print("\n[필요 차량 수 v4]")
    print(fl.pivot(index="차량", columns="이용률_%", values=["거절률_%", "대기_평균"]).round(1).to_string())
    v3fl = pd.read_csv(f"{OUT}/DRT_v3_필요차량수.csv", encoding="utf-8-sig")
    for r in RATES:
        need = lambda t: (lambda g: int(g.차량.min()) if len(g) else "12대 초과")(t[(t["이용률_%"] == r * 100) & (t["거절률_%"] <= 5) & (t.대기_평균 <= 15)])
        print(f"  {r:.0%}: 거절률 5% 이하·대기 15분 이하 최소 차량 v3 {need(v3fl)} -> v4 {need(fl)}")

    # ---- 권역 부하
    popz = {v: {z: ST.loc[[i for i in BI if ZONE[v][i] == z], "배정_70세이상_명"].sum() for z in "AB"} for v in ("v3", "v4")}
    zr = []
    for r in RATES:
        for v, key, c in [("v3", "v3재현", V3), ("v4", "v4", V4)]:
            rs = R[(key, r, "DRT", 2, c, None)]
            for z in "AB":
                zr.append({"이용률_%": r * 100, "권역안": v, "권역": z, "승차정류소": sum(ZONE[v][i] == z for i in BI), "배정_70세이상_명": popz[v][z],
                           "하루호출_건": np.mean([x["zone_req"][z] for x in rs]),
                           "거절률_%": np.sum([x["zone_rej"][z] for x in rs]) / max(1, np.sum([x["zone_req"][z] for x in rs])) * 100,
                           "가동률_%": np.mean([x["zone_util"][z] for x in rs]) * 100})
    zr = pd.DataFrame(zr)
    zr.round(1).to_csv(f"{OUT}/DRT_v4_권역부하.csv", index=False, encoding="utf-8-sig")
    print("\n[권역 부하] v3 권역 vs v4 권역 (DRT 2대)")
    print(zr.round(1).to_string(index=False))
    pickle.dump({"main": main, "eff": eff, "sc": sc, "fl": fl, "zr": zr, "diag": diag}, open(f"{INTERIM}/DRT_v4_결과.pkl", "wb"))
