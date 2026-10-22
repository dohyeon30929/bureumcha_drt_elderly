"""DRT v3 시뮬레이션: 실시간 호출 + 삽입 휴리스틱(DARP 개념), 1대/2대(권역 고정), 고정노선 1대/2대 비교.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_v3_시뮬.py`  (선행: 금천구_DRT_v3_준비.py)
가정(보고서 명시)
  - 새 호출(가는 호출) 접수 8~19시, 이미 생긴 귀가 호출은 19시 이후에도 끝까지 처리. 평균 20km/h, 정원 12인, 승하차 1인당 1분
  - 최대 대기 30분(호출 시각 -> 승차 시작), 최대 승차시간 = min(직행 주행시간 x 2, 30분)(승차 완료 -> 하차 시작). 못 지키면 거절
  - 이용률: 70세 이상 8,817명의 1/3/5%가 하루 1회 왕복. 승차 정류소는 배정 고령인구 비례, 목적지는 DRT_v3_목적지추첨비중.csv
  - 가는 호출 시각: O-D 60세 이상 '가는 방향'(3·4·5동->시흥1동) 8~18시 시간대 분포 + 시간 안 균등
  - 귀가 호출: 목적지 도착 후 U(60, 180)분 뒤, 처음 승차 정류소로. 가는 호출이 거절되면 귀가 호출도 없다.
배차
  - 호출 시점에 차량의 남은 계획(이미 출발한 다음 정류소는 확정) 사이에 승차·하차를 끼워 넣는 모든 위치를 검토,
    승차가 하차보다 먼저, 정원·대기·승차시간 제한을 기존 승객까지 모두 만족하는 위치 중 계획 종료시각 증가가 가장 작은 곳
  - 2대: 권역 고정(권역 A 승차 정류소의 가는·귀가 호출은 A 차량, 나머지는 B 차량). 차량은 할 일이 없으면 마지막 하차 지점에서 대기,
    하루 시작은 대기 장소(단일·B 시흥사거리, A 시흥3동), 마지막 운행 후 대기 장소로 돌아간다(거리 포함).
고정노선: 승차 13 + 목적지 5 최단 순환(73.2분)을 계속 운행. 대기 = 배차간격/2(배차간격 = 순환시간/차량 수),
  승차시간 = 순환 경로를 따라 승차 정류소 -> 하차 정류소(중간 정류소 정차 1분 포함). 정원 미반영
"""
import heapq
import multiprocessing as mp
import pickle
import sys
import time

import numpy as np
import pandas as pd

INTERIM, OUT = "data/interim", "outputs"
POP70, CAP, SVC, MAXWAIT, OPEN, CLOSE = 8817, 12, 1.0, 30.0, 480.0, 1140.0
NREP, SEED = 300, 20261011
RATES = [0.01, 0.03, 0.05]
SCAN = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
FLEET = range(3, 13)  # 참고: 공동 운영 차량 수 스캔(권역 없이 추가 운행시간 최소 차량에 배정)
D = None


def init():
    global D, T, M, BI, DI, P, W, ZONEOF, DEP, LOOPRIDE, B01, H_GO
    D = pickle.load(open(f"{INTERIM}/DRT_v3_입력.pkl", "rb"))
    st = D["stops"]
    T, M = D["T"].tolist(), D["M"].tolist()
    BI, DI, P = D["BI"], D["DI"], D["P"]
    W = st.loc[BI, "배정_70세이상_명"].to_numpy(float)
    W = W / W.sum()
    ZONEOF = {i: st.권역[i] for i in BI}
    DEP = {"single": D["I"]["D2"], "A": D["I"]["DEP_A"], "B": D["I"]["D2"]}
    B01 = D["I"]["B01"]
    h = D["go_hour"]
    H_GO = (h.index.to_numpy(), (h / h.sum()).to_numpy())
    loop = D["loop"]
    n = len(loop)
    pos = {s: k for k, s in enumerate(loop)}
    seg = [T[loop[k]][loop[(k + 1) % n]] for k in range(n)]
    LOOPRIDE = {}
    for o in loop:
        for d in loop:
            if o == d:
                continue
            k, t = pos[o], 0.0
            while loop[(k + 1) % n] != d:
                t += seg[k] + SVC
                k = (k + 1) % n
            LOOPRIDE[o, d] = t + seg[k]


def users(rate, rep, drop_b01=False):
    rng = np.random.default_rng([SEED, int(rate * 10000), rep])
    n = int(round(POP70 * rate))
    o = rng.choice(BI, size=n, p=W)
    d = np.array([DI[rng.choice(5, p=P[BI.index(x)])] for x in o])
    hr = rng.choice(H_GO[0], size=n, p=H_GO[1])
    t = hr * 60 + rng.uniform(0, 60, n)
    stay = rng.uniform(60, 180, n)
    u = [(float(t[k]), int(o[k]), int(d[k]), float(stay[k])) for k in range(n)]
    if drop_b01:
        u = [x for x in u if x[1] != B01]
    return sorted(u)


# ---------------------------------------------------------------- DRT
def run_drt(us, nveh):
    veh = []
    zones = ["single"] if nveh == 1 else ["A", "B"] if nveh == 2 else ["pool"] * nveh
    for k, z in enumerate(zones):
        dep = DEP[z] if z in DEP else (DEP["B"] if k % 2 == 0 else DEP["A"])  # 공동 운영: 시흥사거리·시흥3동 대기 교대
        veh.append({"z": z, "node": dep, "t": OPEN, "plan": [], "load": 0, "dist": 0.0, "busy": 0.0, "dep": dep})
    req, heap = [], []
    for k, (t, o, d, stay) in enumerate(us):
        heapq.heappush(heap, (t, len(req)))
        req.append({"t": t, "o": o, "d": d, "stay": stay, "leg": "가는", "user": k, "mr": min(2 * T[o][d], 30.0),
                    "z": ZONEOF[o], "st": "대기"})

    def advance(v, t, idle=True):
        while v["plan"] and v["t"] <= t:
            n_, kind, rid = v["plan"].pop(0)
            tt = T[v["node"]][n_]
            v["dist"] += M[v["node"]][n_]
            v["busy"] += tt + SVC
            a = v["t"] + tt
            r = req[rid]
            if kind == 1:
                r["ap"], r["sp"] = a, a + SVC
                v["load"] += 1
            else:
                r["ad"] = a
                r["st"] = "완료"
                v["load"] -= 1
                if r["leg"] == "가는":  # 귀가 호출 생성
                    q = {"t": a + r["stay"], "o": r["d"], "d": r["o"], "stay": 0, "leg": "귀가", "user": r["user"],
                         "mr": min(2 * T[r["d"]][r["o"]], 30.0), "z": r["z"], "st": "대기"}
                    heapq.heappush(heap, (q["t"], len(req)))
                    req.append(q)
            v["node"], v["t"] = n_, a + SVC
        if idle and not v["plan"] and v["t"] < t:
            v["t"] = t

    def evaluate(v, plan):
        node, t, load, sp = v["node"], v["t"], v["load"], {}
        for n_, kind, rid in plan:
            a = t + T[node][n_]
            r = req[rid]
            if kind == 1:
                if a - r["t"] > MAXWAIT:
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

    while heap:
        t, rid = heapq.heappop(heap)
        r = req[rid]
        if r["leg"] == "가는" and t >= CLOSE:
            r["st"] = "접수마감"
            continue
        for v in veh:
            advance(v, t)
        cands = [v for v in veh if v["z"] == r["z"]] if nveh == 2 else veh
        best = None
        for v in cands:
            base = evaluate(v, v["plan"]) if v["plan"] else v["t"]
            pl = v["plan"]
            Pk, Dk = (r["o"], 1, rid), (r["d"], -1, rid)
            for i in range(len(pl) + 1):
                for j in range(i, len(pl) + 1):
                    new = pl[:i] + [Pk] + pl[i:j] + [Dk] + pl[j:]
                    e = evaluate(v, new)
                    if e is not None and (best is None or e - base < best[0]):
                        best = (e - base, v, new)
        if best is None:
            r["st"] = "거절"
        else:
            best[1]["plan"] = best[2]
    end = []
    for v in veh:
        advance(v, 1e9, idle=False)
        v["dist"] += M[v["node"]][v["dep"]]
        v["t"] += T[v["node"]][v["dep"]]
        end.append(v["t"])
    pax = [r for r in req if r["st"] == "완료"]
    out = {"요청": sum(r["st"] != "접수마감" for r in req), "거절": sum(r["st"] == "거절" for r in req),
           "거절_귀가": sum(r["st"] == "거절" and r["leg"] == "귀가" for r in req),
           "wait": [r["ap"] - r["t"] for r in pax], "ride": [r["ad"] - r["sp"] for r in pax],
           "km": sum(v["dist"] for v in veh) / 1000, "util": [v["busy"] / (v["t"] - OPEN) for v in veh], "end": max(end),
           "back_t": [r["t"] for r in req if r["leg"] == "귀가"], "b01": sum(r["o"] == B01 or r["d"] == B01 for r in req),
           "zone_req": {z: sum(r["z"] == z and r["st"] != "접수마감" for r in req) for z in "AB"},
           "zone_rej": {z: sum(r["z"] == z and r["st"] == "거절" for r in req) for z in "AB"},
           "zone_util": {v["z"]: v["busy"] / (v["t"] - OPEN) for v in veh} if nveh == 2 else {}}
    return out


# ---------------------------------------------------------------- 고정노선
def run_fixed(us, nveh):
    H = D["cycle"] / nveh
    wait, ride, back_t, last = [], [], [], CLOSE
    for t, o, d, stay in us:
        r1 = LOOPRIDE[o, d]
        arr = t + H / 2 + r1
        bt = arr + stay
        r2 = LOOPRIDE[d, o]
        wait += [H / 2, H / 2]
        ride += [r1, r2]
        back_t.append(bt)
        last = max(last, bt + H / 2 + r2)
    span = last - OPEN
    return {"요청": 2 * len(us), "거절": 0, "거절_귀가": 0, "wait": wait, "ride": ride, "km": nveh * D["loop_m"] / 1000 * span / D["cycle"],
            "util": [1.0] * nveh, "end": last, "back_t": back_t, "b01": sum(o == B01 or d == B01 for _, o, d, _ in us) * 2,
            "zone_req": {z: 2 * sum(ZONEOF[o] == z for _, o, _, _ in us) for z in "AB"}, "zone_rej": {"A": 0, "B": 0}, "zone_util": {}}


def task(args):
    rate, rep, mode, nveh, drop = args
    us = users(rate, rep, drop)
    r = run_drt(us, nveh) if mode == "DRT" else run_fixed(us, nveh)
    r.update({"rate": rate, "rep": rep, "안": f"{mode} {nveh}대", "B01제외": drop})
    return r


def summarize(rs):
    w = np.concatenate([r["wait"] for r in rs]) if rs else np.array([])
    rd = np.concatenate([r["ride"] for r in rs])
    tot = w + rd
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    return {"이용자_명": int(round(POP70 * rs[0]["rate"])), "호출_건": round(req / len(rs), 1),
            "대기_평균": w.mean(), "대기_p95": np.percentile(w, 95), "승차_평균": rd.mean(), "승차_p95": np.percentile(rd, 95),
            "총시간_평균": tot.mean(), "총시간_p95": np.percentile(tot, 95), "거절률_%": rej / req * 100,
            "귀가거절_비중_%": (sum(r["거절_귀가"] for r in rs) / rej * 100) if rej else 0.0,
            "운행거리_km": np.mean([r["km"] for r in rs]), "가동률_%": np.mean([np.mean(r["util"]) for r in rs]) * 100,
            "종료시각_평균": np.mean([r["end"] for r in rs]), "종료시각_최대": np.max([r["end"] for r in rs]),
            "B01_호출_건": np.mean([r["b01"] for r in rs])}


def hm(x):
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


if __name__ == "__main__":
    init()
    import os
    RUNS = f"{INTERIM}/DRT_v3_runs"
    os.makedirs(RUNS, exist_ok=True)
    scen = [(rate, mode, nv, False) for rate in RATES for mode, nv in [("DRT", 1), ("DRT", 2), ("고정", 1), ("고정", 2)]]
    scen += [(rate, "DRT", nv, True) for rate in RATES for nv in (1, 2)]
    scen += [(rate, "DRT", nv, False) for rate in SCAN if rate not in RATES for nv in (1, 2)]
    scen += [(rate, "DRT", nv, False) for rate in RATES for nv in FLEET]
    fname = lambda k: f"{RUNS}/r{k[0] * 1000:05.1f}_{k[1]}{k[2]}대_{'B01제외' if k[3] else '전체'}.pkl"
    todo = [k for k in scen if not os.path.exists(fname(k))]
    print(f"시나리오 {len(scen)}개 중 완료 {len(scen) - len(todo)}개, 남은 {len(todo)}개 (시나리오당 {NREP}회)", flush=True)
    t0 = time.time()
    with mp.Pool(11, initializer=init) as pool:
        for n_, k in enumerate(todo, 1):
            t1 = time.time()
            rs = pool.map(task, [(k[0], rep_, k[1], k[2], k[3]) for rep_ in range(NREP)], chunksize=5)
            tmp = fname(k) + ".tmp"
            pickle.dump(rs, open(tmp, "wb"))
            os.replace(tmp, fname(k))  # 다 쓴 뒤에만 이름을 바꿔, 중간에 꺼져도 깨진 파일이 남지 않게
            print(f"[{n_}/{len(todo)}] 이용률 {k[0]:.1%} {k[1]} {k[2]}대 {'B01제외' if k[3] else ''}: {time.time() - t1:.0f}초 "
                  f"(누적 {(time.time() - t0) / 60:.1f}분)", flush=True)
    df = {}
    for k in scen:
        rs = pickle.load(open(fname(k), "rb"))
        df[(k[0], f"{k[1]} {k[2]}대", k[3])] = rs

    # ---- 본 결과
    rows = []
    for rate in RATES:
        for an in ["DRT 1대", "DRT 2대", "고정 1대", "고정 2대"]:
            s = summarize(df[rate, an, False])
            rows.append({"이용률_%": rate * 100, "안": an.replace("고정", "고정노선"), **s})
    main = pd.DataFrame(rows)
    main["종료시각_평균"] = main.종료시각_평균.map(hm)
    main["종료시각_최대"] = main.종료시각_최대.map(hm)
    main.round(2).to_csv(f"{OUT}/DRT_v3_시뮬레이션_요약.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[결과] 승객 총시간 기준 (분)")
    print(main[["이용률_%", "안", "이용자_명", "호출_건", "총시간_평균", "총시간_p95", "대기_평균", "대기_p95", "승차_평균", "승차_p95", "거절률_%",
                "운행거리_km", "가동률_%", "종료시각_평균", "종료시각_최대"]].round(1).to_string(index=False))

    # ---- 전환 기준 스캔 (DRT 1대 / 2대)
    sc = []
    for rate in sorted(set(SCAN) | set(RATES)):
        for an in ["DRT 1대", "DRT 2대"]:
            s = summarize(df[rate, an, False])
            sc.append({"이용률_%": rate * 100, "안": an, "이용자_명": s["이용자_명"], "대기_평균": s["대기_평균"], "거절률_%": s["거절률_%"],
                       "총시간_평균": s["총시간_평균"], "운행거리_km": s["운행거리_km"], "가동률_%": s["가동률_%"]})
    sc = pd.DataFrame(sc)
    sc.round(2).to_csv(f"{OUT}/DRT_v3_전환기준_스캔.csv", index=False, encoding="utf-8-sig")
    one = sc[sc.안 == "DRT 1대"].reset_index(drop=True)
    hit = one[(one.대기_평균 > 15) | (one["거절률_%"] > 5)]
    print("\n[전환 기준 스캔]")
    print(sc.pivot(index=["이용률_%", "이용자_명"], columns="안", values=["대기_평균", "거절률_%"]).round(1).to_string())
    if len(hit):
        h = hit.iloc[0]
        print(f"-> DRT 1대가 '평균 대기 15분 초과 또는 거절률 5% 초과'가 되는 첫 이용률: {h['이용률_%']:.1f}% (하루 {h.이용자_명}명), "
              f"대기 {h.대기_평균:.1f}분, 거절률 {h['거절률_%']:.1f}%")
    else:
        print("-> 스캔 범위(8%)까지 기준 미도달")

    # ---- 참고: 필요 차량 수 (거절률 5% 이하 / 평균 대기 15분 이하)
    fl = []
    for rate in RATES:
        for nv in [1, 2, *FLEET]:
            s_ = summarize(df[rate, f"DRT {nv}대", False])
            fl.append({"이용률_%": rate * 100, "차량": nv, "운영": "단일" if nv == 1 else "권역 고정" if nv == 2 else "공동 운영",
                       "거절률_%": s_["거절률_%"], "대기_평균": s_["대기_평균"], "총시간_평균": s_["총시간_평균"], "운행거리_km": s_["운행거리_km"],
                       "가동률_%": s_["가동률_%"]})
    fl = pd.DataFrame(fl)
    fl.round(2).to_csv(f"{OUT}/DRT_v3_필요차량수.csv", index=False, encoding="utf-8-sig")
    print("\n[참고] 차량 수별 거절률·대기")
    print(fl.pivot(index="차량", columns="이용률_%", values=["거절률_%", "대기_평균"]).round(1).to_string())
    for rate in RATES:
        g = fl[(fl["이용률_%"] == rate * 100) & (fl["거절률_%"] <= 5) & (fl.대기_평균 <= 15)]
        print(f"  {rate:.0%}: 거절률 5% 이하·평균 대기 15분 이하 최소 차량 = {int(g.차량.min()) if len(g) else '12대 초과'}")

    # ---- 권역 쏠림
    st = D["stops"]
    popz = {z: st.loc[[i for i in BI if ZONEOF[i] == z], "배정_70세이상_명"].sum() for z in "AB"}
    zr = []
    for rate in RATES:
        rs = df[rate, "DRT 2대", False]
        for z in "AB":
            req_ = np.mean([r["zone_req"][z] for r in rs])
            zr.append({"이용률_%": rate * 100, "권역": z, "승차정류소": sum(ZONEOF[i] == z for i in BI), "배정_70세이상_명": popz[z],
                       "배정비중_%": popz[z] / sum(popz.values()) * 100, "하루호출_건": req_,
                       "거절률_%": np.sum([r["zone_rej"][z] for r in rs]) / max(1, np.sum([r["zone_req"][z] for r in rs])) * 100,
                       "차량가동률_%": np.mean([r["zone_util"][z] for r in rs]) * 100})
    zr = pd.DataFrame(zr)
    zr.round(1).to_csv(f"{OUT}/DRT_v3_권역부하.csv", index=False, encoding="utf-8-sig")
    print("\n[권역 부하] DRT 2대(권역 고정)")
    print(zr.round(1).to_string(index=False))

    # ---- B01
    bo = []
    for rate in RATES:
        for an in ["DRT 1대", "DRT 2대"]:
            a, b = summarize(df[rate, an, False]), summarize(df[rate, an, True])
            bo.append({"이용률_%": rate * 100, "안": an, "B01_하루호출_건": a["B01_호출_건"], "운행거리_B01포함_km": a["운행거리_km"],
                       "운행거리_B01제외_km": b["운행거리_km"], "B01로 늘어난 운행거리_km": a["운행거리_km"] - b["운행거리_km"],
                       "총시간_평균_포함": a["총시간_평균"], "총시간_평균_제외": b["총시간_평균"]})
    bo = pd.DataFrame(bo)
    bo.round(2).to_csv(f"{OUT}/DRT_v3_B01영향.csv", index=False, encoding="utf-8-sig")
    print("\n[B01]")
    print(bo.round(2).to_string(index=False))

    # ---- 귀가 호출 시간 분포 vs O-D 오는 방향 (DRT 1대, 3%)
    bt = np.concatenate([r["back_t"] for r in df[0.03, "DRT 1대", False]])
    sim_h = pd.Series((bt // 60).astype(int)).value_counts(normalize=True).sort_index()
    od_h = D["back_hour"] / D["back_hour"].sum()
    hrs = range(6, 24)
    cmp_h = pd.DataFrame({"시뮬_귀가호출_%": [sim_h.get(h, 0) * 100 for h in hrs], "OD_오는방향_%": [od_h.get(h, 0) * 100 for h in hrs]}, index=hrs)
    cmp_h.index.name = "시"
    cmp_h.round(1).to_csv(f"{OUT}/DRT_v3_귀가분포비교.csv", encoding="utf-8-sig")
    r_ = np.corrcoef(cmp_h.iloc[:, 0], cmp_h.iloc[:, 1])[0, 1]
    print(f"\n[귀가 분포] 시뮬 vs O-D 오는 방향 상관 {r_:.2f}")
    print(cmp_h.round(1).T.to_string())
    pickle.dump({"main": main, "sc": sc, "bo": bo, "cmp_h": cmp_h, "corr": r_, "fl": fl}, open(f"{INTERIM}/DRT_v3_결과.pkl", "wb"))
