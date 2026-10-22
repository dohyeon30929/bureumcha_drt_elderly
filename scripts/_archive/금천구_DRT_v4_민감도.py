"""DRT v4 운영 방식 민감도 + 시간대별 거절. v4 가정(최대 승차 min(직행+15, 40), 17시 접수 마감, 새 권역)에서
  (a) 최대 대기 30분 vs 45분, (b) 사전예약: 가는 호출 50%를 전날 예약(희망 시각 ±15분), (c) 둘 다
를 DRT 1대·2대로 비교하고, 이용률 1% 시간대별 호출·거절을 기록한다.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_v4_민감도.py`  (선행: 금천구_DRT_v4_준비.py)
엔진 확장: v4 엔진과 같은 규칙에 시간창만 더했다.
  - 각 승차에 승차 가능 시간창 [e, l]: 실시간 호출 e = 호출 시각, l = 호출 시각 + 최대 대기 / 사전예약 e = 희망 - 15분, l = 희망 + 15분
  - 차량은 시간창 시작 전에 도착하면 그 자리에서 대기(출발을 늦춰 시간창에 맞춤), l을 넘기면 불가
  - 사전예약은 하루 시작(8시)에 희망 시각 순으로 삽입(거절 가능), 귀가 호출은 모두 실시간
  - 사전예약자의 '대기' = |실제 승차 시작 - 희망 시각| (희망 시각과의 차이)
같은 시드(20261011)·같은 이용자 표본을 쓰므로 '실시간 30분' 결과는 v4 본 결과와 같아야 한다(검증용)
보고 기준: 최대 대기 30분이 기본, 45분은 민감도. 한계 이용자 = 평균 대기 15분·거절률 5% 이하를 지키는 마지막 계산 지점
사전예약은 단순 삽입 모형(하루 시작에 희망 시각 순으로 한 번 삽입, 재최적화 없음)이라 효과를 과소평가함 -> 결과 해석에 쓰지 않고
  outputs/DRT_v4_사전예약_참고_단순삽입모형.csv 로만 남김(methodology 한계)
"""
import heapq
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd

INTERIM, OUT = "data/interim", "outputs"
POP70, CAP, SVC, OPEN, CLOSE = 8817, 12, 1.0, 480.0, 1020.0
NREP, SEED, WIN = 300, 20261011, 15.0
RATES = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.015, 0.02, 0.025, 0.03]
FOCUS = {26: 0.003, 62: 0.007, 88: 0.01}  # 1대 한계, 2대 한계, 이용률 1%
VARIANTS = {"실시간·대기30": (30.0, 0.0), "실시간·대기45": (45.0, 0.0), "예약50%·대기30": (30.0, 0.5), "예약50%·대기45": (45.0, 0.5)}
D = None


def init():
    global D, T, M, BI, DI, P, W, ZONE, DEPS, HOURS
    D = pickle.load(open(f"{INTERIM}/DRT_v4_입력.pkl", "rb"))
    st = D["stops"]
    T, M = D["T"].tolist(), D["M"].tolist()
    BI, DI, P = D["BI"], D["DI"], D["P"]
    W = st.loc[BI, "배정_70세이상_명"].to_numpy(float)
    W = W / W.sum()
    ZONE = D["ZONE4"]
    I = D["I"]
    DEPS = {"single": I["D2"], "A": I["DEP4_A"], "B": I["DEP4_B"]}
    HOURS = D["go_hour"]


def users(rate, rep):
    """v4 시뮬과 같은 난수 순서(같은 표본)"""
    rng = np.random.default_rng([SEED, int(rate * 10000), rep])
    n = int(round(POP70 * rate))
    h = HOURS[HOURS.index < CLOSE / 60]
    o = rng.choice(BI, size=n, p=W)
    d = np.array([DI[rng.choice(5, p=P[BI.index(x)])] for x in o])
    hr = rng.choice(h.index.to_numpy(), size=n, p=(h / h.sum()).to_numpy())
    t = hr * 60 + rng.uniform(0, 60, n)
    stay = rng.uniform(60, 180, n)
    us = sorted((float(t[k]), int(o[k]), int(d[k]), float(stay[k])) for k in range(n))
    book = np.random.default_rng([SEED, int(rate * 10000), rep, 7]).random(len(us))  # 예약 여부용 별도 난수
    return us, book


def run(us, book, nveh, maxwait, pbook):
    vz = [("single", DEPS["single"])] if nveh == 1 else [(z, DEPS[z]) for z in "AB"]
    veh = [{"z": z, "node": dp, "t": OPEN, "plan": [], "load": 0, "dist": 0.0, "busy": 0.0, "dep": dp, "ln": 0.0, "mv": 0.0} for z, dp in vz]
    req, heap, pre = [], [], []
    for k, (t, o, d, stay) in enumerate(us):
        r = {"o": o, "d": d, "stay": stay, "leg": "가는", "mr": min(T[o][d] + 15.0, 40.0), "z": ZONE[o], "st": "대기", "want": t}
        if book[k] < pbook:
            r.update({"t": OPEN, "e": t - WIN, "l": t + WIN, "pre": True})
            pre.append(len(req))
        else:
            r.update({"t": t, "e": t, "l": t + maxwait, "pre": False})
            heapq.heappush(heap, (t, len(req)))
        req.append(r)

    def leave(v, n_, rid, kind):
        """다음 이벤트로 떠나는 시각(승차 시간창에 맞춰 늦춤)"""
        if kind == 1:
            return max(v["t"], req[rid]["e"] - T[v["node"]][n_])
        return v["t"]

    def advance(v, t, idle=True):
        while v["plan"]:
            n_, kind, rid = v["plan"][0]
            dep = leave(v, n_, rid, kind)
            if dep > t:
                break
            v["plan"].pop(0)
            tt = T[v["node"]][n_]
            v["dist"] += M[v["node"]][n_]
            v["busy"] += tt + SVC
            v["ln"] += v["load"] * tt
            v["mv"] += tt
            a = dep + tt
            r = req[rid]
            if kind == 1:
                r["ap"], r["sp"] = a, a + SVC
                v["load"] += 1
            else:
                r["ad"], r["st"] = a, "완료"
                v["load"] -= 1
                if r["leg"] == "가는":
                    q = {"t": a + r["stay"], "o": r["d"], "d": r["o"], "stay": 0, "leg": "귀가", "mr": min(T[r["d"]][r["o"]] + 15.0, 40.0),
                         "z": r["z"], "st": "대기", "pre": False}
                    q.update({"e": q["t"], "l": q["t"] + maxwait, "want": q["t"]})
                    heapq.heappush(heap, (q["t"], len(req)))
                    req.append(q)
            v["node"], v["t"] = n_, a + SVC
        if idle and v["t"] < t:
            v["t"] = t

    def evaluate(v, plan):
        node, t, load, sp = v["node"], v["t"], v["load"], {}
        for n_, kind, rid in plan:
            r = req[rid]
            a = t + T[node][n_]
            if kind == 1:
                a = max(a, r["e"])
                if a > r["l"]:
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

    def ready_times(v, plan):
        """기존 계획의 각 이벤트 처리 완료 시각(삽입은 이 시각들을 늦추기만 함 -> 가지치기에 사용)"""
        node, t, out = v["node"], v["t"], []
        for n_, kind, rid in plan:
            a = t + T[node][n_]
            if kind == 1:
                a = max(a, req[rid]["e"])
            node, t = n_, a + SVC
            out.append(t)
        return out

    def insert(rid):
        r = req[rid]
        cands = [v for v in veh if v["z"] == r["z"]] if nveh == 2 else veh
        best = None
        for v in cands:
            base = evaluate(v, v["plan"]) if v["plan"] else v["t"]
            pl = v["plan"]
            rt = ready_times(v, pl)
            Pk, Dk = (r["o"], 1, rid), (r["d"], -1, rid)
            for i in range(len(pl) + 1):
                prev = v["t"] if i == 0 else rt[i - 1]
                if prev > r["l"]:  # 이 위치부터는 승차 시간창을 못 맞춤
                    break
                pick = max(prev + T[v["node"] if i == 0 else pl[i - 1][0]][r["o"]], r["e"]) + SVC
                for j in range(i, len(pl) + 1):
                    if j > i and rt[j - 1] - pick > r["mr"]:  # 이 뒤 위치는 승차시간 초과
                        break
                    new = pl[:i] + [Pk] + pl[i:j] + [Dk] + pl[j:]
                    e = evaluate(v, new)
                    if e is not None and (best is None or e - base < best[0]):
                        best = (e - base, v, new)
        if best is None:
            r["st"] = "거절"
        else:
            best[1]["plan"] = best[2]

    for rid in sorted(pre, key=lambda k: req[k]["want"]):  # 전날 예약: 희망 시각 순
        insert(rid)
    while True:
        if not heap:
            # 호출이 다 끝나도 차량에 남은 '가는' 하차가 귀가 호출을 만들 수 있음 -> 다음 출발 시각까지 진행 후 다시 확인
            busy = [v for v in veh if v["plan"]]
            if not busy:
                break
            tn = min(leave(v, n_, rid_, kind) for v in busy for n_, kind, rid_ in v["plan"][:1])
            for v in veh:
                advance(v, tn)
            continue
        t, rid = heapq.heappop(heap)
        for v in veh:
            advance(v, t)
        insert(rid)
    for v in veh:
        advance(v, 1e9, idle=False)
        v["dist"] += M[v["node"]][v["dep"]]
        v["t"] += T[v["node"]][v["dep"]]
    pax = [r for r in req if r["st"] == "완료"]
    wait = [abs(r["ap"] - r["want"]) if r["pre"] else r["ap"] - r["t"] for r in pax]
    return {"요청": len(req), "거절": sum(r["st"] == "거절" for r in req), "wait": wait, "ride": [r["ad"] - r["sp"] for r in pax],
            "pre_wait": [abs(r["ap"] - r["want"]) for r in pax if r["pre"]], "pre_n": sum(r["pre"] for r in req),
            "pre_rej": sum(r["pre"] and r["st"] == "거절" for r in req),
            "km": sum(v["dist"] for v in veh) / 1000, "occ": sum(v["ln"] for v in veh) / max(1e-9, sum(v["mv"] for v in veh)),
            "end": max(v["t"] for v in veh),
            "hour_req": np.bincount([int(r["want"] // 60) for r in req], minlength=24).tolist(),
            "hour_rej": np.bincount([int(r["want"] // 60) for r in req if r["st"] == "거절"], minlength=24).tolist(),
            "hour_rej_leg": {lg: np.bincount([int(r["want"] // 60) for r in req if r["st"] == "거절" and r["leg"] == lg], minlength=24).tolist()
                             for lg in ("가는", "귀가")},
            "hour_req_leg": {lg: np.bincount([int(r["want"] // 60) for r in req if r["leg"] == lg], minlength=24).tolist() for lg in ("가는", "귀가")}}


def task(a):
    rate, rep, nveh, var = a
    us, book = users(rate, rep)
    mw, pb = VARIANTS[var]
    return run(us, book, nveh, mw, pb)


def summ(rs, rate):
    w, rd = np.concatenate([r["wait"] for r in rs]), np.concatenate([r["ride"] for r in rs])
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    pw = np.concatenate([r["pre_wait"] for r in rs]) if any(r["pre_wait"] for r in rs) else np.array([np.nan])
    pn, pr = sum(r["pre_n"] for r in rs), sum(r["pre_rej"] for r in rs)
    return {"이용자_명": int(round(POP70 * rate)), "거절률_%": rej / req * 100, "대기_평균": w.mean(), "대기_p95": np.percentile(w, 95),
            "승차_평균": rd.mean(), "총시간_평균": (w + rd).mean(), "총시간_p95": np.percentile(w + rd, 95),
            "예약자_희망시각차_평균": np.nanmean(pw), "예약_거절률_%": pr / pn * 100 if pn else np.nan,
            "평균탑승인원": np.mean([r["occ"] for r in rs]), "운행거리_km": np.mean([r["km"] for r in rs]),
            "종료_평균": np.mean([r["end"] for r in rs])}


if __name__ == "__main__":
    init()
    RUNS = f"{INTERIM}/DRT_v4_민감도_runs"
    os.makedirs(RUNS, exist_ok=True)
    # 사전예약은 계산량이 커서(3%·1대 1회 약 13초) 이용률 1.5%까지만: 질문 범위(35·71·88명 = 0.4·0.8·1%)와 한계 근처를 포함
    scen = [(r, nv, var) for var in VARIANTS for nv in (1, 2) for r in RATES if VARIANTS[var][1] == 0 or r <= 0.015]
    fn = lambda s: f"{RUNS}/r{s[0] * 1000:05.1f}_DRT{s[1]}대_{s[2]}.pkl"
    todo = [s for s in scen if not os.path.exists(fn(s))]
    print(f"시나리오 {len(scen)}개 중 완료 {len(scen) - len(todo)}개, 남은 {len(todo)}개", flush=True)
    t0 = time.time()
    with mp.Pool(11, initializer=init) as pool:
        for k, s in enumerate(todo, 1):
            rs = pool.map(task, [(s[0], rp, s[1], s[2]) for rp in range(NREP)], chunksize=5)
            pickle.dump(rs, open(fn(s) + ".tmp", "wb"))
            os.replace(fn(s) + ".tmp", fn(s))
            print(f"[{k}/{len(todo)}] {s[0]:.1%} DRT {s[1]}대 {s[2]} (누적 {(time.time() - t0) / 60:.1f}분)", flush=True)
    R = {s: pickle.load(open(fn(s), "rb")) for s in scen}
    pd.set_option("display.width", 250)

    # ---- 1. 시간대별 호출·거절 (실시간·대기30 = v4 기본, 이용률 1%)
    rows = []
    for nv in (1, 2):
        rs = R[(0.01, nv, "실시간·대기30")]
        for h in range(8, 23):
            rq = np.mean([r["hour_req"][h] for r in rs])
            rj = np.mean([r["hour_rej"][h] for r in rs])
            rows.append({"안": f"DRT {nv}대", "시": h, "호출_가는": np.mean([r["hour_req_leg"]["가는"][h] for r in rs]),
                         "호출_귀가": np.mean([r["hour_req_leg"]["귀가"][h] for r in rs]), "호출_계": rq,
                         "거절_가는": np.mean([r["hour_rej_leg"]["가는"][h] for r in rs]), "거절_귀가": np.mean([r["hour_rej_leg"]["귀가"][h] for r in rs]),
                         "거절_계": rj, "거절률_%": rj / rq * 100 if rq else 0})
    hr = pd.DataFrame(rows)
    hr.round(2).to_csv(f"{OUT}/DRT_v4_시간대별_거절_1pct.csv", index=False, encoding="utf-8-sig")
    print("\n[1] 시간대별 호출·거절 (이용률 1%, 하루 평균 건)")
    print(hr.pivot(index="시", columns="안", values=["호출_계", "거절_계", "거절률_%"]).round(1).to_string())
    for nv in (1, 2):
        g = hr[hr.안 == f"DRT {nv}대"]
        am = g[g.시.between(8, 11)]
        print(f"  DRT {nv}대: 거절 중 8~11시 비중 {am.거절_계.sum() / g.거절_계.sum():.0%}, 호출 중 8~11시 비중 {am.호출_계.sum() / g.호출_계.sum():.0%}, "
              f"거절 중 가는 호출 {g.거절_가는.sum() / g.거절_계.sum():.0%}")

    # ---- 3. 최대 대기 민감도 (30분 기본, 45분 민감도). 사전예약은 참고 파일로만
    allv = []
    for var in VARIANTS:
        for nv in (1, 2):
            for r in RATES:
                if (r, nv, var) in R:
                    allv.append({"방식": var, "안": f"DRT {nv}대", "이용률_%": r * 100, **summ(R[(r, nv, var)], r)})
    allv = pd.DataFrame(allv)
    pre = allv[allv.방식.str.startswith("예약")]
    pre.round(2).to_csv(f"{OUT}/DRT_v4_사전예약_참고_단순삽입모형.csv", index=False, encoding="utf-8-sig")
    out = allv[allv.방식.str.startswith("실시간")].drop(columns=["예약자_희망시각차_평균", "예약_거절률_%"])
    out.round(2).to_csv(f"{OUT}/DRT_v4_최대대기민감도_전체.csv", index=False, encoding="utf-8-sig")
    foc = out[out["이용률_%"].isin([v * 100 for v in FOCUS.values()])]
    foc.round(2).to_csv(f"{OUT}/DRT_v4_최대대기민감도_26_62_88명.csv", index=False, encoding="utf-8-sig")
    print("\n[3] 최대 대기 민감도 (하루 이용자 26·62·88명; 30분 = 기본)")
    print(foc[["방식", "안", "이용자_명", "거절률_%", "대기_평균", "총시간_평균", "평균탑승인원", "운행거리_km"]].round(1).to_string(index=False))
    lim = []
    for var in [v for v in VARIANTS if v.startswith("실시간")]:
        for nv in (1, 2):
            g = out[(out.방식 == var) & (out.안 == f"DRT {nv}대")].sort_values("이용률_%")
            ok = g[(g.대기_평균 <= 15) & (g["거절률_%"] <= 5)]
            last = ok.iloc[-1] if len(ok) else None
            over = g[g.이용자_명 > last.이용자_명] if last is not None else g
            lim.append({"최대 대기": var.split("·")[1].replace("대기", "") + "분" + (" (기본)" if var.endswith("30") else " (민감도)"), "안": f"DRT {nv}대",
                        "한계 이용자_명": int(last.이용자_명) if last is not None else 0,
                        "한계에서 대기·거절": f"{last.대기_평균:.1f}분·{last['거절률_%']:.1f}%" if last is not None else "",
                        "다음 계산 지점_명": int(over.iloc[0].이용자_명) if len(over) else None,
                        "다음 지점 대기·거절": f"{over.iloc[0].대기_평균:.1f}분·{over.iloc[0]['거절률_%']:.1f}%" if len(over) else ""})
    lim = pd.DataFrame(lim)
    lim.to_csv(f"{OUT}/DRT_v4_최대대기별_한계이용자.csv", index=False, encoding="utf-8-sig")
    print("\n[3] 한계 이용자 = 평균 대기 15분 이하·거절률 5% 이하를 지키는 마지막 계산 지점")
    print(lim.to_string(index=False))
    pickle.dump({"hr": hr, "out": out, "lim": lim, "pre": pre}, open(f"{INTERIM}/DRT_v4_민감도_결과.pkl", "wb"))
