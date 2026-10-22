"""팀원 2차 검토: (1) 대기 바닥과 빈 차 대기 규칙 민감도, (2) 고정노선 같은 대수 비교, (4) O-D 민감도 3대 한계, (5) 기본 41명·성장 83명 탄소, (6) 단계표 열 정리.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_검토2.py`  (선행: 금천구_DRT_반납자시뮬.py, 금천구_시흥1동도착_검증.py, 금천구_탄소계산.py, 금천구_DRT_검토보완.py)
기존 결과·체크포인트는 덮어쓰지 않음. 체크포인트 data/interim/검토2_runs/, 결과 outputs/검토2_*.csv
공통 기준: 공동 운영(1대 = 단일), B01 제외, 거절률 5% 이하 그리고 평균 대기 15분 이하, 최대 대기 30분, 300회, 같은 이용자 표본(V.users_n)
빈 차 대기 규칙
  - 기본(= 직전 하차 위치에서 대기): 엔진의 원래 규칙. 계획이 빈 차량은 마지막 하차 지점에 그대로 있다(가는 편도의 하차가 대부분 시흥1동 목적지라 시흥1동 쪽에 머무는 경우가 많음)
  - (a) 차고지 복귀: 계획이 빈 차량은 시흥3동 차고지(시흥대로30길 · 시흥3동공영주차장앞, 권역 A 대기 장소)로 돌아가 대기한다. 돌아가는 중에 받은 호출은 복귀 뒤 순서로 계획에 끼워 넣는다.
  - (b) 직전 하차 위치 대기 = 기본과 같은 규칙이라 따로 계산하지 않는다.
대기 바닥 진단: 빈 차(승객 계획 없음)가 받은 호출의 '차량 위치 -> 승차 정류소' 주행시간, 빈 차/운행 중 차량이 받은 호출의 대기를 나눠 기록
"""
import importlib
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
V = importlib.import_module("금천구_DRT_v4_시뮬")
CB = importlib.import_module("금천구_탄소계산")

OUT, INTERIM = "outputs", "data/interim"
RUNS = f"{INTERIM}/검토2_runs"
NREP, KMH, TAXI_WAIT = V.NREP, 20.0, 30.0
GRID = [int(round(V.POP70 * 0.001 * k)) for k in range(1, 46)]
POOL = None
DEP = None


def task(a):
    n, rep, mode, nveh, rule, hours = a
    us = V.users_n(n, rep, False, hours=hours)
    if mode == "DRT":
        return V.run_drt(us, nveh, V.V4, None, True, idle_dep=(V.D["I"]["DEP4_A"] if rule == "a" else None))
    return V.run_fixed_cap(us, nveh, "noB01")


def get(n, mode, nveh, rule="기본", hours=None, htag=""):
    fn = f"{RUNS}/n{n:03d}_{mode}{nveh}대_{rule}{htag}.pkl"
    if not os.path.exists(fn):
        t1 = time.time()
        rs = POOL.map(task, [(n, rp, mode, nveh, rule, hours) for rp in range(NREP)], chunksize=5)
        pickle.dump(rs, open(fn + ".tmp", "wb"))
        os.replace(fn + ".tmp", fn)
        print(f"  계산 {os.path.basename(fn)}: {time.time() - t1:.0f}초", flush=True)
    return pickle.load(open(fn, "rb"))


def cat(rs, k):
    return np.concatenate([np.asarray(r[k], float) for r in rs]) if rs else np.array([])


def summ(rs):
    w, rd = cat(rs, "wait"), cat(rs, "ride")
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    out = {"거절률_%": rej / req * 100 if req else 0.0, "대기_평균": w.mean(), "총시간_평균": (w + rd).mean(), "총시간_p95": np.percentile(w + rd, 95),
           "운행거리_km": np.mean([r["km"] for r in rs]), "빈차_km": np.mean([r["km_empty"] for r in rs]),
           "택시_km": np.mean([(sum(r["rej_dir"]) + sum(r["rej_ret_dir"])) * KMH / 60 for r in rs]),
           "직행_km": np.mean([(sum(r["dir"]) + sum(r["rej_dir"]) + sum(r["rej_ret_dir"])) * KMH / 60 for r in rs])}
    if "wait_leg" in rs[0]:
        wl = {lg: np.concatenate([np.asarray(r["wait_leg"][lg], float) for r in rs]) for lg in ("가는", "귀가")}
        app, wi, wb = cat(rs, "idle_app"), cat(rs, "wait_idle"), cat(rs, "wait_busy")
        out.update({"가는 대기_평균": wl["가는"].mean(), "귀가 대기_평균": wl["귀가"].mean(),
                    "빈 차가 받은 호출_%": len(wi) / len(w) * 100, "빈 차 -> 승차 정류소 주행_분": app.mean() if len(app) else np.nan,
                    "빈 차 배정 호출 대기_분": wi.mean() if len(wi) else np.nan, "운행 중 차량 배정 호출 대기_분": wb.mean() if len(wb) else np.nan})
    return out


ok = lambda s: s["거절률_%"] <= 5 and s["대기_평균"] <= 15


def co2(s, van, method="A"):
    """DRT - 자가용(준중형) kgCO2/일"""
    car = CB.CARS[0]
    return s["운행거리_km"] * CB.ef(van, method) + s["택시_km"] * CB.ef(CB.TAXI, method) - s["직행_km"] * CB.ef(car, method)


def scan(nv, k0, rule="기본", hours=None, htag=""):
    rows, k = [], k0
    while k < len(GRID):
        n = GRID[k]
        s = summ(get(n, "DRT", nv, rule, hours, htag))
        rows.append({"차량_대": nv, "규칙": rule, "하루 이용자_명": n, **s, "기준 충족": ok(s)})
        if not ok(s):
            break
        k += 1
    return rows


def last_ok(rows):
    g = [r for r in rows if r["기준 충족"]]
    return g[-1] if g else None


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    pd.set_option("display.width", 260)
    POOL = mp.Pool(11, initializer=V.init)
    V.init()
    EV = [v for v in CB.VANS if "전기" in v]  # 4.1, 3.3
    LIM0 = pd.read_csv(f"{OUT}/검토보완_대수별_한계이용자.csv", encoding="utf-8-sig").set_index("차량_대")["한계 이용자_명"].to_dict()

    # ---------------- 1-1. 대기 바닥(기본 규칙, 대수별 한계 지점)
    print("[1-1] 대기 바닥", flush=True)
    fl = []
    for nv in range(1, 9):
        n = int(LIM0[nv])
        s = summ(get(n, "DRT", nv))
        fl.append({"차량_대": nv, "하루 이용자_명(한계)": n, **{k: s[k] for k in ["대기_평균", "가는 대기_평균", "귀가 대기_평균", "빈 차가 받은 호출_%",
                                                                         "빈 차 -> 승차 정류소 주행_분", "빈 차 배정 호출 대기_분", "운행 중 차량 배정 호출 대기_분", "거절률_%"]}})
    FL = pd.DataFrame(fl)
    FL.round(2).to_csv(f"{OUT}/검토2_대기바닥.csv", index=False, encoding="utf-8-sig")
    print(FL.round(1).to_string(index=False))

    # ---------------- 1-2. 빈 차 대기 규칙별 2~4대 한계
    print("\n[1-2] 빈 차 대기 규칙별 한계", flush=True)
    lim_rows, all_rows = [], []
    for rule in ["기본", "a"]:
        k0 = GRID.index(26)
        for nv in (2, 3, 4):
            r = scan(nv, k0, rule)
            all_rows += r
            L = last_ok(r)
            nx = r[-1] if not r[-1]["기준 충족"] else None
            lim_rows.append({"규칙": "기본(직전 하차 위치 대기)" if rule == "기본" else "(a) 시흥3동 차고지 복귀", "차량_대": nv,
                             "한계 이용자_명": L["하루 이용자_명"] if L else 0,
                             "다음 지점_명": nx["하루 이용자_명"] if nx else None,
                             "먼저 걸린 기준": ("" if nx is None else "거절률" if nx["거절률_%"] > 5 and nx["대기_평균"] <= 15 else "평균 대기" if nx["거절률_%"] <= 5 else "둘 다")})
            k0 = GRID.index(L["하루 이용자_명"]) if L else k0
    LR = pd.DataFrame(lim_rows)
    LR.to_csv(f"{OUT}/검토2_빈차규칙_한계.csv", index=False, encoding="utf-8-sig")
    print(LR.to_string(index=False))

    # ---------------- 1-3. 같은 수요(기본 한계 지점)에서 규칙별 대기(가는/귀가)·공차·탄소
    cmp_rows = []
    for nv in (2, 3, 4):
        n = int(LIM0[nv])
        for rule in ["기본", "a"]:
            s = summ(get(n, "DRT", nv, rule))
            cmp_rows.append({"차량_대": nv, "하루 이용자_명": n, "규칙": "기본" if rule == "기본" else "(a) 차고지 복귀", **{k: s[k] for k in [
                "거절률_%", "대기_평균", "가는 대기_평균", "귀가 대기_평균", "빈 차 -> 승차 정류소 주행_분", "운행거리_km", "빈차_km"]},
                "빈차비율_%": s["빈차_km"] / s["운행거리_km"] * 100,
                **{f"DRT-자가용 kgCO2/일({'4.1' if '4.1' in v else '3.3'})": co2(s, v) for v in EV}})
    CMP = pd.DataFrame(cmp_rows)

    # 탄소 손익분기: 규칙별로 각 하루 이용자에서 기준을 맞추는 최소 대수(1~4대)로 운행할 때 감축 시작 이용자
    be = []
    curve = []
    for rule in ["기본", "a"]:
        for n in GRID[:19]:  # 9~168명
            s_need, k_need = None, None
            for nv in range(1, 5):
                s = summ(get(n, "DRT", nv, rule))
                if ok(s):
                    s_need, k_need = s, nv
                    break
            if s_need is None:
                continue
            row = {"규칙": rule, "하루 이용자_명": n, "필요 대수": k_need, "운행거리_km": s_need["운행거리_km"], "빈차_km": s_need["빈차_km"]}
            for v in EV:
                row[f"차이({'4.1' if '4.1' in v else '3.3'})"] = co2(s_need, v)
            curve.append(row)
    CV = pd.DataFrame(curve)
    for rule, g in CV.groupby("규칙", sort=False):
        for e in ("4.1", "3.3"):
            col = f"차이({e})"
            frm = next((n for n in g["하루 이용자_명"] if (g.loc[g["하루 이용자_명"] >= n, col] < 0).all()), None)
            be.append({"규칙": "기본" if rule == "기본" else "(a) 차고지 복귀", "전비_km/kWh": e, "이 이상 모든 계산 지점에서 감축_하루 이용자": frm})
    BE = pd.DataFrame(be)
    CMP.round(2).to_csv(f"{OUT}/검토2_빈차규칙_대기_공차_탄소.csv", index=False, encoding="utf-8-sig")
    BE.to_csv(f"{OUT}/검토2_빈차규칙_탄소손익분기.csv", index=False, encoding="utf-8-sig")
    CV.round(2).to_csv(f"{OUT}/검토2_빈차규칙_탄소곡선.csv", index=False, encoding="utf-8-sig")
    print(CMP.round(1).to_string(index=False))
    print(BE.to_string(index=False))

    # ---------------- 2. 고정노선 같은 대수
    print("\n[2] 고정노선 같은 대수", flush=True)
    M = pd.read_csv(f"{OUT}/DRT_반납자_시나리오_결과.csv", encoding="utf-8-sig")
    rep = M[M.구분 == "대표(반납자)"]
    fr = []
    for n, g in rep.groupby("하루 이용자_명"):
        d = g[g.종류 == "DRT 필요"].iloc[0]
        k = int(d.차량_대)
        f2 = g[g.안 == "고정노선 2대(B01 제외, 정원 12)"].iloc[0]
        fk = summ(get(n, "고정", k))
        rs = get(n, "고정", k)
        left1 = sum(r["left1"] for r in rs) / len(cat(rs, "wait")) * 100
        row = {"하루 이용자_명": n, "DRT 대수": k, "DRT 평균/p95_분": f"{d.총시간_전체승객_평균:.1f} / {d.총시간_전체승객_p95:.1f}",
               "고정노선 같은 대수": k, "고정노선 같은 대수 평균/p95_분": f"{fk['총시간_평균']:.1f} / {fk['총시간_p95']:.1f}", "같은 대수 첫차만차_%": left1,
               "고정노선 2대(기존) 평균/p95_분": f"{f2.총시간_전체승객_평균:.1f} / {f2.총시간_전체승객_p95:.1f}",
               "차이(같은 대수 고정 - DRT) 평균_분": fk["총시간_평균"] - d.총시간_전체승객_평균, "고정노선 같은 대수 운행거리_km": fk["운행거리_km"],
               "총시간 기준": "대기+차내(도보 제외), 정원 12"}
        fr.append(row)
        if n == 24:
            f1 = g[g.안 == "고정노선 1대(B01 제외, 정원 12)"].iloc[0]
            fr.append({"하루 이용자_명": 24, "DRT 대수": k, "DRT 평균/p95_분": row["DRT 평균/p95_분"], "고정노선 같은 대수": 1,
                       "고정노선 같은 대수 평균/p95_분": f"{f1.총시간_전체승객_평균:.1f} / {f1.총시간_전체승객_p95:.1f}", "같은 대수 첫차만차_%": f1["첫차만차_못탐_%"],
                       "고정노선 2대(기존) 평균/p95_분": row["고정노선 2대(기존) 평균/p95_분"],
                       "차이(같은 대수 고정 - DRT) 평균_분": f1.총시간_전체승객_평균 - d.총시간_전체승객_평균, "고정노선 같은 대수 운행거리_km": f1.운행거리_km,
                       "총시간 기준": "대기+차내(도보 제외), 정원 12 — 1대(기존 결과 파일 값)"})
    FR = pd.DataFrame(fr).drop_duplicates(subset=["하루 이용자_명", "고정노선 같은 대수"])
    FR.round(2).to_csv(f"{OUT}/검토2_고정노선_같은대수.csv", index=False, encoding="utf-8-sig")
    print(FR.round(1).to_string(index=False))

    # ---------------- 4. O-D 민감도 3대 한계
    print("\n[4] O-D 민감도 3대 한계", flush=True)
    arr = pd.read_csv(f"{OUT}/검증_시흥1동도착_70세_시간대.csv", encoding="utf-8-sig", index_col=0)["도착70_귀가제외_평일평균"]
    hours = arr.loc[8:16]
    r3 = scan(3, GRID.index(71), "기본", hours, "_OD도착")
    OD3 = pd.DataFrame(r3)
    OD3.round(2).to_csv(f"{OUT}/검토2_OD민감도_3대한계.csv", index=False, encoding="utf-8-sig")
    print(OD3[["하루 이용자_명", "거절률_%", "대기_평균", "기준 충족"]].round(1).to_string(index=False))

    # ---------------- 5. 기본 41명·성장 83명 탄소
    print("\n[5] 기본·성장 시나리오 탄소", flush=True)
    cr = []
    for name, n, k in [("기본 시나리오(2대)", 41, 2), ("성장 시나리오(3대)", 83, 3), ("참고: 성장 83명을 2대로", 83, 2)]:
        s = summ(get(n, "DRT", k))
        for v in EV:
            dd = co2(s, v)
            cr.append({"시나리오": name, "하루 이용자_명": n, "DRT 대수": k, "전비_km/kWh": "4.1" if "4.1" in v else "3.3",
                       "DRT 운행 km/일": s["운행거리_km"], "자가용 직행 km/일": s["직행_km"], "DRT-자가용 kgCO2/일": dd,
                       "연간 tCO2(평일 260일)": dd * 260 / 1000, "연간 tCO2(365일)": dd * 365 / 1000, "판정": "감축" if dd < 0 else "증가",
                       "거절률_%": s["거절률_%"], "평균대기_분": s["대기_평균"]})
    CR = pd.DataFrame(cr)
    CR.round(3).to_csv(f"{OUT}/검토2_탄소_기본성장.csv", index=False, encoding="utf-8-sig")
    print(CR.round(2).to_string(index=False))

    # ---------------- 6. 단계표 열 정리(새 파일)
    ST = pd.read_csv(f"{OUT}/검토보완_단계표.csv", encoding="utf-8-sig").rename(columns={"단계": "차량"})
    ST.insert(0, "단계", ["해당 없음(1대 시범 권하지 않음)"] + [f"{i}단계" for i in range(1, len(ST))])
    ST.to_csv(f"{OUT}/검토2_단계표.csv", index=False, encoding="utf-8-sig")
    print("\n[6] 단계표")
    print(ST[["단계", "차량", "처리 범위(하루 이용자)", "표시"]].fillna("").to_string(index=False))
    POOL.close()
    print("완료")
