"""반납자 수요 기준 메인 시뮬레이션: 대표 시나리오별 필요 DRT 대수·총시간·운행거리 + 같은 승객의 고정노선(정원 반영) 비교.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_반납자시뮬.py`  (선행: 금천구_DRT_v4_준비.py, 금천구_반납자수요.py)
공통 조건(팀원 검토 반영)
  - B01 통일: 기본 비교는 DRT·고정노선 모두 B01 제외. 이용자는 B01을 뺀 승차 12곳 배정 인원 비중으로 추첨(V.users_n)하고
    같은 표본(같은 승객·같은 목적지)을 DRT와 고정노선에 넣음. B01 포함 효과는 별도(4번)
  - 차량 대수 기준 하나로 통일: 거절률 5% 이하 '그리고' 평균 대기 15분 이하를 만족하는 최소 대수, 공동 운영
    (1대 = 단일, 2대 이상 = 공동 운영. 2대 권역 고정은 참고로만)
  - 고정노선: 정원 12명 반영(V.run_fixed_cap). 만차면 다음 차량 대기, 첫 차·다음 차까지 못 탄 비율 기록
  - v4 가정: 20km/h, 정원 12, 승하차 1분, 최대 대기 30분, 최대 승차 min(직행+15, 40), 가는 호출 8~17시, 귀가 U(1,3)시간 후, 300회
  - 거절 편도 = 30분 대기 + 택시(직행) 로 넣은 '전체 승객' 총시간도 함께 계산
시나리오
  1. 대표 시나리오(하루 이용자 24·41·69·98·138·276명): 메인
  2. 민감도: 이용률 1·3·5%(하루 88·265·441명), 같은 파이프라인
  3. 점심 증차 실험안: 필요 k대 대신 'k-1대 종일 + 1대 10~14시 / 10~15시 / 10~16시'. 하루 기준과 시간대별 최대 거절률(8~19시)을 확인한다.
  4. B01 포함 효과(대표 시나리오): 승차 13곳으로 추첨, 고정노선 B01 포함 순환 vs 제외 순환, 같은 대수 DRT의 운행거리, B01 호출 빈도
체크포인트: 시나리오마다 data/interim/DRT_반납자_runs/ 에 저장, 재실행 시 완료분 건너뜀
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

INTERIM, OUT = "data/interim", "outputs"
RUNS = f"{INTERIM}/DRT_반납자_runs"
NREP, TAXI_WAIT, KMH, MAXF = V.NREP, 30.0, 20.0, 14
REP = [24, 41, 69, 98, 138, 276]  # 반납자 수요 범위에서 낮음~높음(최저·참여 30%·50%·중간·높음·상한)
SENS = [88, 265, 441]  # 이용률 1·3·5%
PEAKS = [(600.0, 840.0), (600.0, 900.0), (600.0, 960.0)]  # 10~14, 10~15, 10~16시
POOL = None
if os.environ.get("SIM_SMOKE"):  # 점검용: 3회 반복, 별도 체크포인트 폴더
    NREP, RUNS, REP, SENS = 3, RUNS + "_smoke", [41, 138], [88]


def task(a):
    n, rep, mode, nveh, shifts, loopkey, b01, pool, cap, mw = a
    us = V.users_n(n, rep, b01)
    if mode == "DRT":
        return V.run_drt(us, nveh, V.V4, shifts, pool, mw)
    return V.run_fixed_cap(us, nveh, loopkey, cap)


def get(n, mode, nveh, shifts=None, loopkey="noB01", b01=False, pool=True, cap=V.CAP, mw=V.MAXWAIT):
    sh = "" if not shifts else "_" + "+".join(f"{int(a)}-{int(b)}" for a, b in shifts if b < 1e8)
    tag = f"n{n:03d}_{mode}{nveh}대{'' if mode != 'DRT' else ('_공동' if pool and nveh > 1 else '_권역' if nveh == 2 else '')}" \
          f"{'_' + loopkey if mode != 'DRT' else ''}{'_정원' + str(cap) if mode != 'DRT' and cap < 1e8 else ''}{'_B01포함' if b01 else ''}{sh}{'_대기' + str(int(mw)) if mw != V.MAXWAIT else ''}"
    fn = f"{RUNS}/{tag}.pkl"
    if not os.path.exists(fn):
        t1 = time.time()
        rs = POOL.map(task, [(n, rp, mode, nveh, shifts, loopkey, b01, pool, cap, mw) for rp in range(NREP)], chunksize=5)
        pickle.dump(rs, open(fn + ".tmp", "wb"))
        os.replace(fn + ".tmp", fn)
        print(f"  계산 {tag}: {time.time() - t1:.0f}초", flush=True)
    return pickle.load(open(fn, "rb"))


def summ(rs):
    w, rd, dr = (np.concatenate([np.asarray(r[k], float) for r in rs]) for k in ("wait", "ride", "dir"))
    served = w + rd
    alt = np.concatenate([np.asarray(r["rej_dir"] + r["rej_ret_dir"] + r.get("miss_dir", []), float) for r in rs])
    allp = np.r_[served, TAXI_WAIT + alt]
    nuser = sum(r["n_user"] for r in rs)
    assert len(allp) == 2 * nuser, (len(allp), nuser)
    # 반복별 평균(반복 범위 = 반복 간 표준편차)
    rep_tot = np.array([np.mean(np.r_[np.add(r["wait"], r["ride"]), TAXI_WAIT + np.asarray(r["rej_dir"] + r["rej_ret_dir"] + r.get("miss_dir", []), float)])
                        for r in rs])
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    km, ke, kl, pkm = (np.mean([r[k] for r in rs]) for k in ("km", "km_empty", "km_loaded", "pkm"))
    legs = len(served) / len(rs)
    hq = np.sum([r["hour_req"] for r in rs], axis=0)
    hj = np.sum([r["hour_rej"] for r in rs], axis=0)
    hrate = [hj[h] / hq[h] * 100 for h in range(8, 20) if hq[h] / len(rs) >= 1]  # 하루 1건 이상 호출되는 시간대
    return {"이용자_명": int(round(nuser / len(rs))), "편도_건": 2 * nuser / len(rs), "거절률_%": rej / req * 100 if req else 0.0,
            "대기_평균": w.mean(), "대기_p95": np.percentile(w, 95), "승차_평균": rd.mean(), "승차_p95": np.percentile(rd, 95),
            "총시간_평균": served.mean(), "총시간_p95": np.percentile(served, 95),
            "총시간_전체승객_평균": allp.mean(), "총시간_전체승객_p95": np.percentile(allp, 95), "총시간_전체승객_반복SD": rep_tot.std(ddof=1),
            "택시대체_편도_%": len(alt) / len(allp) * 100,
            "가동률_%": np.mean([np.mean(r["util"]) for r in rs]) * 100,
            "운행거리_km": km, "운행거리_반복SD": np.std([r["km"] for r in rs], ddof=1), "빈차_km": ke, "탑승_km": kl, "빈차비율_%": ke / km * 100,
            "인km": pkm, "탑승편도_건": legs, "탑승편도당_차량km": km / legs, "직행_평균km": dr.mean() * KMH / 60,
            "평균탑승인원_빈차포함": pkm / km, "평균탑승인원_빈차제외": pkm / kl if kl else np.nan,
            "택시대체_km": np.mean([(sum(r["rej_dir"]) + sum(r["rej_ret_dir"]) + sum(r.get("miss_dir", []))) * KMH / 60 for r in rs]),
            "첫차만차_못탐_%": sum(r.get("left1", 0) for r in rs) / len(served) * 100 if "left1" in rs[0] else np.nan,
            "다음차도_못탐_%": sum(r.get("left2", 0) for r in rs) / len(served) * 100 if "left2" in rs[0] else np.nan,
            "구간최대탑승_명": max(r.get("maxload", 0) for r in rs) if "maxload" in rs[0] else np.nan,
            "시간대최대거절률_%": max(hrate) if hrate else 0.0,
            "종료_평균": np.mean([r["end"] for r in rs])}


ok = lambda s: s["거절률_%"] <= 5 and s["대기_평균"] <= 15


def min_fleet(n):
    tried = []
    for nv in range(1, MAXF + 1):
        s = summ(get(n, "DRT", nv))
        tried.append((nv, s))
        if ok(s):
            return nv, s, tried
    return None, s, tried


lab = lambda nv: "DRT 1대" if nv == 1 else f"DRT {nv}대(공동 운영)"
hm = lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"

if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    pd.set_option("display.width", 280)
    POOL = mp.Pool(11, initializer=V.init)
    V.init()
    t0 = time.time()
    dem = pd.read_csv(f"{OUT}/DRT_반납자기준_수요추정.csv", encoding="utf-8-sig")
    combo = dem.groupby("하루 이용자_명").apply(
        lambda g: " / ".join(f"잔존 {r['잔존율_%']}·참여 {r['참여율_%']}·{r['이용빈도']}·{r['운행일']}" for _, r in g.iterrows()), include_groups=False)

    rows, search, need = [], [], {}
    for n in REP + SENS:
        grp = "대표(반납자)" if n in REP else "민감도(이용률)"
        print(f"[{grp}] 하루 {n}명", flush=True)
        k, sk, tried = min_fleet(n)
        need[n] = k
        for nv, s in tried:
            search.append({"하루 이용자_명": n, "구분": grp, "차량_대": nv, "기준 충족": ok(s), **s})
        plans = [(f"DRT 필요 대수: {lab(k) if k else f'{MAXF}대 초과'}", sk, "DRT 필요")]
        if n in REP:
            plans.append(("참고: DRT 2대 권역 고정", summ(get(n, "DRT", 2, pool=False)), "참고"))
        for nv in (1, 2):
            plans.append((f"고정노선 {nv}대(B01 제외, 정원 12)", summ(get(n, "고정", nv)), "고정"))
        plans.append(("참고: 고정노선 2대(정원 무제한)", summ(get(n, "고정", 2, cap=1e9)), "참고"))
        for name, s, kind in plans:
            rows.append({"구분": grp, "하루 이용자_명": n, "반납자 조합": combo.get(n, f"이용률 {n / V.POP70:.0%}"), "안": name, "종류": kind,
                         "차량_대": k if kind == "DRT 필요" else None, **s})
    M = pd.DataFrame(rows)
    M["종료_평균"] = M.종료_평균.map(hm)
    M.round(2).to_csv(f"{OUT}/DRT_반납자_시나리오_결과.csv", index=False, encoding="utf-8-sig")
    S = pd.DataFrame(search)
    S["종료_평균"] = S.종료_평균.map(hm)
    S.round(2).to_csv(f"{OUT}/DRT_반납자_필요차량_탐색.csv", index=False, encoding="utf-8-sig")
    cols = ["구분", "하루 이용자_명", "안", "거절률_%", "대기_평균", "대기_p95", "승차_평균", "승차_p95", "총시간_평균", "총시간_p95",
            "총시간_전체승객_평균", "총시간_전체승객_p95", "총시간_전체승객_반복SD", "가동률_%", "운행거리_km", "첫차만차_못탐_%", "다음차도_못탐_%"]
    print("\n[1·2] 대표 시나리오 + 민감도")
    print(M[cols].round(1).to_string(index=False))

    # ---------------- 3. 점심 증차 실험안
    print("\n[3] 점심 증차 실험안", flush=True)
    pk, hrows = [], []
    for n in REP + SENS:
        k = need[n]
        if not k or k < 3:
            continue
        base = summ(get(n, "DRT", k - 1))
        full = summ(get(n, "DRT", k))
        variants = [(f"{k - 1}대 종일", get(n, "DRT", k - 1), base), (f"{k}대 종일", get(n, "DRT", k), full)]
        for a, b in PEAKS:
            sh = [(V.OPEN, 1e9)] * (k - 1) + [(a, b)]
            rs = get(n, "DRT", k, shifts=sh)
            s = summ(rs)
            variants.append((f"{k - 1}대 + {int(a // 60)}~{int(b // 60)}시 1대", rs, s))
        for name, rs, s in variants:
            pk.append({"하루 이용자_명": n, "필요 대수(종일)": k, "안": name, "거절률_%": s["거절률_%"], "대기_평균": s["대기_평균"],
                       "시간대최대거절률_%": s["시간대최대거절률_%"], "하루 기준 충족": ok(s),
                       "시간대 기준 충족(모든 시간 거절 5% 이하)": ok(s) and s["시간대최대거절률_%"] <= 5, "운행거리_km": s["운행거리_km"]})
            for h in range(8, 20):
                q = np.mean([r["hour_req"][h] for r in rs])
                j = np.mean([r["hour_rej"][h] for r in rs])
                hrows.append({"하루 이용자_명": n, "안": name, "시": h, "호출_건": q, "거절_건": j, "거절률_%": j / q * 100 if q else 0.0})
    PK = pd.DataFrame(pk)
    PK.round(2).to_csv(f"{OUT}/DRT_점심증차_실험안.csv", index=False, encoding="utf-8-sig")
    HR = pd.DataFrame(hrows)
    HR.round(2).to_csv(f"{OUT}/DRT_점심증차_시간대별거절.csv", index=False, encoding="utf-8-sig")
    print(PK.round(1).to_string(index=False))

    # ---------------- 4. B01 포함 효과
    print("\n[4] B01 포함 효과", flush=True)
    bb = []
    for n in REP:
        k = need[n] or MAXF
        d0, d1 = summ(get(n, "DRT", k)), summ(get(n, "DRT", k, b01=True))
        f0, f1 = summ(get(n, "고정", 2)), summ(get(n, "고정", 2, loopkey="B01", b01=True))
        us = [V.users_n(n, rp, True) for rp in range(NREP)]
        b01_users = np.mean([sum(o == V.B01 for _, o, _, _ in u) for u in us])
        bb.append({"하루 이용자_명": n, "DRT 대수": k, "B01 이용자_명/일": b01_users, "B01 편도 호출_건/일": 2 * b01_users,
                   "DRT 운행거리_B01제외_km": d0["운행거리_km"], "DRT 운행거리_B01포함_km": d1["운행거리_km"],
                   "DRT 운행거리 증가_km": d1["운행거리_km"] - d0["운행거리_km"],
                   "DRT 운행거리 증가_B01 편도 1건당_km": (d1["운행거리_km"] - d0["운행거리_km"]) / max(1e-9, 2 * b01_users),
                   "DRT 거절률_제외/포함_%": f"{d0['거절률_%']:.1f} / {d1['거절률_%']:.1f}", "DRT 총시간_제외/포함": f"{d0['총시간_평균']:.1f} / {d1['총시간_평균']:.1f}",
                   "고정노선 순환_제외/포함_분": f"{V.D['cycle0']:.1f} / {V.D['cycle']:.1f}",
                   "고정노선 2대 총시간_제외_분": f0["총시간_평균"], "고정노선 2대 총시간_포함_분": f1["총시간_평균"],
                   "고정노선 2대 운행거리_제외/포함_km": f"{f0['운행거리_km']:.0f} / {f1['운행거리_km']:.0f}",
                   "고정노선 첫차만차_제외/포함_%": f"{f0['첫차만차_못탐_%']:.1f} / {f1['첫차만차_못탐_%']:.1f}"})
    BB = pd.DataFrame(bb)
    BB.round(2).to_csv(f"{OUT}/DRT_B01_포함효과.csv", index=False, encoding="utf-8-sig")
    print(BB.round(1).to_string(index=False))

    # ---------------- 5. 탄소용 거리 지표
    ccols = ["구분", "하루 이용자_명", "안", "운행거리_km", "빈차_km", "탑승_km", "빈차비율_%", "인km", "탑승편도_건", "탑승편도당_차량km", "직행_평균km",
             "평균탑승인원_빈차포함", "평균탑승인원_빈차제외", "택시대체_편도_%", "택시대체_km", "구간최대탑승_명"]
    M[ccols].round(3).to_csv(f"{OUT}/DRT_반납자_탄소용_거리지표.csv", index=False, encoding="utf-8-sig")
    print("\n[5] 탄소용 거리 지표")
    print(M[ccols].round(2).to_string(index=False))
    # ---------------- 6. 전체 반납자 조합 필요 대수(같은 기준, 조합표용)
    print("\n[6] 전체 조합 필요 대수", flush=True)
    for n in sorted(dem["하루 이용자_명"].unique()):
        if n not in need:
            need[n] = min_fleet(n)[0]
    full = {n: summ(get(n, "DRT", need[n])) for n in sorted(dem["하루 이용자_명"].unique())}
    dem["필요 차량_대(거절 5%·대기 15분, 공동 운영, B01 제외)"] = dem["하루 이용자_명"].map(need)
    dem["그때 거절률_%"] = dem["하루 이용자_명"].map(lambda n: round(full[n]["거절률_%"], 1))
    dem["그때 평균대기_분"] = dem["하루 이용자_명"].map(lambda n: round(full[n]["대기_평균"], 1))
    dem["대표 시나리오"] = dem["하루 이용자_명"].isin(REP)
    dem.to_csv(f"{OUT}/DRT_반납자기준_필요차량.csv", index=False, encoding="utf-8-sig")
    print(dem[["잔존율_%", "참여율_%", "이용빈도", "운행일", "하루 이용자_명", "하루 편도 호출_건", "필요 차량_대(거절 5%·대기 15분, 공동 운영, B01 제외)",
               "그때 거절률_%", "그때 평균대기_분"]].to_string(index=False))

    # ---------------- 7. 한계 이용자 재계산(같은 기준: B01 제외, 1대 단일 / 2대 공동 운영, 2대 권역 고정은 참고)
    print("\n[7] 한계 이용자 스캔", flush=True)
    GRID = [9, 18, 26, 35, 44, 53, 62, 71, 79, 88, 97, 106, 115, 124, 132]
    lim_rows = []
    for nv, pool, name in [(1, True, "DRT 1대"), (2, True, "DRT 2대(공동 운영)"), (2, False, "참고: DRT 2대 권역 고정")]:
        for n in GRID:
            s = summ(get(n, "DRT", nv, pool=pool))
            lim_rows.append({"안": name, "하루 이용자_명": n, "거절률_%": s["거절률_%"], "대기_평균": s["대기_평균"], "기준 충족": ok(s)})
            if s["거절률_%"] > 15:
                break
    LS = pd.DataFrame(lim_rows)
    LS.round(2).to_csv(f"{OUT}/DRT_한계이용자_스캔_B01제외.csv", index=False, encoding="utf-8-sig")
    LIM = {}
    for name, g in LS.groupby("안", sort=False):
        bad = g[~g["기준 충족"]]
        okg = g[g["하루 이용자_명"] < bad["하루 이용자_명"].min()] if len(bad) else g
        LIM[name] = int(okg["하루 이용자_명"].max()) if len(okg) else 0
    print(LS.round(1).to_string(index=False))
    print("한계 이용자(기준을 지키는 마지막 계산 지점):", LIM)
    # ---------------- 8. 최대 대기 민감도(통일 기준): 30분(기본) vs 45분, 1대·2대 공동 운영
    print("\n[8] 최대 대기 30/45분 민감도", flush=True)
    mw_rows = []
    for mw in (30.0, 45.0):
        for nv, name in [(1, "DRT 1대"), (2, "DRT 2대(공동 운영)")]:
            for n in GRID:
                s = summ(get(n, "DRT", nv, mw=mw))
                mw_rows.append({"최대 대기_분": int(mw), "안": name, "하루 이용자_명": n, "거절률_%": s["거절률_%"], "대기_평균": s["대기_평균"],
                                "총시간_전체승객_평균": s["총시간_전체승객_평균"], "운행거리_km": s["운행거리_km"], "기준 충족": ok(s)})
                if s["거절률_%"] > 15 or s["대기_평균"] > 18:
                    break
    MW = pd.DataFrame(mw_rows)
    MW.round(2).to_csv(f"{OUT}/DRT_최대대기민감도_B01제외.csv", index=False, encoding="utf-8-sig")
    MWL = []
    for (mw, name), g in MW.groupby(["최대 대기_분", "안"], sort=False):
        bad = g[~g["기준 충족"]]
        okg = g[g["하루 이용자_명"] < bad["하루 이용자_명"].min()] if len(bad) else g
        last = okg.iloc[-1] if len(okg) else None
        MWL.append({"최대 대기_분": mw, "안": name, "한계 이용자_명": int(last["하루 이용자_명"]) if last is not None else 0,
                    "한계에서 거절·대기": f"{last['거절률_%']:.1f}%·{last['대기_평균']:.1f}분" if last is not None else "",
                    "다음 지점_명": int(bad.iloc[0]["하루 이용자_명"]) if len(bad) else None,
                    "다음 지점 거절·대기": f"{bad.iloc[0]['거절률_%']:.1f}%·{bad.iloc[0]['대기_평균']:.1f}분" if len(bad) else ""})
    MWL = pd.DataFrame(MWL)
    MWL.to_csv(f"{OUT}/DRT_최대대기별_한계이용자_B01제외.csv", index=False, encoding="utf-8-sig")
    print(MW.round(1).to_string(index=False))
    print(MWL.to_string(index=False))
    pickle.dump({"M": M, "S": S, "PK": PK, "HR": HR, "BB": BB, "need": need, "dem": dem, "LS": LS, "LIM": LIM, "MW": MW, "MWL": MWL},
                open(f"{INTERIM}/DRT_반납자_결과.pkl", "wb"))
    POOL.close()
    print(f"\n완료 (누적 {(time.time() - t0) / 60:.1f}분)")
