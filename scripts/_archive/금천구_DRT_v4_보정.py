"""[이전 버전, 2026-10-12 대체] B01 처리(DRT 포함·고정노선 제외)와 차량 기준이 메인과 달라 금천구_DRT_반납자시뮬.py로 대체.
출력은 outputs/_이전_v4보정_20261012/ 로 옮김.

DRT v4 보정: (a) 같은 서비스 수준 비교, (b) 거절 승객 포함 총시간, 탄소용 거리 지표, 반납자 수요 조합별 필요 차량, 점심 전후 증차.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_v4_보정.py`  (선행: 금천구_DRT_v4_준비.py, 금천구_반납자수요.py)
엔진: 금천구_DRT_v4_시뮬.py와 같은 v4 가정·같은 이용자 표본(같은 시드). 최대 대기 30분(기본). 지표만 더했다.
  - 차량 수: 1대 = 단일, 2대 = 권역 고정(새 권역), 3대 이상 = 공동 운영(대기 장소 A·B 교대). v4 필요차량수와 같은 구분이다.
(a) 같은 서비스 수준: 이용률 1·3·5%마다 거절률 5% 이하가 되는 최소 DRT 차량 수를 찾고, 고정노선 2대(B01 제외)와 총시간·운행거리 비교
(b) 전체 승객 총시간: 거절 편도는 '30분 대기 후 택시(승차시간 = 직행)'. 가는 호출이 거절된 승객의 귀가 편도도 같은 규칙(귀가 호출이 생기지 않으므로)
    고정노선 B01 미서비스 편도도 같은 규칙. 분모 = 이용자 x 2(왕복 편도 전부)
탄소용 거리 지표(배출계수는 다음 단계): 총·빈 차·탑승 운행거리, 인-km, 탑승 편도 1건당 차량 km, 평균 탑승 인원(거리 가중, 빈 차 포함/제외),
    거절 택시 km(직행 거리). 고정노선 빈 차/탑승 구분은 시간표(차량 j가 8시 + j x 배차간격에 출발) 위에 승객을 처음 오는 차량에 태워 계산
반납자 수요: outputs/DRT_반납자기준_수요추정.csv의 하루 이용자 값마다 평균 대기 15분·거절률 5% 이하 최소 차량 수
점심 전후 증차: 필요 차량 k대(3대 이상)인 수요에서 'k-1대 종일 + 1대 10~14시'가 기준을 지키는지
체크포인트: 시나리오마다 data/interim/DRT_v4_보정_runs/ 에 저장, 재실행 시 완료분 건너뜀
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
RUNS = f"{INTERIM}/DRT_v4_보정_runs"
POP70, NREP, TAXI_WAIT, KMH = V.POP70, V.NREP, 30.0, 20.0
RATES = [0.01, 0.03, 0.05]
MAXFLEET = 16
PEAK = (600.0, 840.0)  # 10~14시
POOL = None


def task(a):
    rate, rep, mode, nveh, shifts, loopkey = a
    us = V.users(rate, rep, V.V4)
    return V.run_drt(us, nveh, V.V4, shifts) if mode == "DRT" else V.run_fixed(us, nveh, loopkey)


def fname(rate, mode, nveh, shifts=None, loopkey=None):
    sh = "" if not shifts else "_" + "+".join(f"{int(a)}-{int(b)}" for a, b in shifts if b < 1e8)
    return f"{RUNS}/r{rate * 1000:07.3f}_{mode}{nveh}대_{loopkey or ''}{sh}.pkl"


def get(rate, mode, nveh, shifts=None, loopkey=None):
    fn = fname(rate, mode, nveh, shifts, loopkey)
    if not os.path.exists(fn):
        t1 = time.time()
        rs = POOL.map(task, [(rate, rp, mode, nveh, shifts, loopkey) for rp in range(NREP)], chunksize=5)
        pickle.dump(rs, open(fn + ".tmp", "wb"))
        os.replace(fn + ".tmp", fn)
        print(f"  계산 {os.path.basename(fn)}: {time.time() - t1:.0f}초", flush=True)
    return pickle.load(open(fn, "rb"))


def summ(rs):
    w, rd = (np.concatenate([r[k] for r in rs]) for k in ("wait", "ride"))
    served = w + rd
    # (b) 거절·미서비스 편도 = 30분 대기 + 택시(직행)
    alt = np.concatenate([np.asarray(r["rej_dir"] + r["rej_ret_dir"] + r.get("miss_dir", []), float) for r in rs])
    allp = np.r_[served, TAXI_WAIT + alt]
    nuser = sum(r["n_user"] for r in rs)
    assert len(allp) == 2 * nuser, (len(allp), nuser)
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    km, ke, kl, pkm = (np.mean([r[k] for r in rs]) for k in ("km", "km_empty", "km_loaded", "pkm"))
    legs = len(served) / len(rs)
    return {"이용자_명": int(round(nuser / len(rs))), "거절률_%": rej / req * 100 if req else 0.0, "대기_평균": w.mean(), "승차_평균": rd.mean(),
            "총시간_탑승승객_평균": served.mean(), "총시간_전체승객_평균": allp.mean(), "총시간_전체승객_p95": np.percentile(allp, 95),
            "택시대체_편도_%": len(alt) / len(allp) * 100,
            "운행거리_km": km, "빈차_km": ke, "탑승_km": kl, "빈차비율_%": ke / km * 100, "인km": pkm,
            "탑승편도_건": legs, "탑승편도당_차량km": km / legs, "이용자당_차량km(왕복)": km / (nuser / len(rs)),
            "평균탑승인원_빈차포함": pkm / km, "평균탑승인원_빈차제외": pkm / kl,
            "택시대체_km": np.mean([(sum(r["rej_dir"]) + sum(r["rej_ret_dir"]) + sum(r.get("miss_dir", []))) * KMH / 60 for r in rs]),
            "구간최대탑승_명": np.max([r.get("maxload", np.nan) for r in rs]) if "maxload" in rs[0] else np.nan,
            "종료_평균": np.mean([r["end"] for r in rs])}


def ok(s, wait=True):
    return s["거절률_%"] <= 5 and (s["대기_평균"] <= 15 or not wait)


def min_fleet(rate, wait=True):
    """기준을 처음 만족하는 차량 수(1대부터 차례로, 차량이 늘면 서비스가 나아진다고 가정)"""
    for nv in range(1, MAXFLEET + 1):
        s = summ(get(rate, "DRT", nv))
        if ok(s, wait):
            return nv, s
    return None, s


lab = lambda nv: "DRT 1대" if nv == 1 else "DRT 2대(새 권역)" if nv == 2 else f"DRT {nv}대(공동 운영)"
hm = lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"

if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    pd.set_option("display.width", 260)
    POOL = mp.Pool(11, initializer=V.init)
    V.init()
    t0 = time.time()

    # ---------------- (a)(b) + 탄소 지표: 이용률 1·3·5%
    print("[1] 이용률 1·3·5%: DRT 1·2대, 거절률 5% 이하 최소 차량, 고정노선", flush=True)
    rows = []
    for rate in RATES:
        fx = {(nv, lk): summ(get(rate, "고정", nv, loopkey=lk)) for nv in (1, 2) for lk in ("noB01", "B01")}
        k, sk = min_fleet(rate, wait=False)
        plans = [(lab(1), summ(get(rate, "DRT", 1)), "DRT"), (lab(2), summ(get(rate, "DRT", 2)), "DRT"), (lab(k) + " ← 거절률 5% 이하 최소", sk, "DRT 최소")]
        plans += [("고정노선 1대(B01 제외)", fx[1, "noB01"], "고정"), ("고정노선 2대(B01 제외)", fx[2, "noB01"], "고정"),
                  ("참고: 고정노선 1대(B01 포함)", fx[1, "B01"], "고정참고"), ("참고: 고정노선 2대(B01 포함)", fx[2, "B01"], "고정참고")]
        for name, s, kind in plans:
            rows.append({"이용률_%": rate * 100, "안": name, "구분": kind, "차량_대": k if kind == "DRT 최소" else None, **s})
    A = pd.DataFrame(rows)
    A["종료_평균"] = A.종료_평균.map(hm)
    cols_b = ["이용률_%", "안", "이용자_명", "거절률_%", "택시대체_편도_%", "대기_평균", "승차_평균", "총시간_탑승승객_평균", "총시간_전체승객_평균", "총시간_전체승객_p95",
              "운행거리_km", "종료_평균"]
    A[cols_b].round(2).to_csv(f"{OUT}/DRT_v4_보정_전체승객총시간.csv", index=False, encoding="utf-8-sig")
    cmp_ = A[A.구분.isin(["DRT 최소"]) | (A.안 == "고정노선 2대(B01 제외)")]
    cols_a = ["이용률_%", "안", "이용자_명", "거절률_%", "대기_평균", "승차_평균", "총시간_탑승승객_평균", "총시간_전체승객_평균", "운행거리_km", "탑승편도당_차량km",
              "구간최대탑승_명"]
    cmp_[cols_a].round(2).to_csv(f"{OUT}/DRT_v4_보정_같은서비스수준.csv", index=False, encoding="utf-8-sig")
    cols_c = ["이용률_%", "안", "이용자_명", "운행거리_km", "빈차_km", "탑승_km", "빈차비율_%", "인km", "탑승편도_건", "탑승편도당_차량km", "이용자당_차량km(왕복)",
              "평균탑승인원_빈차포함", "평균탑승인원_빈차제외", "택시대체_편도_%", "택시대체_km", "구간최대탑승_명"]
    print("\n[(b)] 전체 승객 기준 총시간 (거절·미서비스 편도 = 30분 대기 + 택시 직행)")
    print(A[cols_b].round(1).to_string(index=False))
    print("\n[(a)] 같은 서비스 수준: 거절률 5% 이하 최소 DRT vs 고정노선 2대(B01 제외)")
    print(cmp_[cols_a].round(1).to_string(index=False))

    # ---------------- 반납자 수요 조합별 필요 차량 (대기 15분·거절 5%)
    print("\n[2] 반납자 수요 조합별 필요 차량", flush=True)
    dem = pd.read_csv(f"{OUT}/DRT_반납자기준_수요추정.csv", encoding="utf-8-sig")
    need, carbon_extra = {}, []
    for n in sorted(dem["하루 이용자_명"].unique()):
        k, s = min_fleet(n / POP70)
        need[n] = (k, s)
        carbon_extra.append({"이용률_%": n / POP70 * 100, "안": f"반납자 수요 {n}명/일: " + lab(k), **s})
    dem["필요 차량_대"] = dem["하루 이용자_명"].map(lambda n: need[n][0])
    dem["그때 거절률_%"] = dem["하루 이용자_명"].map(lambda n: round(need[n][1]["거절률_%"], 1))
    dem["그때 평균대기_분"] = dem["하루 이용자_명"].map(lambda n: round(need[n][1]["대기_평균"], 1))
    dem["그때 운행거리_km"] = dem["하루 이용자_명"].map(lambda n: round(need[n][1]["운행거리_km"]))
    dem.to_csv(f"{OUT}/DRT_반납자기준_필요차량.csv", index=False, encoding="utf-8-sig")
    print(dem[["잔존율_%", "참여율_%", "이용빈도", "운행일", "하루 이용자_명", "필요 차량_대", "그때 거절률_%", "그때 평균대기_분", "그때 운행거리_km"]].to_string(index=False))

    # ---------------- 점심 전후 증차 + 시간대별 거절
    print("\n[3] 점심 전후 증차(10~14시 1대) 검토", flush=True)
    pk, hrows = [], []
    for n in sorted(need):
        k = need[n][0]
        if k is None or k < 3:
            continue
        rate = n / POP70
        base = summ(get(rate, "DRT", k - 1))
        sh = [(V.OPEN, 1e9)] * (k - 1) + [PEAK]
        rp = get(rate, "DRT", k, shifts=sh)
        peak = summ(rp)
        pk.append({"하루 이용자_명": n, "필요 차량(종일)_대": k, "종일 k대 거절률_%": need[n][1]["거절률_%"], "종일 k대 대기_분": need[n][1]["대기_평균"],
                   "종일 k대 운행거리_km": need[n][1]["운행거리_km"],
                   "k-1대 거절률_%": base["거절률_%"], "k-1대 대기_분": base["대기_평균"],
                   "k-1대+점심1대 거절률_%": peak["거절률_%"], "k-1대+점심1대 대기_분": peak["대기_평균"], "k-1대+점심1대 운행거리_km": peak["운행거리_km"],
                   "점심증차로 기준 충족": ok(peak)})
        for tag, rs_, nv in [(f"{k - 1}대 종일", get(rate, "DRT", k - 1), k - 1), (f"{k}대 종일", get(rate, "DRT", k), k), (f"{k - 1}대+점심 1대", rp, k)]:
            for h in range(8, 21):
                q = np.mean([r["hour_req"][h] for r in rs_])
                j = np.mean([r["hour_rej"][h] for r in rs_])
                hrows.append({"하루 이용자_명": n, "안": tag, "시": h, "호출_건": q, "거절_건": j, "거절률_%": j / q * 100 if q else 0.0})
    pk = pd.DataFrame(pk)
    pk.round(2).to_csv(f"{OUT}/DRT_v4_점심증차_검토.csv", index=False, encoding="utf-8-sig")
    hr = pd.DataFrame(hrows)
    hr.round(2).to_csv(f"{OUT}/DRT_v4_수요별_시간대별거절.csv", index=False, encoding="utf-8-sig")
    print(pk.round(1).to_string(index=False))

    # ---------------- 탄소용 거리 지표 모음
    C = pd.concat([A, pd.DataFrame(carbon_extra)], ignore_index=True)
    C[cols_c].round(3).to_csv(f"{OUT}/DRT_v4_탄소용_거리지표.csv", index=False, encoding="utf-8-sig")
    print("\n[4] 탄소용 거리 지표 (이용률 1·3·5%)")
    print(A[cols_c].round(2).to_string(index=False))
    pickle.dump({"A": A, "dem": dem, "pk": pk, "hr": hr, "C": C}, open(f"{INTERIM}/DRT_v4_보정_결과.pkl", "wb"))
    POOL.close()
    print(f"\n완료 (누적 {(time.time() - t0) / 60:.1f}분)")
