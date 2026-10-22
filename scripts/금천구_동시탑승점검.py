"""동시 탑승 최대 인원 점검: 단계표 대수 기준 7개 시나리오를 같은 시드로 다시 돌려 차량 한 대에 동시에 탄 최대 인원과 10명 초과 운행 시간 비율을 기록.

실행: 프로젝트 루트에서 `python -u scripts/금천구_동시탑승점검.py`
기존 체크포인트와 결과는 그대로 두고 새 체크포인트 data/interim/동시탑승_runs/ 에 저장. 출력 = outputs/동시탑승_점검.csv
재현 확인: 다시 돌린 결과의 대기·차내·운행거리가 기존 체크포인트(반납자 시뮬, 83명·3대는 검토2)와 300회 모두 같은지 비교
10명 초과 시간 비율 = 승객 10명 초과로 달린 주행시간 ÷ 전체 주행시간(빈 차 주행 포함)
"""
import importlib
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
V = importlib.import_module("금천구_DRT_v4_시뮬")
INTERIM, OUT = "data/interim", "outputs"
RUNS = f"{INTERIM}/동시탑승_runs"
PTS = [(24, 1), (41, 2), (69, 2), (83, 3), (98, 3), (138, 4), (276, 8)]


def task(a):
    n, rep, k = a
    return V.run_drt(V.users_n(n, rep, False), k, V.V4, None, True)


def old(n, k):
    a = f"{INTERIM}/DRT_반납자_runs/n{n:03d}_DRT{k}대{'_공동' if k > 1 else ''}.pkl"
    b = f"{INTERIM}/검토2_runs/n{n:03d}_DRT{k}대_기본.pkl"
    return pickle.load(open(a if os.path.exists(a) else b, "rb"))


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    pool = mp.Pool(11, initializer=V.init)
    rows = []
    for n, k in PTS:
        fn = f"{RUNS}/n{n:03d}_DRT{k}대.pkl"
        if not os.path.exists(fn):
            rs = pool.map(task, [(n, rp, k) for rp in range(V.NREP)], chunksize=5)
            pickle.dump(rs, open(fn + ".tmp", "wb"))
            os.replace(fn + ".tmp", fn)
        rs = pickle.load(open(fn, "rb"))
        o = old(n, k)
        same = all(np.allclose(a["wait"], b["wait"]) and np.allclose(a["ride"], b["ride"]) and np.isclose(a["km"], b["km"]) for a, b in zip(rs, o))
        mx = np.array([r["max_onboard"] for r in rs])
        t10, tm = sum(r["t_over10"] for r in rs), sum(r["t_move"] for r in rs)
        rows.append({"하루 이용자_명": n, "DRT 대수": k, "기존 결과 재현(300회)": same, "최대 동시 탑승_명": int(mx.max()),
                     "반복별 최대의 중앙값_명": float(np.median(mx)), "10명 초과가 나온 반복_%": (mx > 10).mean() * 100,
                     "10명 초과 운행 시간 비율_%": t10 / tm * 100, "정원": V.CAP})
        print(rows[-1], flush=True)
    pool.close()
    T = pd.DataFrame(rows)
    T.round(4).to_csv(f"{OUT}/동시탑승_점검.csv", index=False, encoding="utf-8-sig")
    print(T.to_string(index=False))
