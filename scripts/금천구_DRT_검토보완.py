"""팀원 검토 보완: (1) DRT vs 고정노선 비교표 재정리, (2) 차량 1~8대 한계 이용자, (3) 단계표, (5) O-D 민감도(2대 한계), (6) 운행거리 정의 점검.

실행: 프로젝트 루트에서 `python -u scripts/금천구_DRT_검토보완.py`  (선행: 금천구_DRT_반납자시뮬.py)
기존 결과 파일은 덮어쓰지 않음. 출력 = outputs/검토보완_*.csv, 그림 outputs/검토보완_대수별_한계이용자.png
기준(통일): 공동 운영(1대 = 단일), B01 제외, 거절률 5% 이하 그리고 평균 대기 15분 이하, 최대 대기 30분, 300회
  - 한계 이용자 = 계산 격자(하루 이용자 = round(8,817 x 0.1% x k), 약 9명 간격)에서 기준을 지키는 마지막 지점
  - 기본 이용자 표본·체크포인트는 금천구_DRT_반납자시뮬.py의 것을 그대로 읽고(같은 이름이면 재사용), 없는 격자점만 새로 계산해 같은 폴더에 더한다.
  - O-D 민감도: 가는 호출 시각 분포를 '생활이동 목적별 2026년 8월 시흥1동 도착 70세 이상(귀가 제외) 평일 평균' 8~16시 분포로 바꾼다.
    체크포인트는 data/interim/검토보완_runs/ (기본안 체크포인트와 섞이지 않게)
"""
import importlib
import multiprocessing as mp
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
B = importlib.import_module("금천구_DRT_반납자시뮬")
V = B.V
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, INTERIM = "outputs", "data/interim"
RUNS_OD = f"{INTERIM}/검토보완_runs"
GRID = [int(round(V.POP70 * 0.001 * k)) for k in range(1, 46)]  # 9, 18, 26, ... 397
OD_HOURS = None


def task_od(a):
    n, rep, nveh, hours = a
    return V.run_drt(V.users_n(n, rep, False, hours=hours), nveh, V.V4, None, True)


def get_od(n, nveh, hours):
    fn = f"{RUNS_OD}/OD도착분포_n{n:03d}_DRT{nveh}대_공동.pkl"
    if not os.path.exists(fn):
        t1 = time.time()
        rs = B.POOL.map(task_od, [(n, rp, nveh, hours) for rp in range(B.NREP)], chunksize=5)
        pickle.dump(rs, open(fn + ".tmp", "wb"))
        os.replace(fn + ".tmp", fn)
        print(f"  계산 {os.path.basename(fn)}: {time.time() - t1:.0f}초", flush=True)
    return pickle.load(open(fn, "rb"))


def diag(rs):
    """기준 판정 + 어느 기준이 걸리는지 + 거절 시간대"""
    s = B.summ(rs)
    hq = np.sum([r["hour_req"] for r in rs], axis=0)
    hj = np.sum([r["hour_rej"] for r in rs], axis=0)
    share1013 = hj[10:14].sum() / hj.sum() * 100 if hj.sum() else np.nan
    peak = int(np.argmax([hj[h] / hq[h] if hq[h] else 0 for h in range(24)]))
    return s, {"거절률_%": s["거절률_%"], "대기_평균": s["대기_평균"], "기준 충족": B.ok(s),
               "걸린 기준": "" if B.ok(s) else ("거절·대기 둘 다" if s["거절률_%"] > 5 and s["대기_평균"] > 15 else "거절률" if s["거절률_%"] > 5 else "평균 대기"),
               "거절 중 10~13시_%": share1013, "거절률 최대 시": peak, "가동률_%": s["가동률_%"], "운행거리_km": s["운행거리_km"]}


def scan(nv, start_k, getter):
    rows, k = [], start_k
    while k < len(GRID):
        n = GRID[k]
        s, d = diag(getter(n, nv))
        rows.append({"차량_대": nv, "하루 이용자_명": n, **d})
        if not d["기준 충족"]:
            break
        k += 1
    return rows


if __name__ == "__main__":
    os.makedirs(RUNS_OD, exist_ok=True)
    pd.set_option("display.width", 250)
    B.POOL = mp.Pool(11, initializer=V.init)
    V.init()

    # ---------------- (1) DRT vs 고정노선 비교표
    M = pd.read_csv(f"{OUT}/DRT_반납자_시나리오_결과.csv", encoding="utf-8-sig")
    rep = M[M.구분 == "대표(반납자)"]
    rows = []
    for n, g in rep.groupby("하루 이용자_명"):
        d = g[g.종류 == "DRT 필요"].iloc[0]
        f = g[g.안 == "고정노선 2대(B01 제외, 정원 12)"].iloc[0]
        rows.append({"하루 이용자_명": n, "DRT 대수": int(d.차량_대), "DRT 평균_분": d.총시간_전체승객_평균, "DRT p95_분": d.총시간_전체승객_p95,
                     "DRT 대기/차내_분": f"{d.대기_평균:.1f} / {d.승차_평균:.1f}", "DRT 거절_%": d["거절률_%"],
                     "고정노선 대수": 2, "고정노선 평균_분": f.총시간_전체승객_평균, "고정노선 p95_분": f.총시간_전체승객_p95,
                     "고정노선 대기/차내_분": f"{f.대기_평균:.1f} / {f.승차_평균:.1f}", "고정노선 첫차만차_%": f["첫차만차_못탐_%"],
                     "차이 평균_분(고정-DRT)": f.총시간_전체승객_평균 - d.총시간_전체승객_평균,
                     "차이 p95_분(고정-DRT)": f.총시간_전체승객_p95 - d.총시간_전체승객_p95,
                     "총시간 기준": "대기+차내(도보 제외), DRT 거절 편도는 30분 대기+택시 직행"})
    C1 = pd.DataFrame(rows)
    C1.round(2).to_csv(f"{OUT}/검토보완_DRT_vs_고정노선.csv", index=False, encoding="utf-8-sig")
    print("[1] DRT vs 고정노선 (대기+차내, 도보 제외)")
    print(C1.round(1).to_string(index=False))

    # ---------------- (2) 차량 1~8대 한계 이용자
    print("\n[2] 차량 대수별 한계 이용자", flush=True)
    rows, k0 = [], 0
    for nv in range(1, 9):
        r = scan(nv, k0, lambda n, v: B.get(n, "DRT", v))
        rows += r
        ok_rows = [x for x in r if x["기준 충족"]]
        k0 = GRID.index(ok_rows[-1]["하루 이용자_명"]) if ok_rows else k0
    SC = pd.DataFrame(rows)
    SC.round(2).to_csv(f"{OUT}/검토보완_대수별_스캔.csv", index=False, encoding="utf-8-sig")
    lim = []
    for nv, g in SC.groupby("차량_대"):
        okg, bad = g[g["기준 충족"]], g[~g["기준 충족"]]
        L, nx = okg.iloc[-1], (bad.iloc[0] if len(bad) else None)
        lim.append({"차량_대": nv, "한계 이용자_명": int(L["하루 이용자_명"]), "한계에서 거절_%": L["거절률_%"], "한계에서 대기_분": L["대기_평균"],
                    "다음 지점_명": int(nx["하루 이용자_명"]) if nx is not None else None,
                    "다음 지점 거절_%": nx["거절률_%"] if nx is not None else None, "다음 지점 대기_분": nx["대기_평균"] if nx is not None else None,
                    "먼저 걸린 기준": nx["걸린 기준"] if nx is not None else "", "다음 지점 거절 중 10~13시_%": nx["거절 중 10~13시_%"] if nx is not None else None,
                    "다음 지점 가동률_%": nx["가동률_%"] if nx is not None else None})
    LIM = pd.DataFrame(lim)
    LIM["1대 늘릴 때 증가_명"] = LIM["한계 이용자_명"].diff()
    LIM["차량 1대당 이용자_명"] = LIM["한계 이용자_명"] / LIM["차량_대"]
    LIM.round(2).to_csv(f"{OUT}/검토보완_대수별_한계이용자.csv", index=False, encoding="utf-8-sig")
    print(LIM.round(1).to_string(index=False))
    s110, d110 = diag(B.get(110, "DRT", 3))
    print(f"  확인: 하루 110명·3대 -> 거절 {d110['거절률_%']:.1f}%, 대기 {d110['대기_평균']:.1f}분, 기준 {'충족' if d110['기준 충족'] else '미달'}"
          " (110은 계산 격자점이 아닌 반납자 조합 값)")

    # ---------------- (3) 단계표
    dem = pd.read_csv(f"{OUT}/DRT_반납자기준_필요차량.csv", encoding="utf-8-sig")
    st, prev = [], 0
    for _, r in LIM.iterrows():
        nv, L = int(r.차량_대), int(r["한계 이용자_명"])
        combos = dem[(dem["하루 이용자_명"] > prev) & (dem["하루 이용자_명"] <= L)]
        st.append({"단계": f"{nv}대", "처리 범위(하루 이용자)": f"{prev + 1}~{L}명", "전환 조건(다음 단계로)": f"평일 하루 이용자 {L}명 초과가 지속(예: 4주 평균) 또는 거절률 5%·평균 대기 15분 초과",
                   "해당 반납자 조합 수": len(combos), "해당 하루 이용자 값": ", ".join(map(str, sorted(combos["하루 이용자_명"].unique())))})
        prev = L
        if L >= dem["하루 이용자_명"].max():
            break
    ST = pd.DataFrame(st)
    ST.to_csv(f"{OUT}/검토보완_단계표.csv", index=False, encoding="utf-8-sig")
    print("\n[3] 단계표")
    print(ST.to_string(index=False))

    # ---------------- (5) O-D 민감도: 시흥1동 도착 70세 이상 한 달 분포
    print("\n[5] O-D 민감도(2대 한계)", flush=True)
    arr = pd.read_csv(f"{OUT}/검증_시흥1동도착_70세_시간대.csv", encoding="utf-8-sig", index_col=0)["도착70_귀가제외_평일평균"]
    hours = arr.loc[8:16]
    base = V.HOURS[V.HOURS.index < 17]
    hcmp = pd.DataFrame({"기본(O-D 8/31 60세 이상 가는 출발)_%": base / base.sum() * 100, "민감도(8월 시흥1동 도착 70세 이상)_%": hours / hours.sum() * 100})
    od = scan(2, 0, lambda n, v: get_od(n, v, hours))
    OD = pd.DataFrame(od)
    basec = SC[SC.차량_대 == 2][["하루 이용자_명", "거절률_%", "대기_평균", "기준 충족"]].rename(columns=lambda c: f"기본_{c}" if c != "하루 이용자_명" else c)
    OD = OD.merge(basec, on="하루 이용자_명", how="left")
    OD.round(2).to_csv(f"{OUT}/검토보완_OD민감도_2대한계.csv", index=False, encoding="utf-8-sig")
    hcmp.round(1).to_csv(f"{OUT}/검토보완_OD민감도_시간분포.csv", encoding="utf-8-sig")
    print(hcmp.round(1).to_string())
    print(OD[["하루 이용자_명", "거절률_%", "대기_평균", "기준 충족", "걸린 기준", "기본_거절률_%", "기본_대기_평균", "기본_기준 충족"]].round(1).to_string(index=False))

    # ---------------- (6) 운행거리 정의 점검
    C = pd.read_csv(f"{OUT}/DRT_반납자_탄소용_거리지표.csv", encoding="utf-8-sig")
    T = pd.read_csv(f"{OUT}/탄소_시나리오별_배출량.csv", encoding="utf-8-sig")
    chk = C[["구분", "하루 이용자_명", "안", "운행거리_km", "빈차_km", "탑승_km"]].copy()
    chk["빈차+탑승-운행거리_km"] = chk.빈차_km + chk.탑승_km - chk.운행거리_km
    t = T[(T.방식 == "A") & T.자가용.str.contains("준중형") & T["승합 차종"].str.contains("경유")][["하루 이용자_명", "DRT 운행 km/일"]].drop_duplicates("하루 이용자_명")
    m = M[M.종류 == "DRT 필요"][["하루 이용자_명", "운행거리_km"]].merge(t, on="하루 이용자_명")
    m["차이_km"] = m["DRT 운행 km/일"] - m.운행거리_km
    chk.round(3).to_csv(f"{OUT}/검토보완_운행거리정의_점검.csv", index=False, encoding="utf-8-sig")
    print("\n[6] 빈차+탑승 = 운행거리 최대 오차 km:", round(chk["빈차+탑승-운행거리_km"].abs().max(), 6), "| 탄소 계산 거리 = 시뮬 운행거리 최대 차이 km:",
          round(m.차이_km.abs().max(), 6))

    # ---------------- 그림
    SURF, INK2, GRIDC = "#fcfcfb", "#52514e", "#e6e5e0"
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150, facecolor=SURF)
    axes[0].plot(LIM.차량_대, LIM["한계 이용자_명"], marker="o", color="#2a78d6", lw=2)
    for _, r in LIM.iterrows():
        axes[0].annotate(f"{int(r['한계 이용자_명'])}명", (r.차량_대, r["한계 이용자_명"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set(title="차량 대수별 한계 이용자 (거절 5%·대기 15분, 공동 운영)", xlabel="차량 (대)", ylabel="하루 이용자 (명)")
    axes[1].bar(LIM.차량_대[1:], LIM["1대 늘릴 때 증가_명"][1:], color="#1baf7a", edgecolor=SURF, lw=2)
    axes[1].set(title="차량 1대를 늘릴 때 한계 이용자 증가", xlabel="차량 (대)", ylabel="명")
    for a in axes:
        a.set_facecolor(SURF)
        for s_ in ["top", "right"]:
            a.spines[s_].set_visible(False)
        a.grid(axis="y", color=GRIDC, lw=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/검토보완_대수별_한계이용자.png", facecolor=SURF)
    B.POOL.close()
    pickle.dump({"C1": C1, "SC": SC, "LIM": LIM, "ST": ST, "OD": OD, "hcmp": hcmp}, open(f"{INTERIM}/검토보완_결과.pkl", "wb"))
    print("완료")
