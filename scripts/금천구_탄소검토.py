"""탄소 계산 검토: 단계표 대수 기준 시나리오 점(24·41·69·83·98·138·276명) 재정리, 전비 3.9 기본, 4.1·3.3 민감도, 택시 계수 민감도.

실행: 프로젝트 루트에서 `python scripts/금천구_탄소계산.py` 이후 `python scripts/금천구_탄소검토.py`
시뮬레이션은 새로 돌리지 않고 기존 체크포인트만 읽는다(data/interim/DRT_반납자_runs/, 83명·3대는 data/interim/검토2_runs/)
기존 결과 파일은 덮어쓰지 않음. 출력 = outputs/탄소검토_*.csv
계산(금천구_탄소계산.py와 같은 정의, 방식 A, 자가용 = 준중형 휘발유)
  - 자가용 kgCO2/일 = 모든 편도의 직행 도로거리(km) ÷ 연비(15.3km/L) x 휘발유 2.178kgCO2/L
  - DRT kgCO2/일 = 빈 차 포함 운행거리 x 승합 배출계수 + 거절 편도 택시 직행 거리 x 택시 계수
  - 대수 = 단계표(검토보완_단계표.csv): 1대 ~26명, 2대 ~79명, 3대 ~123명, 4대 ~168명, … 8대 ~291명
  - 전비 3.9 기본, 4.1·3.3 민감도. 3.9km/kWh = 제조사 복합전비(스타리아 투어러 11인승 EV), 한국에너지공단 확인 전
택시 민감도: 기본 = 표시 CO2 137g/km / 방식 A = 9.5km/L ÷ LPG kgCO2/L. LPG(부탄) 순발열량 45.5MJ/kg·탄소계수 18.094tC/TJ는
  공식값(에너지법 열량환산기준·IPCC 2006)이나 kg -> L 밀도(0.584kg/L)는 공식 고시값이 아닌 일반 물성값으로 둔 민감도 / 공차 = 직행 거리만큼 빈 차 이동을 더한 경우(2배)
"""
import importlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
CB = importlib.import_module("금천구_탄소계산")

OUT, INTERIM, KMH = "outputs", "data/interim", 20.0
PTS = [(24, 1), (41, 2), (69, 2), (83, 3), (98, 3), (138, 4), (276, 8)]
CAR = CB.CARS[0]
DIESEL = CB.VANS[0]
EFF = {"4.1": 4.1, "3.9": 3.9, "3.3": 3.3}
LPG_KG = 45.5 * 1e-6 * 18.094 * 44 / 12 * 1000  # kgCO2/kg(부탄)
LPG_DENS = 0.584  # kg/L, 일반 물성값(공식 아님), 민감도 전용
TAXI = {"표시 CO2 137g/km(기본)": 0.137, "방식 A: 9.5km/L ÷ LPG(밀도 0.584kg/L 가정)": LPG_KG * LPG_DENS / 9.5}


def load(n, k):
    a = f"{INTERIM}/DRT_반납자_runs/n{n:03d}_DRT{k}대{'_공동' if k > 1 else ''}.pkl"
    b = f"{INTERIM}/검토2_runs/n{n:03d}_DRT{k}대_기본.pkl"
    fn = a if os.path.exists(a) else b
    return pickle.load(open(fn, "rb")), fn


def metrics(rs):
    km = np.mean([r["km"] for r in rs])
    taxi = np.mean([(sum(r["rej_dir"]) + sum(r["rej_ret_dir"])) * KMH / 60 for r in rs])
    direct = np.mean([sum(r["dir"]) * KMH / 60 for r in rs]) + taxi  # 탑승 편도 직행 + 거절 편도 직행
    req, rej = sum(r["요청"] for r in rs), sum(r["거절"] for r in rs)
    return km, taxi, direct, rej / req * 100


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    rows, trows = [], []
    for n, k in PTS:
        rs, fn = load(n, k)
        km, taxi, direct, rejp = metrics(rs)
        car = direct * CB.ef(CAR, "A")
        row = {"하루 이용자_명": n, "DRT 대수(단계표)": k, "체크포인트": os.path.relpath(fn).replace("\\", "/"), "거절률_%": rejp,
               "DRT 운행 km/일": km, "거절 택시 km/일": taxi, "자가용 직행 km/일": direct, "자가용 kgCO2/일": car}
        vans = {"경유 승합": CB.ef(DIESEL, "A")} | {f"전기 {e}": CB.GRID / v for e, v in EFF.items()}
        for name, efv in vans.items():
            drt = km * efv + taxi * TAXI["표시 CO2 137g/km(기본)"]
            row[f"{name} kgCO2/일"] = drt
            row[f"{name} 증감_kg"] = drt - car
            row[f"{name} 증감_%"] = (drt - car) / car * 100
        rows.append(row)
        for tname, tef in TAXI.items():
            for dead, mult in [("공차 없음(기본)", 1.0), ("공차 = 직행만큼 추가", 2.0)]:
                tr = {"하루 이용자_명": n, "택시 계수": tname, "택시 kg/km": tef, "택시 공차": dead}
                for e, v in EFF.items():
                    tr[f"전기 {e} 증감_kg"] = km * CB.GRID / v + taxi * mult * tef - car
                tr["경유 증감_kg"] = km * CB.ef(DIESEL, "A") + taxi * mult * tef - car
                trows.append(tr)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(f"{OUT}/탄소검토_단계표기준.csv", index=False, encoding="utf-8-sig")
    first = []
    for name in ["경유 승합"] + [f"전기 {e}" for e in EFF]:
        neg = T[T[f"{name} 증감_kg"] < 0]["하루 이용자_명"].tolist()
        first.append({"승합": name, "처음 감축되는 시나리오_명": neg[0] if neg else "없음",
                      "그 이후 모든 점 감축": bool(neg) and all(T.loc[T["하루 이용자_명"] >= neg[0], f"{name} 증감_kg"] < 0),
                      "감축되는 점": ", ".join(map(str, neg)) if neg else "없음"})
    F = pd.DataFrame(first)
    F.to_csv(f"{OUT}/탄소검토_처음감축.csv", index=False, encoding="utf-8-sig")
    TX = pd.DataFrame(trows)
    TX.round(3).to_csv(f"{OUT}/탄소검토_택시민감도.csv", index=False, encoding="utf-8-sig")
    # 택시 민감도에서 감축/증가 판정이 바뀌는 칸
    base = TX[(TX["택시 계수"] == list(TAXI)[0]) & (TX["택시 공차"] == "공차 없음(기본)")].set_index("하루 이용자_명")
    flips = []
    for (tn, dd), g in TX.groupby(["택시 계수", "택시 공차"]):
        g = g.set_index("하루 이용자_명")
        for c in [c for c in g.columns if c.endswith("증감_kg")]:
            for n in g.index:
                if np.sign(g.loc[n, c]) != np.sign(base.loc[n, c]):
                    flips.append(f"{tn} / {dd} / {c} / {n}명")
    cols = ["하루 이용자_명", "DRT 대수(단계표)", "자가용 kgCO2/일"] + [f"{a} {b}" for a in ["경유 승합", "전기 4.1", "전기 3.9", "전기 3.3"] for b in ["kgCO2/일", "증감_kg", "증감_%"]]
    print(T[cols].round(1).to_string(index=False))
    print(F.to_string(index=False))
    print("택시 LPG kgCO2/kg(공식 계수):", round(LPG_KG, 4), "| 방식 A 택시 kg/km(밀도 가정):", round(TAXI[list(TAXI)[1]], 4))
    print("거절 택시 km/일:", T[["하루 이용자_명", "거절 택시 km/일"]].round(2).values.tolist())
    print("택시 민감도로 감축/증가 판정이 바뀌는 칸:", flips if flips else "없음")
    print(TX.round(2).to_string(index=False))
