"""탄소 계산: '면허 반납 후에도 자가용을 계속 이용하는 경우' vs '반납 후 DRT를 이용하는 경우' (같은 이동, 수단만 다름).

실행: 프로젝트 루트에서 `python scripts/금천구_탄소계산.py`  (선행: 금천구_DRT_반납자시뮬.py. 체크포인트를 그대로 읽고, 없는 단계안 시나리오만 계산)
비교 틀(교수님 피드백): 반납률과 탄소를 직접 연결하지 않고, 같은 승객 편도를 어떤 수단으로 하느냐의 배출량 차이
  - 자가용(기준선): 같은 승객 편도 전부를 각자 승용차로 직행(출발 정류소 -> 목적지 차량 최단 도로거리). 주차 탐색 거리 없음(보수적)
  - DRT: 빈 차 포함 총 운행거리 x 승합 배출계수 + 거절 편도의 택시 직행 거리 x 택시 배출계수
  - 고정노선(정원 12 반영, 참고): 총 운행거리 x 승합 배출계수
배출계수(운행 단계, kgCO2/km)
  - 방식 A(기본): 표시연비(km/L, 실주행 보정) x 국가 연료 CO2 계수 / 전기: 전력 배출계수 ÷ 전비(km/kWh)
  - 방식 B(민감도): 차량 표시 CO2(g/km, 인증 시험값). 전기는 A와 같음
  - 연료 CO2 계수 = 순발열량(에너지법 시행규칙 열량환산기준, 2022.11) x 탄소배출계수(IPCC 2006, 2022.1 공표) x 44/12
  - 전력 = 2023년 국가 전력배출계수(소비단) 0.4173 tCO2eq/MWh (기후에너지환경부, 2025.12 공표)
  - 차량 연비·표시 CO2는 자동차 정보 사이트(카눈) 제원표 값이며, 한국에너지공단 표시연비 DB로 최종 확인이 필요하다.
연간 = 하루 x 260일(평일, 52주 x 5일, 공휴일 미차감) 또는 x 365일(주 7일)
"""
import importlib
import multiprocessing as mp
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
B = importlib.import_module("금천구_DRT_반납자시뮬")
V = B.V
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, INTERIM = "outputs", "data/interim"
TJ = 1e-6  # MJ -> TJ
FUEL = {  # kgCO2/L = 순발열량(MJ/L) x 1e-6 TJ/MJ x 탄소계수(tC/TJ) x 44/12 x 1000 kg/t
    "휘발유": 30.1 * TJ * 19.731 * 44 / 12 * 1000,
    "경유": 35.3 * TJ * 20.090 * 44 / 12 * 1000,
}
GRID = 0.4173  # kgCO2eq/kWh
# 차량: (구분, 연료, 표시연비 km/L 또는 전비 km/kWh, 표시 CO2 g/km, 출처)
VEH = {
    "자가용 준중형(아반떼 1.6 가솔린, 기본)": ("휘발유", 15.3, 107, "카눈 제원표: 더 뉴 아반떼 2025 가솔린 1.6 15인치 복합 15.3km/L, CO2 107g/km"),
    "자가용 중형(쏘나타 2.0 가솔린, 참고)": ("휘발유", 12.6, 132, "카눈 제원표: 쏘나타 디 엣지 가솔린 2.0 17인치 복합 12.6km/L, CO2 132g/km"),
    "DRT 경유 승합(스타리아 2.2 디젤 11인승)": ("경유", 10.8, 179, "카눈 제원표: 스타리아 투어러 2.2 디젤 11인승 자동 2WD 복합 10.8km/L, CO2 179g/km"),
    "DRT 전기 승합(4.1km/kWh, 기준)": ("전기", 4.1, None, "더 뉴 스타리아 일렉트릭 인증 전비 4.1km/kWh(카고 3·5인승 기준, 언론 보도)"),
    "DRT 전기 승합(3.3km/kWh, 낮은 값)": ("전기", 3.3, None, "기준 전비의 80%(11인승 만석·냉난방 감소 가정, 사용자 지정)"),
    "택시 LPG(쏘나타 LPG 2.0 택시)": ("LPG", 9.5, 137, "카눈 제원표: 쏘나타 디 엣지 LPG 2.0 택시 17인치 복합 9.5km/L, CO2 137g/km — 방식 A·B 모두 표시 CO2 사용(LPG 밀도 공식값 없음)"),
}


def ef(name, method):
    fuel, eco, label, _ = VEH[name]
    if fuel == "전기":
        return GRID / eco
    if fuel == "LPG" or method == "B":
        return label / 1000
    return FUEL[fuel] / eco


CARS = ["자가용 준중형(아반떼 1.6 가솔린, 기본)", "자가용 중형(쏘나타 2.0 가솔린, 참고)"]
VANS = ["DRT 경유 승합(스타리아 2.2 디젤 11인승)", "DRT 전기 승합(4.1km/kWh, 기준)", "DRT 전기 승합(3.3km/kWh, 낮은 값)"]
TAXI = "택시 LPG(쏘나타 LPG 2.0 택시)"

if __name__ == "__main__":
    pd.set_option("display.width", 260)
    B.POOL = mp.Pool(11, initializer=V.init)
    V.init()
    R = pickle.load(open(f"{INTERIM}/DRT_반납자_결과.pkl", "rb"))
    need = R["need"]

    # ---------------- 배출계수 표
    rows = [{"항목": "휘발유 CO2", "값": FUEL["휘발유"], "단위": "kgCO2/L", "근거": "순발열량 30.1MJ/L(에너지법 시행규칙 열량환산기준 2022.11) x 탄소계수 19.731tC/TJ(IPCC 2006, 2022.1 공표) x 44/12"},
            {"항목": "경유 CO2", "값": FUEL["경유"], "단위": "kgCO2/L", "근거": "순발열량 35.3MJ/L x 탄소계수 20.090tC/TJ x 44/12 (같은 출처)"},
            {"항목": "전력", "값": GRID, "단위": "kgCO2eq/kWh", "근거": "2023년 국가 전력배출계수(소비단) 0.4173tCO2eq/MWh, 기후에너지환경부 2025.12 공표(온실가스종합정보센터 검증)"}]
    for k, (fuel, eco, label, src) in VEH.items():
        rows.append({"항목": k, "값": eco, "단위": "km/kWh" if fuel == "전기" else "km/L", "근거": src,
                     "방식A_kgCO2/km": ef(k, "A"), "방식B_kgCO2/km": ef(k, "B")})
    EF = pd.DataFrame(rows)
    EF.round(4).to_csv(f"{OUT}/탄소_배출계수_출처.csv", index=False, encoding="utf-8-sig")
    print(EF.round(4).to_string(index=False))

    # ---------------- 시나리오
    scen = []  # (구분, 이름, n, DRT 대수)
    for n in B.REP:
        scen.append(("메인(반납자 대표)", f"하루 {n}명 · DRT {need[n]}대", n, need[n]))
    for n in (26, 44, 62, 79):
        scen.append(("단계안", f"1단계 2대 공동 운영 · 하루 {n}명", n, 2))
    for n in (88, 98, 110):
        scen.append(("단계안", f"2단계 3대 · 하루 {n}명", n, 3))
    for n in B.SENS:
        k = need[n] or B.MAXF
        scen.append(("민감도(이용률)", f"하루 {n}명 · DRT {k}대" + ("(기준 미달)" if not need[n] else ""), n, k))
    curve_ns = sorted(n for n, k in need.items() if k)  # 손익분기 곡선용: 모든 계산 수요(필요 대수)

    def metrics(n, k):
        d = B.summ(B.get(n, "DRT", k))
        f = B.summ(B.get(n, "고정", 2))
        direct_all = d["직행_평균km"] * d["탑승편도_건"] + d["택시대체_km"]  # 모든 편도의 직행 거리(자가용 기준선)
        return d, f, direct_all

    out = []
    for grp, name, n, k in scen:
        d, f, direct_all = metrics(n, k)
        legs = d["편도_건"]
        for m in ("A", "B"):
            for car in CARS:
                car_kg = direct_all * ef(car, m)
                for van in VANS:
                    drt_kg = d["운행거리_km"] * ef(van, m) + d["택시대체_km"] * ef(TAXI, m)
                    fx_kg = f["운행거리_km"] * ef(van, m)
                    out.append({"구분": grp, "시나리오": name, "하루 이용자_명": n, "DRT 대수": k, "방식": m, "자가용": car, "승합 차종": van,
                                "편도_건/일": legs, "자가용 직행 km/일": direct_all, "DRT 운행 km/일": d["운행거리_km"], "택시 km/일": d["택시대체_km"],
                                "고정노선 2대 km/일": f["운행거리_km"], "DRT 편도당 차량km / 직행km": d["탑승편도당_차량km"] / d["직행_평균km"],
                                "평균탑승(빈차포함)": d["평균탑승인원_빈차포함"],
                                "자가용 kgCO2/일": car_kg, "DRT kgCO2/일": drt_kg, "고정노선 kgCO2/일(참고)": fx_kg,
                                "DRT-자가용 kgCO2/일": drt_kg - car_kg, "감축률_%": (car_kg - drt_kg) / car_kg * 100,
                                "연간 차이 tCO2(평일 260일)": (drt_kg - car_kg) * 260 / 1000, "연간 차이 tCO2(365일)": (drt_kg - car_kg) * 365 / 1000,
                                "자가용 g/편도": car_kg / legs * 1000, "DRT g/편도": drt_kg / legs * 1000, "고정노선 g/편도(참고)": fx_kg / legs * 1000,
                                "판정": "감축" if drt_kg < car_kg else "증가",
                                # 손익분기 평균 탑승 인원(빈 차 포함, 거리 가중): 택시 몫 제외 근사 = 인km / (자가용 직행 km x EF자가용/EF승합)
                                "손익분기 평균탑승_명": d["인km"] / (direct_all * ef(car, m) / ef(van, m))})
    O = pd.DataFrame(out)
    O.round(3).to_csv(f"{OUT}/탄소_시나리오별_배출량.csv", index=False, encoding="utf-8-sig")

    base = O[(O.방식 == "A") & (O.자가용 == CARS[0])]
    view = base.pivot_table(index=["구분", "시나리오"], columns="승합 차종", values="DRT-자가용 kgCO2/일", sort=False).round(1)
    print("\n[방식 A, 자가용 준중형] DRT - 자가용 (kgCO2/일, 음수 = 감축)")
    print(view.to_string())

    # ---------------- 손익분기: 하루 이용자 곡선(필요 대수 기준)
    cv = []
    for n in curve_ns:
        d, f, direct_all = metrics(n, need[n])
        row = {"하루 이용자_명": n, "DRT 대수": need[n], "DRT km/일": d["운행거리_km"], "직행 km/일": direct_all, "편도당 차량km/직행km": d["탑승편도당_차량km"] / d["직행_평균km"],
               "평균탑승(빈차포함)": d["평균탑승인원_빈차포함"]}
        for m in ("A", "B"):
            for van in VANS:
                for car in CARS:
                    diff = d["운행거리_km"] * ef(van, m) + d["택시대체_km"] * ef(TAXI, m) - direct_all * ef(car, m)
                    row[f"{m}|{van}|{car}"] = diff
        cv.append(row)
    CV = pd.DataFrame(cv)
    be = []
    for m in ("A", "B"):
        for van in VANS:
            for car in CARS:
                r = ef(car, m) / ef(van, m)  # 손익분기 '차량km/직행km' 비율
                col = f"{m}|{van}|{car}"
                neg = CV[CV[col] < 0]["하루 이용자_명"].tolist()
                allneg_from = next((n for n in CV["하루 이용자_명"] if (CV.loc[CV["하루 이용자_명"] >= n, col] < 0).all()), None)
                be.append({"방식": m, "승합 차종": van, "자가용": car, "승합 kgCO2/km": ef(van, m), "자가용 kgCO2/km": ef(car, m),
                           "손익분기 비율(DRT 차량km ÷ 승객 직행km 이하면 감축)": r,
                           "손익분기 평균탑승(빈차포함, 우회 1.0 가정)": 1 / r,
                           "감축되는 하루 이용자(계산 지점)": ", ".join(map(str, neg)) if neg else "없음(24~441명 전 구간 증가)",
                           "이 이상 모든 지점 감축_명": allneg_from})
    BE = pd.DataFrame(be)
    BE.round(3).to_csv(f"{OUT}/탄소_손익분기.csv", index=False, encoding="utf-8-sig")
    CV.round(3).to_csv(f"{OUT}/탄소_하루이용자별_차이.csv", index=False, encoding="utf-8-sig")
    print("\n[손익분기]")
    print(BE.round(2).to_string(index=False))

    # ---------------- 그림
    SURF, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
    COL = {VANS[0]: "#eb6834", VANS[1]: "#1baf7a", VANS[2]: "#2a78d6"}
    LAB = {VANS[0]: "경유 승합(10.8km/L)", VANS[1]: "전기 승합 4.1km/kWh", VANS[2]: "전기 승합 3.3km/kWh"}
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), dpi=150, facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    mb = base[base.구분 != "민감도(이용률)"]
    names = list(dict.fromkeys(mb.시나리오))
    ngrp = (mb.drop_duplicates("시나리오").구분 == "메인(반납자 대표)").sum()
    x = np.arange(len(names)) + np.where(np.arange(len(names)) >= ngrp, 0.8, 0)  # 메인과 단계안 사이 간격
    short = lambda t: (t.replace("하루 ", "").replace("1단계 2대 공동 운영 · ", "1단계 2대\n").replace("2단계 3대 · ", "2단계 3대\n")
                       .replace(" · DRT ", "\nDRT "))
    w = 0.27
    for i, van in enumerate(VANS):
        v = mb[mb["승합 차종"] == van].set_index("시나리오").loc[names, "DRT-자가용 kgCO2/일"]
        ax.bar(x + (i - 1) * w, v, width=w * 0.92, color=COL[van], label=LAB[van], edgecolor=SURF, lw=1.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [short(t) for t in names], fontsize=7.5)
    ax.text(np.mean(x[:ngrp]), 0.02, "메인(반납자 대표)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
    ax.text(np.mean(x[ngrp:]), 0.02, "단계안", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
    ax.set(title="DRT - 자가용(준중형 휘발유) 하루 CO2 차이 (방식 A, 음수 = 감축)", ylabel="kgCO2/일")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    for van in VANS:
        ax.plot(CV["하루 이용자_명"], CV[f"A|{van}|{CARS[0]}"], color=COL[van], marker="o", ms=4, lw=2, label=LAB[van])
    ax.axhline(0, color=INK2, lw=1)
    ax.set(title="하루 이용자별 DRT - 자가용 (필요 대수 운행, 방식 A)", xlabel="하루 이용자 (명)", ylabel="kgCO2/일")
    ax.legend(frameon=False, fontsize=8.5)
    for a in axes:
        a.set_facecolor(SURF)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=GRIDC, lw=0.8)
        a.set_axisbelow(True)
    fig.text(0.01, 0.01, "운행 단계만. 자가용 = 같은 편도를 각자 직행(주차 탐색 제외). DRT = 빈 차 포함 운행거리 + 거절 택시. 연료계수: 에너지법 열량환산기준(2022)·IPCC 2006, "
             "전력 0.4173kgCO2eq/kWh(2023, 기후에너지환경부). 차량 연비는 제원표 값(공식 DB 확인 필요).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/탄소_DRT_vs_자가용.png", facecolor=SURF)
    B.POOL.close()
    pickle.dump({"EF": EF, "O": O, "BE": BE, "CV": CV}, open(f"{INTERIM}/탄소_결과.pkl", "wb"))
    print("저장: 탄소_배출계수_출처.csv, 탄소_시나리오별_배출량.csv, 탄소_손익분기.csv, 탄소_하루이용자별_차이.csv, 탄소_DRT_vs_자가용.png")
