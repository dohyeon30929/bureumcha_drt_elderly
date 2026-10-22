"""반납자 시나리오 그림: (1) 대표 시나리오별 필요 DRT vs 고정노선 2대(정원 반영) 총시간·운행거리, (2) 하루 이용자별 필요 차량,
(3) 한계 이용자 스캔(B01 제외), (4) 점심 증차 실험안 시간대별 거절, (5) 탄소용 편도 1건당 차량 km vs 직행 거리, (6) 시흥1동 도착 분포 검증.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_반납자_그림.py`  (선행: 금천구_DRT_반납자시뮬.py, 금천구_시흥1동도착_검증.py)
"""
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, INTERIM = "outputs", "data/interim"
R = pickle.load(open(f"{INTERIM}/DRT_반납자_결과.pkl", "rb"))
M, PK, HR, dem, LS, LIM, need = R["M"], R["PK"], R["HR"], R["dem"], R["LS"], R["LIM"], R["need"]
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C1, C2, CF1, CF2, BAND = "#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#f0efec"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def style(ax, grid="y"):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def note(fig, txt, y=0.01):
    fig.text(0.01, y, txt, fontsize=7.5, color=MUTED)


# ---------------------------------------------------------------- 1. 대표 시나리오 비교
rep = M[M.구분 == "대표(반납자)"]
ns = sorted(rep["하루 이용자_명"].unique())
drt = rep[rep.종류 == "DRT 필요"].set_index("하루 이용자_명").loc[ns]
fx = rep[rep.안 == "고정노선 2대(B01 제외, 정원 12)"].set_index("하루 이용자_명").loc[ns]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=150, facecolor=SURF)
x = np.arange(len(ns))
xw = 0.36
for ax, col_m, title, unit in [(axes[0], "총시간_전체승객", "승객 1인당 총시간(대기+승차) — 전체 승객 기준", "분"), (axes[1], "운행거리", "하루 차량 운행거리", "km")]:
    for k, (g, col, lab) in enumerate([(drt, INK, "DRT 필요 대수(거절 5%·대기 15분, 공동 운영)"), (fx, CF2, "고정노선 2대(B01 제외, 정원 12)")]):
        xs = x + (k - 0.5) * xw
        if unit == "분":
            v, sd, p95 = g[f"{col_m}_평균"], g[f"{col_m}_반복SD"], g[f"{col_m}_p95"]
            ax.bar(xs, v, width=xw * 0.92, color=col, label=lab, edgecolor=SURF, lw=2)
            ax.errorbar(xs, v, yerr=sd, fmt="none", ecolor=INK2 if col != INK else MUTED, capsize=3, lw=1)
            ax.scatter(xs, np.minimum(p95, 124), marker="_", s=180, color=INK2, zorder=5, label="95퍼센타일" if k == 0 else None)
            for xx, vv, pp in zip(xs, v, p95):
                ax.text(xx, min(pp, 124) + 2, f"{vv:.0f}" + (f"\n(p95 {pp:.0f}↑)" if pp > 124 else ""), ha="center", fontsize=8, color=INK)
        else:
            v = g["운행거리_km"]
            ax.bar(xs, v, width=xw * 0.92, color=col, label=lab, edgecolor=SURF, lw=2)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 8, f"{vv:.0f}", ha="center", fontsize=8, color=INK)
    lbl = [f"{n}명\nDRT {int(drt.loc[n, '차량_대'])}대" for n in ns]
    ax.set_xticks(x, lbl)
    ax.set(title=title, ylabel=unit, xlabel="하루 이용자 (반납자 수요 대표 시나리오)")
    style(ax)
axes[0].set_ylim(0, 138)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for k, n in enumerate(ns):
    f1 = fx.loc[n, "첫차만차_못탐_%"]
    if f1 >= 1:
        axes[0].text(k + 0.5 * xw, 4, f"만차\n{f1:.0f}%", ha="center", fontsize=7, color=INK)
note(fig, "막대 = 평균, 오차막대 = 300회 반복 간 표준편차, 가로선 = 95퍼센타일. 같은 승객·같은 목적지(B01 제외 승차 12곳). DRT 거절 편도는 30분 대기 + 택시 직행. "
          "고정노선 '만차' = 첫 차량이 만차라 못 탄 편도 비율(다음 차 대기).")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/DRT_반납자_대표시나리오_비교.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 2. 하루 이용자별 필요 차량
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150, facecolor=SURF)
pts = sorted((n, k) for n, k in need.items() if k)
ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=MUTED, lw=1)
for n, k in pts:
    isrep = n in ns
    issens = n in (88, 265, 441)
    ax.scatter(n, k, s=60 if isrep else 30, color=INK if isrep else (CF1 if issens else C1), zorder=4, edgecolor=SURF, lw=1.5)
    if isrep or issens:
        ax.annotate(f"{n}명 → {k}대", (n, k), xytext=(4, -14) if issens else (4, 6), textcoords="offset points", fontsize=8, color=INK)
lo, hi = dem["하루 이용자_명"].min(), dem["하루 이용자_명"].max()
ax.axvspan(lo, hi, color=BAND, zorder=0)
ax.text((lo + hi) / 2, 0.97, f"반납자 수요 범위 {lo}~{hi}명/일", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=INK2)
ax.scatter([], [], color=INK, s=60, label="대표 시나리오")
ax.scatter([], [], color=C1, s=30, label="기타 반납자 조합")
ax.scatter([], [], color=CF1, s=30, label="민감도(이용률 1·3·5%)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set(title="하루 이용자별 필요 DRT 차량 (거절률 5%·평균 대기 15분 이하, 공동 운영, B01 제외)", xlabel="하루 이용자 (명, 왕복)", ylabel="차량 (대)")
style(ax, "both")
fig.tight_layout()
fig.savefig(f"{OUT}/DRT_반납자기준_필요차량.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 3. 한계 이용자 스캔
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=150, facecolor=SURF)
SC = {"DRT 1대": (C1, "-", "o"), "DRT 2대(공동 운영)": (C2, "-", "s"), "참고: DRT 2대 권역 고정": (MUTED, "--", "^")}
for ax, y, thr, yl in [(axes[0], "거절률_%", 5, "거절률 (%)"), (axes[1], "대기_평균", 15, "평균 대기 (분)")]:
    for name, (col, ls, mk) in SC.items():
        g = LS[LS.안 == name]
        ax.plot(g["하루 이용자_명"], g[y], color=col, ls=ls, marker=mk, ms=4, lw=2, label=f"{name} (한계 {LIM[name]}명)")
    ax.axhline(thr, color=INK2, lw=1, ls=":")
    ax.set(xlabel="하루 이용자 (명)", ylabel=yl, title=f"{yl.split(' (')[0]} — 한계 이용자 = 평균 대기 15분·거절률 5% 이하를 지키는 마지막 지점")
    style(ax)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
note(fig, "B01 제외 승차 12곳, 최대 대기 30분, 300회. 계산 격자 하루 약 9명 간격.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/DRT_한계이용자_스캔_B01제외.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 4. 점심 증차 실험안
if len(PK):
    # 10~14시안이 시간대 기준을 못 맞추는 대표 시나리오 중 가장 큰 수요(없으면 첫 시나리오)
    cand = PK[PK["하루 이용자_명"].isin(ns) & PK.안.str.contains("10~14") & ~PK["시간대 기준 충족(모든 시간 거절 5% 이하)"]]
    n4 = int(cand["하루 이용자_명"].max()) if len(cand) else int(PK["하루 이용자_명"].iloc[0])
    g4 = HR[HR["하루 이용자_명"] == n4]
    order = list(dict.fromkeys(g4.안))
    sty = [(CF1, "-", "o"), (INK2, "--", "s"), (C1, "-", "^"), (C2, "-", "D"), ("#7c4dc4", "-", "v")]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150, facecolor=SURF)
    for name, (col, ls, mk) in zip(order, sty):
        g = g4[g4.안 == name]
        ax.plot(g.시, g["거절률_%"], color=col, ls=ls, marker=mk, ms=5, lw=2, label=name)
    ax.axhline(5, color=INK2, lw=1, ls=":")
    ax.axvspan(9.5, 13.5, color=BAND, zorder=0)
    ax.set(title=f"하루 {n4}명: 점심 증차 실험안 — 시간대별 거절률 (회색 = 10~13시)", xlabel="호출 시각(시)", ylabel="거절률 (%)")
    ax.set_xticks(range(8, 20))
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/DRT_점심증차_실험안_시간대별거절.png", facecolor=SURF)
    plt.close(fig)

# ---------------------------------------------------------------- 5. 탄소용: 편도 1건당 차량 km vs 직행 거리
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, facecolor=SURF)
for k, (g, col, lab) in enumerate([(drt, INK, "DRT 필요 대수"), (fx, CF2, "고정노선 2대(정원 12)")]):
    xs = x + (k - 0.5) * xw
    ax.bar(xs, g.탑승편도당_차량km, width=xw * 0.92, color=col, label=f"{lab}: 승객 편도 1건당 차량 km", edgecolor=SURF, lw=2)
    for xx, vv, oc in zip(xs, g.탑승편도당_차량km, g.평균탑승인원_빈차포함):
        ax.text(xx, vv + 0.05, f"{vv:.2f}\n(탑승 {oc:.1f}명)", ha="center", fontsize=7.5, color=INK)
ax.plot(x, drt.직행_평균km, color=C1, marker="o", lw=2, label=f"승객 편도 직행 평균 거리 {drt.직행_평균km.mean():.2f}km(자가용·택시 대체 기준)")
ax.set_xticks(x, [f"{n}명" for n in ns])
ax.set(title="승객 편도 1건당 차량 운행거리 vs 직행 거리 (탄소 계산 준비, 배출계수는 다음 단계)", ylabel="km", xlabel="하루 이용자 (대표 시나리오)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/DRT_탄소용_편도당거리.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 6. 검증: 시흥1동 도착 분포
sh = pd.read_csv(f"{OUT}/검증_시흥1동도착_vs_OD_8_16시비중.csv", encoding="utf-8-sig", index_col=0)
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150, facecolor=SURF)
for col, (c, ls, mk) in zip(sh.columns, [(INK, "-", "o"), (MUTED, "--", "s"), (CF1, "-", "^")]):
    ax.plot(sh.index, sh[col], color=c, ls=ls, marker=mk, ms=5, lw=2, label=col)
ax.axvspan(9.5, 13.5, color=BAND, zorder=0)
ax.set(title="시흥1동 도착 70세 이상(2026년 8월, 귀가 제외) vs 시뮬레이션 가는 호출 분포(O-D 8/31, 60세 이상)", xlabel="시", ylabel="8~16시 안 비중 (%)")
ax.set_xticks(range(8, 17))
ax.legend(frameon=False, fontsize=8.5)
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/검증_시흥1동도착_분포비교.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 7. 최대 대기 30분(기본) vs 45분(민감도), 통일 기준
MW, MWL = R["MW"], R["MWL"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=150, facecolor=SURF)
ST7 = {(30, "DRT 1대"): (C1, "-", "o"), (45, "DRT 1대"): (C1, "--", "s"), (30, "DRT 2대(공동 운영)"): (C2, "-", "o"), (45, "DRT 2대(공동 운영)"): (C2, "--", "s")}
for ax, y, thr, yl in [(axes[0], "거절률_%", 5, "거절률 (%)"), (axes[1], "대기_평균", 15, "평균 대기 (분)")]:
    for (mw, name), (col, ls, mk) in ST7.items():
        g = MW[(MW["최대 대기_분"] == mw) & (MW.안 == name)]
        lim = int(MWL[(MWL["최대 대기_분"] == mw) & (MWL.안 == name)]["한계 이용자_명"].iloc[0])
        ax.plot(g["하루 이용자_명"], g[y], color=col, ls=ls, marker=mk, ms=4, lw=2,
                label=f"{name} · 최대 대기 {mw}분{' (기본)' if mw == 30 else ''} · 한계 {lim}명")
    ax.axhline(thr, color=INK2, lw=1, ls=":")
    ax.set(xlabel="하루 이용자 (명)", ylabel=yl, title=yl.split(" (")[0])
    style(ax)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
note(fig, "통일 기준(B01 제외, 공동 운영, 300회). 45분은 거절을 줄이지만 더 많이 태우면서 평균 대기가 15분을 먼저 넘어 2대 한계가 79 -> 62명으로 낮아짐.")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/DRT_최대대기별_거절대기_B01제외.png", facecolor=SURF)
plt.close(fig)
print("저장: DRT_반납자_대표시나리오_비교.png, DRT_반납자기준_필요차량.png, DRT_한계이용자_스캔_B01제외.png, DRT_점심증차_실험안_시간대별거절.png, "
      "DRT_탄소용_편도당거리.png, 검증_시흥1동도착_분포비교.png, DRT_최대대기별_거절대기_B01제외.png")
