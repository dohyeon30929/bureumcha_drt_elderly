"""DRT v4 그림: (2) 새 권역 노선 지도, (3) 시간대별 호출·거절(1%, 이전 기준 참고). 최대 대기 민감도는 새 기준으로 금천구_DRT_반납자_그림.py
반납자 시나리오·고정노선 비교·점심 증차·탄소 거리·검증 그림은 금천구_DRT_반납자_그림.py

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_v4_그림.py`  (선행: 금천구_DRT_v4_시뮬.py, 금천구_DRT_v4_민감도.py)
"""
import pickle
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT, INTERIM = "outputs", "data/interim"
Din = pickle.load(open(f"{INTERIM}/DRT_v4_입력.pkl", "rb"))
R = pickle.load(open(f"{INTERIM}/DRT_v4_결과.pkl", "rb"))
S = pickle.load(open(f"{INTERIM}/DRT_v4_민감도_결과.pkl", "rb"))
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C1, C2, CF1, CF2, BAND = "#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#f0efec"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
XT = [9, 18, 26, 44, 62, 88, 132, 176, 265, 441]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def logx(ax):
    ax.set_xscale("log")
    ax.set_xticks(XT, [str(x) for x in XT])
    ax.minorticks_off()



# ---------------------------------------------------------------- 2. 노선 지도 (새 권역)
roads = Roads(path="data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json")
st, snaps, I = Din["stops"], Din["snaps"], Din["I"]
sx = np.array([s["sx"] for s in snaps])
sy = np.array([s["sy"] for s in snaps])
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
loop = Din["loop0"]
fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), dpi=150, facecolor=SURF)
pts = []
for a, c in zip(loop, loop[1:] + loop[:1]):
    p = np.array(roads.drive_path(snaps[a], snaps[c]))
    pts.append(p)
    axes[0].plot(p[:, 0], p[:, 1], color=CF1, lw=2, alpha=0.85, zorder=3)
    m = len(p) // 2
    if len(p) > 2:
        axes[0].annotate("", xy=p[min(m + 1, len(p) - 1)], xytext=p[m], arrowprops=dict(arrowstyle="-|>", color=CF1, lw=0, mutation_scale=10), zorder=4)
allp = np.vstack(pts + [np.c_[sx, sy]])
ext = np.r_[np.minimum(cl.total_bounds[:2], allp.min(0)) - 200, np.maximum(cl.total_bounds[2:], allp.max(0)) + 200]
ZONE = Din["ZONE4"]
popA = st.loc[[i for i in Din["BI"] if ZONE[i] == "A"], "배정_70세이상_명"].sum()
popB = st.loc[[i for i in Din["BI"] if ZONE[i] == "B"], "배정_70세이상_명"].sum()
for ax, title in [(axes[0], f"고정노선 순환 (기본안, B01 제외): {Din['loop0_m'] / 1000:.1f}km, {Din['cycle0']:.0f}분 (승차 12 + 목적지 5)"),
                  (axes[1], "DRT 새 권역·대기 장소 (2대 권역 고정, 목적지 5곳 공통)")]:
    ax.set_facecolor(SURF)
    cl.boundary.plot(ax=ax, color="#c3c2b7", lw=1, zorder=1)
    for r in cl.itertuples():
        pt = r.geometry.representative_point()
        ax.text(pt.x - 250, pt.y + 150, r.ADM_NM, fontsize=10, color=MUTED, ha="center", zorder=2,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
for ax, zc in [(axes[0], {"A": INK2, "B": INK2}), (axes[1], {"A": C1, "B": C2})]:
    for i in Din["BI"]:
        on_loop = i in loop
        ax.scatter(sx[i], sy[i], s=60, color=zc[ZONE[i]] if (on_loop or ax is axes[1]) else "#c3c2b7", edgecolor=SURF, lw=1.5, zorder=6)
        ax.annotate(st.ID[i] + ("" if on_loop or ax is axes[1] else " (순환 제외)"), (sx[i], sy[i]), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
    for i in Din["DI"]:
        ax.scatter(sx[i], sy[i], s=170, marker="*", color="#e34948", edgecolor=SURF, lw=1, zorder=6)
        ax.annotate(st.이름[i].replace(".시흥4동주민센터", ""), (sx[i], sy[i]), xytext=(7, -12), textcoords="offset points", fontsize=8,
                    color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
for z, col in [("A", C1), ("B", C2)]:
    d = I[f"DEP4_{z}"]
    axes[1].scatter([sx[d]], [sy[d]], s=170, marker="s", color=col, edgecolor=INK, lw=1, zorder=8)
    axes[1].annotate(f"{z} 대기", (sx[d], sy[d]), xytext=(-9, -4), textcoords="offset points", ha="right", fontsize=9, color=INK, zorder=9,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=C1, ms=9, label=f"권역 A 승차 (시흥3·5동 8곳, 배정 {popA:,.0f}명)"),
                        Line2D([], [], marker="o", ls="", color=C2, ms=9, label=f"권역 B 승차 (시흥4동 4곳 + B01, 배정 {popB:,.0f}명)"),
                        Line2D([], [], marker="*", ls="", color="#e34948", ms=13, label="목적지 정류소 5곳"),
                        Line2D([], [], marker="s", ls="", color=MUTED, markeredgecolor=INK, ms=9, label="권역별 대기 장소")],
               frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "도로: OSM 주행망(관악구 난향동까지 확장, 일방통행 반영). B01은 차량으로 약 4km 우회 필요(관악구 경유)라 고정노선 기본안에서 제외. © OpenStreetMap contributors",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/DRT_v4_노선지도.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 3. 시간대별 호출·거절 (1%)
hr = S["hr"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), dpi=150, facecolor=SURF, sharey=True)
for ax, an in zip(axes, ["DRT 1대", "DRT 2대"]):
    g = hr[hr.안 == an]
    ax.bar(g.시, g.호출_계, color="#c3c2b7", width=0.75, label="호출")
    ax.bar(g.시, g.거절_가는, color=CF1, width=0.75, label="거절: 가는 호출")
    ax.bar(g.시, g.거절_귀가, bottom=g.거절_가는, color=CF2, width=0.75, label="거절: 귀가 호출")
    for h, q, j in zip(g.시, g.호출_계, g.거절_계):
        if q >= 1:
            ax.text(h, q + 0.3, f"{j / q * 100:.0f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set(title=f"{an}{' (새 권역)' if '2' in an else ''} — 이용률 1% (하루 88명), 숫자 = 시간대 거절률", xlabel="호출 시각(시)", ylabel="하루 평균 건수")
    ax.set_xticks(range(8, 23))
    style(ax)
axes[0].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/DRT_v4_시간대별_호출거절.png", facecolor=SURF)
plt.close(fig)

print("저장: DRT_v4_노선지도.png, DRT_v4_시간대별_호출거절.png")
