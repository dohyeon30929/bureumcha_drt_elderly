"""DRT v3 그림: 이용률별 대기·거절률 곡선 + 총시간 비교, 노선 지도(고정노선 순환 / DRT 권역·대기장소), 귀가 분포 비교.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_v3_그림.py`  (선행: 금천구_DRT_v3_시뮬.py)
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

OUT, INTERIM = "outputs", "data/interim"
Din = pickle.load(open(f"{INTERIM}/DRT_v3_입력.pkl", "rb"))
R = pickle.load(open(f"{INTERIM}/DRT_v3_결과.pkl", "rb"))
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C1, C2, CF1, CF2 = "#2a78d6", "#1baf7a", "#eb6834", "#eda100"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. 곡선
sc, main = R["sc"], R["main"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150, facecolor=SURF)
for an, col in [("DRT 1대", C1), ("DRT 2대", C2)]:
    g = sc[sc.안 == an]
    axes[0].plot(g.이용자_명, g.대기_평균, color=col, lw=2, marker="o", ms=4, label=an + (" (권역 고정)" if "2" in an else ""))
    axes[1].plot(g.이용자_명, g["거절률_%"], color=col, lw=2, marker="o", ms=4, label=an)
fx = main[main.안.str.startswith("고정노선")].drop_duplicates("안")
for an, col in [("고정노선 1대", CF1), ("고정노선 2대", CF2)]:
    w = fx.loc[fx.안 == an, "대기_평균"].iloc[0]
    axes[0].axhline(w, color=col, lw=1.6, ls="--", label=f"{an} (배차간격/2 = {w:.0f}분)")
axes[0].axhline(15, color=INK2, lw=1, ls=":")
axes[0].text(sc.이용자_명.max(), 15.4, "전환 기준 15분", ha="right", fontsize=8.5, color=INK2)
axes[1].axhline(5, color=INK2, lw=1, ls=":")
axes[1].text(sc.이용자_명.max(), 5.8, "전환 기준 5%", ha="right", fontsize=8.5, color=INK2)
axes[0].set(title="평균 대기시간", xlabel="하루 이용자 수 (명, 왕복)", ylabel="분")
axes[1].set(title="거절률", xlabel="하루 이용자 수 (명, 왕복)", ylabel="%")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
# 총시간 (헤드라인)
ans = ["DRT 1대", "DRT 2대", "고정노선 1대", "고정노선 2대"]
cols = [C1, C2, CF1, CF2]
rates = sorted(main["이용률_%"].unique())
xw = 0.2
for k, (an, col) in enumerate(zip(ans, cols)):
    g = main[main.안 == an].set_index("이용률_%").loc[rates]
    xs = np.arange(len(rates)) + (k - 1.5) * xw
    axes[2].bar(xs, g.총시간_평균, width=xw * 0.92, color=col, label=an, edgecolor=SURF, lw=1)
    for x, v, rj in zip(xs, g.총시간_평균, g["거절률_%"]):
        axes[2].text(x, v + 1, f"{v:.0f}" + (f"\n거절{rj:.0f}%" if rj >= 1 else ""), ha="center", va="bottom", fontsize=7.5, color=INK2)
axes[2].set_xticks(range(len(rates)), [f"{r:g}%\n({int(round(8817 * r / 100))}명)" for r in rates])
axes[2].set(title="승객 1인당 총시간(대기+승차) 평균 — 태운 승객 기준", ylabel="분")
axes[2].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
for ax in axes:
    style(ax)
for ax in axes[:2]:
    ax.set_xscale("log")
    ax.set_xticks([9, 18, 26, 44, 88, 176, 264, 441], ["9", "18", "26", "44", "88", "176", "264", "441"])
fig.text(0.01, 0.01, "가정: 20km/h, 정원 12, 승하차 1인당 1분, 최대 대기 30분, 최대 승차 min(직행×2, 30분), 새 호출 8~19시. 시나리오별 300회 반복 평균.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/DRT_v3_이용률별_대기거절_총시간.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 2. 노선 지도
roads = Roads(path="data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json")
st, snaps = Din["stops"], Din["snaps"]
sx = np.array([s["sx"] for s in snaps])
sy = np.array([s["sy"] for s in snaps])
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
loop = Din["loop"]
fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), dpi=150, facecolor=SURF)
pts_all = []
for a, c in zip(loop, loop[1:] + loop[:1]):
    p = np.array(roads.drive_path(snaps[a], snaps[c]))
    pts_all.append(p)
    axes[0].plot(p[:, 0], p[:, 1], color=CF1, lw=2, alpha=0.85, zorder=3)
    m = len(p) // 2
    if len(p) > 2:
        axes[0].annotate("", xy=p[min(m + 1, len(p) - 1)], xytext=p[m], arrowprops=dict(arrowstyle="-|>", color=CF1, lw=0, mutation_scale=10), zorder=4)
allp = np.vstack(pts_all + [np.c_[sx, sy]])
ext = np.r_[np.minimum(cl.total_bounds[:2], allp.min(0)) - 200, np.maximum(cl.total_bounds[2:], allp.max(0)) + 200]
for ax, title in [(axes[0], f"고정노선 순환: {Din['loop_m'] / 1000:.1f}km, {Din['cycle']:.0f}분 (승차 13 + 목적지 5, 정류소당 1분)"),
                  (axes[1], "DRT 권역·대기 장소 (2대: 권역 고정, 목적지 5곳 공통)")]:
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
        ax.scatter(sx[i], sy[i], s=60, color=zc[st.권역[i]], edgecolor=SURF, lw=1.5, zorder=6)
        ax.annotate(st.ID[i], (sx[i], sy[i]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
    for i in Din["DI"]:
        ax.scatter(sx[i], sy[i], s=170, marker="*", color="#e34948", edgecolor=SURF, lw=1, zorder=6)
        ax.annotate(st.이름[i].replace(".시흥4동주민센터", ""), (sx[i], sy[i]), xytext=(7, -12), textcoords="offset points", fontsize=8,
                    color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
dA, dB = Din["I"]["DEP_A"], Din["I"]["D2"]
axes[1].scatter([sx[dA]], [sy[dA]], s=160, marker="s", color=C1, edgecolor=SURF, lw=1.5, zorder=8)
axes[1].scatter([sx[dB]], [sy[dB]], s=160, marker="s", color=C2, edgecolor=SURF, lw=1.5, zorder=8)
axes[1].annotate("A 대기", (sx[dA], sy[dA]), xytext=(-8, -14), textcoords="offset points", ha="right", fontsize=9, color=INK, zorder=9,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
axes[1].annotate("B·단일 대기 (시흥사거리)", (sx[dB], sy[dB]), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=9, color=INK,
                 zorder=9, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
popA = st.loc[[i for i in Din["BI"] if st.권역[i] == "A"], "배정_70세이상_명"].sum()
popB = st.loc[[i for i in Din["BI"] if st.권역[i] == "B"], "배정_70세이상_명"].sum()
axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=C1, ms=9, label=f"권역 A 승차 (시흥3동 5곳, 배정 {popA:,.0f}명)"),
                        Line2D([], [], marker="o", ls="", color=C2, ms=9, label=f"권역 B 승차 (시흥4·5동+B01 8곳, 배정 {popB:,.0f}명)"),
                        Line2D([], [], marker="*", ls="", color="#e34948", ms=13, label="목적지 정류소 5곳"),
                        Line2D([], [], marker="s", ls="", color=MUTED, ms=9, label="차량 대기 장소")], frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "도로: OSM 주행망(관악구 난향동까지 확장, 일방통행 반영). B01은 관악구 난향15길(관악구 협의 필요). © OpenStreetMap contributors",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/DRT_v3_노선지도.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 3. 귀가 분포
h = R["cmp_h"]
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=SURF)
ax.bar(h.index - 0.2, h.iloc[:, 0], width=0.4, color=C1, label="시뮬레이션 귀가 호출 (DRT 1대, 이용률 3%)")
ax.bar(h.index + 0.2, h.iloc[:, 1], width=0.4, color="#c3c2b7", label="O-D 오는 방향 (시흥1동→3·4·5동, 60세 이상)")
ax.set(xlabel="시", ylabel="하루 중 비중 (%)", title=f"귀가 호출 시간 분포 비교 (상관 {R['corr']:.2f})")
ax.set_xticks(h.index)
ax.legend(frameon=False, fontsize=8.5)
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/DRT_v3_귀가분포비교.png", facecolor=SURF)
plt.close(fig)
print("저장: DRT_v3_이용률별_대기거절_총시간.png, DRT_v3_노선지도.png, DRT_v3_귀가분포비교.png")
