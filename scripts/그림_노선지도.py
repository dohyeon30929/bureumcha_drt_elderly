"""보고서용 노선 지도: 고정노선 순환 vs DRT(메인 기준 §11-5-1: 2대 공동 운영, B01 제외).

실행: 프로젝트 루트에서 `python scripts/그림_노선지도.py`  (선행: 금천구_DRT_v4_시뮬.py)
좌표·노선·숫자는 금천구_DRT_v4_그림.py와 같은 입력(data/interim/DRT_v4_입력.pkl)을 그대로 씀.
- 왼쪽: 고정노선 기본안 순환(B01 제외) = Din["loop0"], 길이·주기 = loop0_m·cycle0
- 오른쪽: DRT 공동 운영. 권역·대기 장소는 그리지 않음(공동 운영은 권역 구분 없음, 빈 차는 직전 하차 위치 대기 §11-11).
  옅은 파랑 곡선 = 승차 정류소 → 시흥1동(목적지 허브, 목적지 D1~D4가 있는 동) 대표점의 연결 관계. 실제 주행 경로가 아님
- B01(관악구 난향15길)은 두 안 모두 기본안에서 제외 → 회색 빈 원
배경: 같은 OSM 도로망을 연회색 선으로 직접 그린 Positron풍 배경(라벨 없음).
  contextily 타일은 2026-10 확인 시 CartoDB=API 키 필요(워터마크), OSM 기본=403 차단, Esri 회색=빈 타일이라 쓰지 않는다.
크기: 세로 6.5cm 고정, 가로 = 두 패널 비율 + 범례 열(최대 16cm). 이 크기로 인쇄 시 제목·동 이름·범례 8~9pt, 라벨 7pt, 출처 6.5pt
출력: outputs/figures/노선지도_고정노선vsDRT.png (300dpi). 그림 제목은 보고서 캡션으로 쓴다(그림 안에 넣지 않음).
"""
import pickle
import sys

import geopandas as gpd
import matplotlib
import numpy as np

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from shapely.geometry import Point

FIG, INTERIM = "outputs/figures", "data/interim"
Din = pickle.load(open(f"{INTERIM}/DRT_v4_입력.pkl", "rb"))
INK, MUTED = "#222222", "#666666"
LINE, EDGE, STOP0, OFF = "#7a7a7a", "#8f8e88", "#4a4a4a", "#9a9a9a"
BLUE, ORANGE, HUB = "#2a78d6", "#eb6834", "#fbe1cf"
BG, ROAD = "#f6f6f4", "#dcdcd8"
MAJOR = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
FS_TITLE, FS_DONG, FS_LEG, FS_LABEL, FS_SRC = 9, 8, 8, 7, 6.5
CM = 1 / 2.54

roads = Roads(path="data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json")
st, snaps = Din["stops"], Din["snaps"]
sx = np.array([s["sx"] for s in snaps])
sy = np.array([s["sy"] for s in snaps])
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
hub = cl[cl.ADM_NM == "시흥1동"]
hub_pt = hub.geometry.iloc[0].representative_point()
loop = Din["loop0"]
B01 = int(st.index[st.ID == "B01"][0])
BOARD = [i for i in Din["BI"] if i != B01]   # 승차 12곳
assert len(BOARD) == 12 and all(i in loop for i in BOARD)
# 목적지 라벨: (글자, 오프셋 pt, 정렬). 금천구청역은 무지개아파트 별, 석수역은 B06과 겹치지 않게
DEST_LABEL = {"D1": ("금천구청역", (-3, -6.5), "left"), "D5": ("석수역", (-4.5, 0), "right")}
# 동 이름 위치(EPSG:5179 m). 8pt 글자 폭이 지도에서 약 900m라 대표점에 두면 정류소·별과 겹쳐 정류소 없는 자리로 옮김(중심이 각 동 안인지 확인)
DONG_XY = {"시흥1동": (946720, 1938980), "시흥3동": (947750, 1938250), "시흥4동": (947800, 1940020), "시흥5동": (948300, 1938600)}

paths = [np.array(roads.drive_path(snaps[a], snaps[c])) for a, c in zip(loop, loop[1:] + loop[:1])]
allp = np.vstack(paths + [np.c_[sx, sy]])
ext = np.r_[np.minimum(cl.total_bounds[:2], allp.min(0)) - 200, np.maximum(cl.total_bounds[2:], allp.max(0)) + 200]

# ---- 크기(cm): 세로 6.5 고정, 지도 높이에서 패널 가로를 비율로 계산
H, TOP, BOT, L, GAP, LEGW, R = 6.5, 0.62, 0.42, 0.1, 0.9, 4.6, 0.1
MH = H - TOP - BOT
MW = MH * (ext[2] - ext[0]) / (ext[3] - ext[1])
W = L + MW + GAP + MW + 0.25 + LEGW + R
assert W <= 16, f"가로 {W:.2f}cm > 16cm"
fig = plt.figure(figsize=(W * CM, H * CM), dpi=300, facecolor="white")
axes = [fig.add_axes([(L + k * (MW + GAP)) / W, BOT / H, MW / W, MH / H]) for k in range(2)]

titles = [f"고정노선: 1개 순환 {Din['loop0_m'] / 1000:.1f}km·{Din['cycle0']:.0f}분", "DRT: 2대 공동 운영, 호출 시 직행"]
for ax, title in zip(axes, titles):
    ax.set_facecolor(BG)
    for keep, lw in [(lambda h: h not in MAJOR, 0.2), (lambda h: h in MAJOR, 0.7)]:
        seg = [[roads.XY[u], roads.XY[v]] for u, v, _, _, h, _ in roads.segs if keep(h)]
        ax.add_collection(LineCollection(seg, colors=ROAD, linewidths=lw, capstyle="round", zorder=0))
    hub.plot(ax=ax, color=HUB, alpha=0.75, lw=0, zorder=1)
    cl.boundary.plot(ax=ax, color=EDGE, lw=0.5, zorder=2)
    for r in cl.itertuples():
        assert r.geometry.contains(Point(DONG_XY[r.ADM_NM]))
        ax.text(*DONG_XY[r.ADM_NM], r.ADM_NM, fontsize=FS_DONG, color=MUTED, ha="center", va="center", zorder=5, path_effects=HALO)
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=FS_TITLE, color=INK, fontweight="bold", pad=3)

# 왼쪽: 고정노선 순환 (회색 선, 진행 방향 작은 화살표)
for p in paths:
    axes[0].plot(p[:, 0], p[:, 1], color=LINE, lw=1.2, solid_capstyle="round", zorder=3)
    m = len(p) // 2
    if len(p) > 2:
        axes[0].annotate("", xy=p[min(m + 1, len(p) - 1)], xytext=p[m],
                         arrowprops=dict(arrowstyle="-|>", color=LINE, lw=0, mutation_scale=5), zorder=4)


# 오른쪽: 승차 정류소 → 시흥1동 허브 연결(곡선, 노선 아님). 허브 대표점 방향으로 휘어 가다가 시흥1동 경계에 닿으면 끝냄
# (한 점에 모으면 그 점의 목적지 별로 가는 것처럼 읽혀서, '허브 동으로 들어간다'까지만 표시)
hub_geom = hub.geometry.iloc[0]


def curve(x0, y0, x1, y1, bend=0.18, n=200):
    mx, my, dx, dy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    c = np.array([mx - dy * bend, my + dx * bend])
    t = np.linspace(0, 1, n)[:, None]
    q = (1 - t) ** 2 * np.array([x0, y0]) + 2 * (1 - t) * t * c + t ** 2 * np.array([x1, y1])
    inside = [hub_geom.contains(Point(p)) for p in q]
    return q[: inside.index(True) + 1] if True in inside else q


for i in BOARD:
    q = curve(sx[i], sy[i], hub_pt.x, hub_pt.y)
    axes[1].plot(q[:, 0], q[:, 1], color=BLUE, lw=0.8, alpha=0.25, solid_capstyle="round", zorder=3)

# 정류소: 승차 12곳(왼쪽 진회색, 오른쪽 파랑) 흰 테두리 원, B01 회색 빈 원, 목적지 주황 별(라벨은 금천구청역·석수역만)
for ax, col in [(axes[0], STOP0), (axes[1], BLUE)]:
    ax.scatter(sx[BOARD], sy[BOARD], s=20, color=col, edgecolor="white", lw=0.6, zorder=6)
    ax.scatter([sx[B01]], [sy[B01]], s=16, facecolor="white", edgecolor=OFF, lw=0.8, zorder=6)
    for i in Din["DI"]:
        ax.scatter(sx[i], sy[i], s=55, marker="*", color=ORANGE, edgecolor="white", lw=0.4, zorder=7)
        if st.ID[i] in DEST_LABEL:
            txt, off, ha = DEST_LABEL[st.ID[i]]
            ax.annotate(txt, (sx[i], sy[i]), xytext=off, textcoords="offset points", va="center", ha=ha,
                        fontsize=FS_LABEL, color=INK, zorder=8, path_effects=HALO)

# 공통 범례(오른쪽 열): 고정노선 / DRT / 공통 세 묶음을 위에서부터 쌓음
groups = [("고정노선", [Line2D([], [], color=LINE, lw=1.2, label="순환 노선"),
                        Line2D([], [], marker="o", ls="", color=STOP0, markeredgecolor="white", ms=5, label="승차 정류소 12곳")]),
          ("DRT", [Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", ms=5, label="승차 정류소 12곳"),
                   Line2D([], [], color=BLUE, lw=0.8, alpha=0.25, label="목적지 직행 연결(노선 아님)")]),
          ("공통", [Line2D([], [], marker="*", ls="", color=ORANGE, markeredgecolor="white", ms=8, label="목적지 정류소 5곳"),
                   Patch(facecolor=HUB, edgecolor="none", label="시흥1동(목적지 허브)"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=OFF, ms=4.5, label="B01: 관악구 경계(협의 후 추가)")])]
fig.canvas.draw()
rd = fig.canvas.get_renderer()
leg_x, leg_y, legs = (L + 2 * MW + GAP + 0.25) / W, (BOT + MH) / H, []
for name, hs in groups:
    lg = fig.legend(handles=hs, title=name, alignment="left", loc="upper left", bbox_to_anchor=(leg_x, leg_y), frameon=False,
                    fontsize=FS_LEG, title_fontproperties={"size": FS_LEG, "weight": "bold"}, handlelength=1.6, handletextpad=0.5,
                    borderpad=0, borderaxespad=0, labelspacing=0.35)
    legs.append(lg)
    leg_y -= lg.get_window_extent(rd).height / fig.bbox.height + 0.12 / H
src = fig.text(L / W, 0.12 / H, "노선·배경: OSM 도로망(일방통행 반영) © OpenStreetMap contributors. B01은 관악구 경유 약 4km 우회라 두 안 모두 기본안에서 뺌",
         fontsize=FS_SRC, color=MUTED, va="bottom")

# 겹침 점검: 제목·범례가 그림 밖으로 나가거나 서로 겹치지 않는지
fig.canvas.draw()
rd = fig.canvas.get_renderer()
fw = fig.bbox.width
boxes = {"왼쪽 제목": axes[0]._left_title.get_window_extent(rd), "오른쪽 제목": axes[1]._left_title.get_window_extent(rd),
         "출처": src.get_window_extent(rd), **{f"범례 {g[0]}": lg.get_window_extent(rd) for g, lg in zip(groups, legs)}}
for k, b in boxes.items():
    print(f"{k}: x {b.x0 / fw * W:.2f}~{b.x1 / fw * W:.2f}cm")
assert boxes["왼쪽 제목"].x1 < boxes["오른쪽 제목"].x0
assert all(b.x1 <= fw for b in boxes.values()), "그림 밖으로 나감"
assert legs[-1].get_window_extent(rd).y0 > src.get_window_extent(rd).y1, "범례가 출처와 겹침"
fig.savefig(f"{FIG}/노선지도_고정노선vsDRT.png", facecolor="white")
plt.close(fig)
print(f"저장: {FIG}/노선지도_고정노선vsDRT.png ({W:.2f} x {H}cm)")
