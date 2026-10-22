"""보고서 그림 3개: (1) 자치구 접근성 x 70세 이상 반납률 산점도, (2) 정류소 지도(도보 400m 새로 커버·여전히 밖), (3) 하루 이용자별 필요 차량.

실행: 프로젝트 루트에서 `python scripts/그림_보고서.py`
분석은 다시 돌리지 않음. 기존 결과 파일만 읽어 그림을 그리고, 그림 1의 70세 이상 반납률·상관만 검증된 연령구간 파일에서 집계.
출력: outputs/figures/그림1_자치구_산점도.png, 그림2_정류소지도.png, 그림3_필요차량.png (300dpi)
      outputs/figures/그림1_자치구_70세이상_반납률.csv, 그림1_상관_70세이상.csv (그림 1에 쓴 값)
그림 1
  - 반납률(70세 이상, 2025) = 70-74·75-79·80+ 자진반납 합 / 같은 구간 총면허소지자수_B(1종소계 + 2종소계 - 2종 원자)
    입력 data/verified/surrender_rate_by_gu_year_agegroup.csv (2025년만 정확값)
  - 접근성 = v3 지수의 자치구 고령인구가중평균 (data/verified/gu_accessibility_income_rate_2025.csv, gu_correlation_recheck.py와 같은 열)
  - 상관·소득 통제 편상관 = gu_correlation_recheck.py와 같은 공식(1차 편상관, t검정 df=n-3), 소득 = 같은 파일의 소득분위
  - 평균 점선 = 25개 구 단순평균
그림 2
  - 커버 판정은 금천구_접근성_DRT전후.py와 같은 함수·입력(osm_roads.Roads.walk_dist, 보행 400m)으로 수요점별로 다시 표시만 한다.
  - 1패널, 세로 6.5cm 고정·가로는 지도 비율. 격자는 그리지 않고 요약 상자 숫자만 표시
    = 접근성_DRT전후_신규400m커버.csv의 375.5명·96.4%와 같은 정의(기존 97곳 + DRT 승차 12곳), 그릴 때마다 다시 계산해 대조
  - 배경: 시흥1·3·4·5동 안만 OSM 도로망 연회색(그림_노선지도.py와 같은 관악 확장 도로망), 밖은 흰색. 시흥1동 = 목적지 허브(연한 주황)
    동 이름·범례 7.5pt, 요약 상자 9pt
그림 3: outputs/검토보완_대수별_한계이용자.csv(1~8대 한계), 검토2_단계표.csv(기본 41명·성장 83명)
"""
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path
from shapely.geometry import Point, box

OUT, FIG = "outputs", "outputs/figures"
CM = 1 / 2.54
BLUE, ORANGE, LIGHT, GRAY = "#1F5FA8", "#E8743B", "#A9C1DD", "#8C8C8C"
INK, MUTED = "#222222", "#666666"
STOP_BLUE = "#2A78D6"
# 동 이름 위치(EPSG:5179 m): 대표점에 두면 정류소·별·격자와 겹쳐서, 겹치지 않는 자리를 찾아 둔 값(중심이 각 동 안인지 그릴 때 확인)
DONG_XY = {"시흥1동": (946720, 1938980), "시흥3동": (947113, 1937816), "시흥4동": (947800, 1940020), "시흥5동": (948063, 1938579)}

names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "NanumGothic" if "NanumGothic" in names else "Malgun Gothic",
    "axes.unicode_minus": False, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": "#999999", "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장:", f"{FIG}/{name}")


def partial_corr(x, y, z, method):
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    rxy, rxz, ryz = f(x, y)[0], f(x, z)[0], f(y, z)[0]
    r_p = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    df = len(x) - 3
    t = r_p * np.sqrt(df / (1 - r_p**2))
    return r_p, 2 * stats.t.sf(abs(t), df)


# ---------------------------------------------------------------- 그림 1
def fig1():
    a = pd.read_csv("data/verified/surrender_rate_by_gu_year_agegroup.csv", encoding="utf-8-sig")
    a = a[(a.연도 == 2025) & (a.연령구간.isin(["70-74", "75-79", "80+"]))]
    r = a.groupby("구")[["자진반납건수", "총면허소지자수_B"]].sum()
    r["반납률70_%"] = r.자진반납건수 / r.총면허소지자수_B * 100
    g = pd.read_csv("data/verified/gu_accessibility_income_rate_2025.csv", encoding="utf-8-sig")
    d = g[["구", "v3_고령인구가중평균", "소득분위"]].merge(r, left_on="구", right_index=True)
    assert len(d) == 25
    d["반납률_순위"] = d["반납률70_%"].rank(ascending=False).astype(int)
    d["접근성_낮은순위"] = d["v3_고령인구가중평균"].rank().astype(int)
    d.round(4).to_csv(f"{FIG}/그림1_자치구_70세이상_반납률.csv", index=False, encoding="utf-8-sig")
    x, y, z = d["v3_고령인구가중평균"].to_numpy(), d["반납률70_%"].to_numpy(), d["소득분위"].to_numpy()
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    ppr, ppp = partial_corr(x, y, z, "pearson")
    spr, spp = partial_corr(x, y, z, "spearman")
    C = pd.DataFrame([{"기준": "70세 이상 2025, 분모 B, 접근성 v3 고령인구가중평균", "n": 25,
                       "피어슨_단순": pr, "피어슨_p": pp, "스피어만_단순": sr, "스피어만_p": sp,
                       "피어슨_소득통제편상관": ppr, "피어슨_편상관_p": ppp, "스피어만_소득통제편상관": spr, "스피어만_편상관_p": spp,
                       "반납률_25개구_단순평균_%": y.mean(), "반납률_서울합산_%": r.자진반납건수.sum() / r.총면허소지자수_B.sum() * 100,
                       "접근성_25개구_단순평균": x.mean()}])
    C.round(4).to_csv(f"{FIG}/그림1_상관_70세이상.csv", index=False, encoding="utf-8-sig")
    print(C.round(4).T.to_string())

    fig, ax = plt.subplots(figsize=(16 * CM, 9 * CM))
    gc = d.구 == "금천"
    ax.axhline(y.mean(), color=GRAY, ls=(0, (3, 3)), lw=0.8, zorder=1)
    ax.axvline(x.mean(), color=GRAY, ls=(0, (3, 3)), lw=0.8, zorder=1)
    ax.scatter(x[~gc], y[~gc], s=26, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(x[gc], y[gc], s=60, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=4)
    gx, gy = x[gc][0], y[gc][0]
    ax.annotate(f"금천구\n반납률 {gy:.2f}%(25개 구 중 {d.loc[gc, '반납률_순위'].item()}위)\n접근성 {gx:.1f}(낮은 쪽 {d.loc[gc, '접근성_낮은순위'].item()}위)",
                (gx, gy), xytext=(6, 8), textcoords="offset points", fontsize=7.5, color=INK, va="bottom", fontweight="bold")
    ax.text(x.mean(), ax.get_ylim()[1], f" 접근성 평균 {x.mean():.1f}", fontsize=6.5, color=MUTED, va="top", ha="left")
    ax.text(ax.get_xlim()[1], y.mean(), f"반납률 평균 {y.mean():.2f}% ", fontsize=6.5, color=MUTED, va="bottom", ha="right")
    ax.text(0.99, 0.03, f"Pearson r = {pr:+.2f}, p = {pp:.2f} (p > 0.05)\n소득 통제 편상관 r = {ppr:+.2f}, p = {ppp:.2f}  ·  n = 25",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("대중교통 접근성 지수(v3, 자치구 고령인구 가중평균, 높을수록 좋음)")
    ax.set_ylabel("2025년 70세 이상 면허 반납률(%)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.75, ms=5, label="자치구(25개)"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="금천구"),
                       Line2D([], [], color=GRAY, ls=(0, (3, 3)), lw=0.8, label="25개 구 단순평균")],
              loc="upper right", frameon=False)
    save(fig, "그림1_자치구_산점도.png")
    return d, C


# ---------------------------------------------------------------- 그림 2
def cell_box(code):
    a, b = int(code[2:5]), int(code[5:8])
    return box(900000 + a * 100, 1900000 + b * 100, 900000 + a * 100 + 100, 1900000 + b * 100 + 100)


def fig2():
    sys.path.insert(0, "scripts")
    from osm_roads import Roads
    dem = pd.read_csv(f"{OUT}/MCLP_v3_수요점_100m보정.csv", encoding="utf-8-sig")
    cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
    ex = cand[cand.후보유형 == "기존 버스정류소"]
    st = pd.read_csv(f"{OUT}/MCLP_v4_최종승차정류소.csv", encoding="utf-8-sig")
    dst = pd.read_csv(f"{OUT}/DRT_v3_정류소.csv", encoding="utf-8-sig")
    dst = dst[dst.구분 == "목적지"]
    roads = Roads()  # 금천구_접근성_DRT전후.py와 같은 보행망·같은 함수
    dxy = dem[["x_5179", "y_5179"]].to_numpy()
    cov_ex = np.isfinite(roads.walk_dist(ex[["x", "y"]].to_numpy(), dxy, 400)).any(axis=1)
    # 색칠 기준 = 기존 97곳 + DRT 12곳(B01 제외): 커버율 96.4%(96.37)·새로 375명(54%)과 같은 정의. 13곳으로 칠하면 숫자와 어긋남
    cov_d12 = np.isfinite(roads.walk_dist(st[st.ID != "B01"][["x_5179", "y_5179"]].to_numpy(), dxy, 400)).any(axis=1)
    w = dem.가중치_70세이상.to_numpy()
    chk = {"전체_명": w.sum(), "기존 400m 안_명": w[cov_ex].sum(), "기존 커버율_%": w[cov_ex].sum() / w.sum() * 100,
           "기존 400m 밖_명": w[~cov_ex].sum(), "DRT 12곳으로 새로 400m 안_명": w[~cov_ex & cov_d12].sum(),
           "기존+DRT 12곳 커버율_%": w[cov_ex | cov_d12].sum() / w.sum() * 100}
    ref = pd.read_csv(f"{OUT}/접근성_DRT전후_신규400m커버.csv", encoding="utf-8-sig").set_index("동").loc["합계"]
    for k, rk in [("기존 커버율_%", "기존 커버율_%"), ("기존 400m 밖_명", "기존 400m 밖_명"),
                  ("DRT 12곳으로 새로 400m 안_명", "그중 DRT 12곳(B01 제외)으로 새로 400m 안_명"), ("기존+DRT 12곳 커버율_%", "기존+DRT 12곳 커버율_%")]:
        assert abs(chk[k] - ref[rk]) < 0.06, (k, chk[k], ref[rk])
    new_pct = chk["DRT 12곳으로 새로 400m 안_명"] / chk["기존 400m 밖_명"] * 100
    print({k: round(v, 2) for k, v in chk.items()}, "신규 비율 %:", round(new_pct, 1))

    bd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
    bd["ADM_NM"] = bd.ADM_NM.str.strip()
    four = bd[bd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
    hub = four[four.ADM_NM == "시흥1동"]
    geo = {r.ADM_NM: r.geometry for r in four.itertuples()}
    board, off = st[st.ID != "B01"], st[st.ID == "B01"]
    assert len(board) == 12

    # 지도 범위: 4개 동 + 정류소. 위쪽 띠(PAD_T)에 요약 상자, 오른쪽(PAD_R, 시흥3동 동쪽 아래 빈 곳)에 범례가 들어가게 더 비움
    # 세로 6.5cm 고정, 가로 = 범위 비율
    xmin, ymin, xmax, ymax = four.total_bounds
    xmax = max(xmax, st.x_5179.max())
    PAD, PAD_T, PAD_R = 120, 900, 2150
    x0, x1, y0, y1 = xmin - PAD, xmax + PAD_R, ymin - PAD, ymax + PAD_T
    H = 6.5
    W = H * (x1 - x0) / (y1 - y0)
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")   # 시흥1·3·4·5동 밖은 흰색(도로도 그리지 않음)
    area = four.geometry.union_all()
    polys = list(area.geoms) if area.geom_type == "MultiPolygon" else [area]
    clip = PathPatch(Path.make_compound_path(*[Path(np.asarray(g.exterior.coords)) for g in polys]), transform=ax.transData,
                     facecolor="#F6F6F4", edgecolor="none", zorder=0)
    ax.add_patch(clip)
    bg = Roads(path="data/raw/OSM_도로망_시흥클러스터_관악확장_20261011.json")
    major = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}
    for keep, lw in [(lambda h: h not in major, 0.2), (lambda h: h in major, 0.7)]:
        lc = ax.add_collection(LineCollection([[bg.XY[u], bg.XY[v]] for u, v, _, _, h, _ in bg.segs if keep(h)],
                                              colors="#DCDCD8", linewidths=lw, capstyle="round", zorder=0.5))
        lc.set_clip_path(clip)
    hub.plot(ax=ax, color="#FBE1CF", alpha=0.8, lw=0, zorder=1)
    four.boundary.plot(ax=ax, color="#8F8E88", lw=0.5, zorder=3)
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    dong_txt = []
    for name, (lx, ly) in DONG_XY.items():
        assert geo[name].contains(Point(lx, ly))
        dong_txt.append(ax.text(lx, ly, name, fontsize=7.5, color=MUTED, ha="center", va="center", zorder=8, path_effects=halo))
    ax.scatter(board.x_5179, board.y_5179, marker="o", s=20, color=STOP_BLUE, edgecolor="white", linewidth=0.6, zorder=6)
    ax.scatter(off.x_5179, off.y_5179, marker="o", s=16, facecolor="white", edgecolor=GRAY, linewidth=0.8, zorder=6)
    ax.scatter(dst.x, dst.y, marker="*", s=55, color=ORANGE, edgecolor="white", linewidth=0.4, zorder=7)
    for rr in dst[dst.ID.isin(["D1", "D5"])].itertuples():
        txt, xy, ha = {"D1": ("금천구청역", (-3, -6.5), "left"), "D5": ("석수역", (-4.5, 0), "right")}[rr.ID]
        ax.annotate(txt, (rr.x, rr.y), xytext=xy, textcoords="offset points", va="center", ha=ha,
                    fontsize=7, color=INK, zorder=8, path_effects=halo)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 요약 상자(왼쪽 위 빈 곳)
    summ = ax.text(0.025, 0.97, f"커버율 {chk['기존 커버율_%']:.1f}% → {chk['기존+DRT 12곳 커버율_%']:.1f}%\n"
                   f"새로 400m 안 {chk['DRT 12곳으로 새로 400m 안_명']:,.0f}명({new_pct:.0f}%)",
                   transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="top", ha="left", linespacing=1.4, zorder=10,
                   bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#BDBDBD", lw=0.5))
    leg = ax.legend(handles=[Line2D([], [], marker="o", ls="", color=STOP_BLUE, mec="white", ms=5, label="승차 정류소 12곳"),
                             Line2D([], [], marker="*", ls="", color=ORANGE, mec="white", mew=0.4, ms=8, label="목적지 정류소 5곳"),
                             Patch(color="#FBE1CF", label="시흥1동(목적지 허브)")],
                    loc="lower right", bbox_to_anchor=(0.985, 0.07), frameon=False, fontsize=7.5, handlelength=1.3, handletextpad=0.5,
                    borderpad=0, borderaxespad=0, labelspacing=0.35)
    ax.text(0.995, 0.012, "배경: © OpenStreetMap contributors", transform=ax.transAxes, fontsize=6, color=MUTED, ha="right", va="bottom", zorder=10)

    # 겹침 점검: 요약 상자·범례·동 이름이 정류소를 덮지 않는지, 그림 밖으로 나가지 않는지
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    pts = to_px(np.r_[st[["x_5179", "y_5179"]].to_numpy(), dst[["x", "y"]].to_numpy()])
    err = []
    for nm, art in [("요약 상자", summ.get_bbox_patch()), ("범례", leg)] + [(t.get_text(), t) for t in dong_txt]:
        bb = art.get_window_extent(rd)
        if not (0 <= bb.x0 and bb.x1 <= fig.bbox.width and 0 <= bb.y0 and bb.y1 <= fig.bbox.height):
            err.append(f"{nm} 그림 밖")
        if any(bb.contains(*p) for p in pts):
            err.append(f"{nm}가 정류소를 덮음")
    if err:
        fig.savefig(os.environ.get("FIG_DEBUG", f"{FIG}/_점검실패_그림2.png"), dpi=300, facecolor="white")
        raise AssertionError(", ".join(err))
    fig.savefig(f"{FIG}/그림2_정류소지도.png", dpi=300, facecolor="white")  # 크기 고정(bbox_inches 미사용)
    plt.close(fig)
    print("저장:", f"{FIG}/그림2_정류소지도.png", f"({W:.2f} x {H}cm)")
    return chk, new_pct


# ---------------------------------------------------------------- 그림 3
def fig3():
    L = pd.read_csv(f"{OUT}/검토보완_대수별_한계이용자.csv", encoding="utf-8-sig")
    lim = L.set_index("차량_대")["한계 이용자_명"].astype(int)
    assert lim.tolist() == [26, 79, 123, 168, 203, 238, 265, 291], lim.tolist()
    S = pd.read_csv(f"{OUT}/검토2_단계표.csv", encoding="utf-8-sig")
    assert "기본 시나리오 41명" in S.loc[S.차량 == "2대", "표시"].item() and "성장 시나리오 83명" in S.loc[S.차량 == "3대", "표시"].item()

    fig, ax = plt.subplots(figsize=(16 * CM, 7 * CM))
    lo = 0
    for k, hi in lim.items():
        ax.add_patch(Rectangle((lo, 0), hi - lo, k, facecolor=LIGHT, edgecolor="white", lw=1.5, zorder=2))
        ax.text((lo + hi) / 2, k + 0.15, f"{k}대", ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(hi, -0.25, f"{hi}", ha="center", va="top", fontsize=6.3, color=MUTED)
        lo = hi
    for n, k, lab, col, dy in [(41, 2, "기본 시나리오 41명 → 2대", BLUE, 7.8), (83, 3, "성장 시나리오 83명 → 3대", ORANGE, 6.6)]:
        ax.plot([n, n], [0, dy - 0.25], color=col, lw=1.6, zorder=4)
        ax.scatter([n], [k], s=22, color=col, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(n + 3, dy, lab, color=INK, fontsize=7.5, va="center", ha="left", fontweight="bold")
        ax.plot([n, n + 2], [dy - 0.25, dy], color=col, lw=1.0)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks(range(0, 9, 2))
    ax.set_ylabel("필요 차량(대)")
    ax.set_xlabel("하루 이용자 수(명) — 막대 끝 숫자 = 차량 대수별 수용 한계", labelpad=12)
    ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.24, "서비스 기준: 거절률 5% 이하, 평균 대기 15분 이하(최대 대기 30분, 공동 운영, B01 제외). 수용 한계 = 기준을 지키는 마지막 하루 이용자 수",
            transform=ax.transAxes, fontsize=6.3, color=MUTED, va="top")
    save(fig, "그림3_필요차량.png")
    return lim


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1()
    fig2()
    fig3()
