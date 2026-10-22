"""3단계 DRT 노선 설계: 보행 400m 기준 승차 14곳 + 목적지 5곳, 비대칭 TSP·호출 시뮬레이션·단일 vs 권역 분리.

실행: 프로젝트 루트에서 `python scripts/금천구_DRT_TSP.py`  (선행: 금천구_MCLP_보행.py)
정류소
  - 승차 14곳: outputs/MCLP_보행400_p14_승차정류소.csv. 권역 A = 시흥3동(6곳), B = 시흥4·5동(8곳)
    '석수역'(시흥3동 버스정류소)은 '석수역 버스정류소(승차)'로 표기해 목적지 석수역(1호선, 안양시)과 구분
  - 목적지 5곳: 금천구청역, 시흥사거리(대기 장소), 현대시장.시흥4동주민센터, 무지개아파트, 석수역(1호선)
  - 도로거리: osm_roads.Roads.drive_matrix (일방통행 반영, 행=출발 열=도착)
가정: 평균 운행속도 20km/h, 정류소당 정차 1분(호출 승차 정류소·하차 목적지에서만, 대기 장소 출발 시 제외)
(1) 전체 순회: 시흥사거리 출발·도착, 담당 승차 정류소 + 해당 목적지 전부 1회씩 방문하는 최단 비대칭 순회(MTZ 정수계획, CBC)
(2) 호출 시뮬레이션: 운행 1회 = 승차 정류소 k곳(k=2,3,4) 호출, 정류소별 보행 배정 고령인구 비례·비복원 추첨, 1곳당 승객 1명
    편도: 시흥사거리 -> 호출 정류소들(순서는 총거리 최소가 되게 전수 탐색) -> 목적지 1곳
    목적지: 목적지 정류소 300m 내 병의원·복지시설 수 비례 추첨. 단일안은 5곳 중, 분리안은 권역별 목적지 중 따로 추첨
    분리안: 같은 호출을 권역별로 나눠 차량 2대가 동시에 처리. 운행거리 = 두 대 합, 소요시간 = 늦게 끝나는 차량 기준
    고정노선: 담당 승차 정류소를 (1)의 순서로 전부 지나간 뒤 목적지로. 호출 정류소에서만 정차
    승차시간 = 자기 정류소 출발 -> 목적지 도착 (중간 정차 포함, 본인 승하차 1분 제외)
"""
import itertools
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, "scripts")
from osm_roads import Roads  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "outputs"
SPEED = 20 / 3.6       # m/s
DWELL = 60.0           # s
KS, NREP, SEED = [2, 3, 4], 1000, 42
DEPOT = "시흥사거리"
ZONE = {"시흥3동": "A", "시흥4동": "B", "시흥5동": "B"}
ZONE_DEST = {"A": ["시흥사거리", "석수역"], "B": ["현대시장.시흥4동주민센터", "시흥사거리"]}
DEST_R = 300

# ---------------------------------------------------------------- 정류소·행렬
b = pd.read_csv(f"{OUT}/MCLP_보행400_p14_승차정류소.csv", encoding="utf-8-sig")
b["표시이름"] = b.표시이름.replace({"석수역": "석수역 버스정류소(승차)"})
b["권역"] = b.동.map(ZONE)
b = b.sort_values(["권역", "동", "배정_70세이상_명(보행)"], ascending=[True, True, False]).reset_index(drop=True)
fac = pd.read_csv("data/interim/MCLP_시설_중복제거.csv", encoding="utf-8-sig")
fixed = pd.read_csv(f"{OUT}/MCLP_고정목적지정류소.csv", encoding="utf-8-sig")
mu = fac[(fac.시설명 == "무지개아파트") & (fac.행정동 == "시흥1동")].iloc[0]
ss = fac[(fac.시설명 == "석수역") & (fac.유형 == "지하철역(연계)")].iloc[0]
dests = pd.DataFrame([(r.이름, r.x, r.y, "시흥1동") for r in fixed.itertuples()] +
                     [("무지개아파트", mu.x_5179, mu.y_5179, "시흥1동"), ("석수역", ss.x_5179, ss.y_5179, "안양시(서울 밖)")],
                     columns=["이름", "x", "y", "동"])
dest_fac = fac[fac.유형.isin(["병의원", "사회복지시설"]) & (fac.시흥클러스터 == "내부")][["x_5179", "y_5179"]].to_numpy()
dests["반경300m_목적지수"] = [int((np.hypot(*(dest_fac - (x, y)).T) <= DEST_R).sum()) for x, y in zip(dests.x, dests.y)]
stops = pd.concat([
    pd.DataFrame({"구분": "승차", "권역": b.권역, "이름": b.표시이름, "동": b.동, "x": b.x_5179, "y": b.y_5179,
                  "배정_70세이상_명": b["배정_70세이상_명(보행)"], "반경300m_목적지수": np.nan}),
    pd.DataFrame({"구분": "목적지", "권역": "", "이름": dests.이름, "동": dests.동, "x": dests.x, "y": dests.y,
                  "배정_70세이상_명": 0, "반경300m_목적지수": dests.반경300m_목적지수})], ignore_index=True)
stops.insert(0, "정류소ID", [f"S{i + 1:02d}" for i in range(len(stops))])
roads = Roads()
snaps = [roads.snap(x, y) for x, y in zip(stops.x, stops.y)]
stops["스냅거리_m"] = [round(s["스냅거리_m"], 1) for s in snaps]
ll = gpd.GeoSeries(gpd.points_from_xy([s["sx"] for s in snaps], [s["sy"] for s in snaps]), crs=5179).to_crs(4326)
stops["위도"], stops["경도"] = ll.y.round(6), ll.x.round(6)
M = roads.drive_matrix(snaps)
ids = stops.정류소ID.tolist()
pd.DataFrame(np.round(M), index=ids, columns=ids).to_csv(f"{OUT}/DRT_도로거리행렬_19x19_m.csv", encoding="utf-8-sig")
stops.drop(columns=["x", "y"]).to_csv(f"{OUT}/DRT_정류소19.csv", index=False, encoding="utf-8-sig")
off = ~np.eye(len(ids), dtype=bool)
print(f"[행렬] {len(ids)}x{len(ids)}: 도달 불가 {np.isinf(M[off]).sum()}칸, 왕복 비대칭 >100m "
      f"{(np.abs(M - M.T)[np.triu_indices(len(ids), 1)] > 100).sum()}쌍")
idx = {n: i for i, n in enumerate(stops.이름)}
BI = stops.index[stops.구분 == "승차"].to_numpy()
DI = stops.index[stops.구분 == "목적지"].to_numpy()
depot = idx[DEPOT]

# ---------------------------------------------------------------- 권역별 목적지 거리 재확인
rows = []
for z in ["A", "B"]:
    zi = stops.index[stops.권역 == z]
    w = stops.loc[zi, "배정_70세이상_명"].to_numpy()
    for d in DI:
        dd = M[zi, d]
        rows.append({"권역": z, "목적지": stops.이름[d], "인구가중평균_m": round(np.average(dd, weights=w)), "최소_m": round(dd.min()),
                     "최대_m": round(dd.max()), "반경300m_목적지수": int(stops.반경300m_목적지수[d]),
                     "현재구성": "Y" if stops.이름[d] in ZONE_DEST[z] else ""})
zd = pd.DataFrame(rows)
zd.to_csv(f"{OUT}/DRT_권역별_목적지거리_14곳안.csv", index=False, encoding="utf-8-sig")
print("\n[권역별 승차 -> 목적지 도로거리]")
print(zd.to_string(index=False))
det = pd.DataFrame(np.round(M[np.ix_(BI, DI)]), columns=stops.이름[DI].values)
det.insert(0, "승차", stops.이름[BI].values)
det.insert(0, "권역", stops.권역[BI].values)
det.to_csv(f"{OUT}/DRT_승차to목적지_14곳안_m.csv", index=False, encoding="utf-8-sig")
print(det.to_string(index=False))


# ---------------------------------------------------------------- (1) 전체 순회 (비대칭 TSP, MTZ)
def atsp(nodes, start):
    nodes = [start] + [n for n in nodes if n != start]
    n = len(nodes)
    m = pulp.LpProblem("ATSP", pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for i in range(n) for j in range(n) if i != j}
    u = {i: pulp.LpVariable(f"u_{i}", 1, n - 1) for i in range(1, n)}
    m += pulp.lpSum(M[nodes[i], nodes[j]] * v for (i, j), v in x.items())
    for i in range(n):
        m += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        m += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=600))
    nxt = {i: j for (i, j), v in x.items() if v.value() > 0.5}
    seq, cur = [0], 0
    while True:
        cur = nxt[cur]
        if cur == 0:
            break
        seq.append(cur)
    order = [nodes[k] for k in seq] + [nodes[0]]
    return pulp.LpStatus[m.status], order, sum(M[a, c] for a, c in zip(order[:-1], order[1:]))


plans = {"단일(1대)": {"승차": list(BI), "목적지": list(DI)},
         "분리 A": {"승차": list(stops.index[stops.권역 == "A"]), "목적지": [idx[d] for d in ZONE_DEST["A"]]},
         "분리 B": {"승차": list(stops.index[stops.권역 == "B"]), "목적지": [idx[d] for d in ZONE_DEST["B"]]}}
tours, trow = {}, []
for k, pl in plans.items():
    st, order, dist = atsp(pl["승차"] + pl["목적지"], depot)
    tours[k] = order
    nstop = len(order) - 2  # 출발·도착 대기장소 제외
    trow.append({"안": k, "방문 정류소": nstop, "거리_km": round(dist / 1000, 2), "소요시간_분": round(dist / SPEED / 60 + nstop * DWELL / 60, 1),
                 "상태": st, "순서": " → ".join(stops.정류소ID[o] for o in order)})
tdf = pd.DataFrame(trow)
both = tdf[tdf.안.str.startswith("분리")]
tdf = pd.concat([tdf, pd.DataFrame([{"안": "분리(2대) 합계/최대", "방문 정류소": both["방문 정류소"].sum(), "거리_km": both.거리_km.sum(),
                                     "소요시간_분": both.소요시간_분.max(), "상태": "", "순서": "거리=합계, 시간=느린 차량"}])], ignore_index=True)
tdf.to_csv(f"{OUT}/DRT_전체순회.csv", index=False, encoding="utf-8-sig")
print("\n[(1) 전체 순회]")
print(tdf.drop(columns="순서").to_string(index=False))
for k in plans:
    print(f"  {k}: " + " → ".join(stops.이름[o] for o in tours[k]))


# ---------------------------------------------------------------- (2) 호출 시뮬레이션
def trip_drt(calls, dest):
    """최적 순서. 반환: 거리(m), 소요시간(s), 승객별 승차시간(s) 리스트"""
    best = None
    for perm in itertools.permutations(calls):
        seq = [depot, *perm, dest]
        d = sum(M[a, c] for a, c in zip(seq[:-1], seq[1:]))
        if best is None or d < best[0]:
            best = (d, seq)
    return evaluate(best[1], set(calls))


def trip_fixed(order, calls, dest):
    return evaluate([depot, *order, dest], set(calls))


def evaluate(seq, calls):
    """seq: 대기장소 -> ... -> 목적지. calls 정류소에서만 1분 정차, 목적지 1분 하차 정차."""
    t, depart, dist = 0.0, {}, 0.0
    for a, c in zip(seq[:-1], seq[1:]):
        dist += M[a, c]
        t += M[a, c] / SPEED
        if c in calls:
            t += DWELL
            depart[c] = t
    arrive = t
    t += DWELL  # 목적지 하차
    return dist, t, [arrive - depart[c] for c in calls]


fixed_order = {k: [o for o in tours[k][1:-1] if stops.구분[o] == "승차"] for k in plans}
w_all = stops.loc[BI, "배정_70세이상_명"].to_numpy(float)
dw = lambda ds: np.array([stops.반경300m_목적지수[d] for d in ds], float)
rng = np.random.default_rng(SEED)
recs = []
for k in KS:
    for rep in range(NREP):
        calls = list(rng.choice(BI, size=k, replace=False, p=w_all / w_all.sum()))
        d_single = rng.choice(DI, p=dw(DI) / dw(DI).sum())
        dz = {z: rng.choice([idx[n] for n in ZONE_DEST[z]], p=dw([idx[n] for n in ZONE_DEST[z]]) / dw([idx[n] for n in ZONE_DEST[z]]).sum())
              for z in "AB"}
        for mode, fn in [("DRT", trip_drt), ("고정노선", None)]:
            # 단일
            if mode == "DRT":
                d, tt, ride = trip_drt(calls, d_single)
            else:
                d, tt, ride = trip_fixed(fixed_order["단일(1대)"], calls, d_single)
            recs.append({"k": k, "rep": rep, "안": "단일(1대)", "방식": mode, "운행거리_km": d / 1000, "소요시간_분": tt / 60,
                         "1인당승차시간_분": np.mean(ride) / 60, "최대승차시간_분": np.max(ride) / 60, "운행차량": 1})
            # 분리
            D_, T_, R_, used = 0.0, 0.0, [], 0
            for z, key in [("A", "분리 A"), ("B", "분리 B")]:
                cz = [c for c in calls if stops.권역[c] == z]
                if not cz:
                    continue
                used += 1
                d, tt, ride = trip_drt(cz, dz[z]) if mode == "DRT" else trip_fixed(fixed_order[key], cz, dz[z])
                D_, T_, R_ = D_ + d, max(T_, tt), R_ + ride
            recs.append({"k": k, "rep": rep, "안": "분리(2대)", "방식": mode, "운행거리_km": D_ / 1000, "소요시간_분": T_ / 60,
                         "1인당승차시간_분": np.mean(R_) / 60, "최대승차시간_분": np.max(R_) / 60, "운행차량": used})
sim = pd.DataFrame(recs)
sim.to_csv("data/interim/DRT_시뮬레이션_원자료.csv", index=False, encoding="utf-8-sig")
q90 = lambda s: s.quantile(0.9)
agg = sim.groupby(["k", "안", "방식"]).agg(
    운행거리_평균=("운행거리_km", "mean"), 운행거리_중앙=("운행거리_km", "median"), 운행거리_p90=("운행거리_km", q90),
    소요시간_평균=("소요시간_분", "mean"), 소요시간_중앙=("소요시간_분", "median"), 소요시간_p90=("소요시간_분", q90),
    승차시간_평균=("1인당승차시간_분", "mean"), 승차시간_중앙=("1인당승차시간_분", "median"), 승차시간_p90=("1인당승차시간_분", q90),
    평균운행차량=("운행차량", "mean")).round(2).reset_index()
agg.to_csv(f"{OUT}/DRT_호출시뮬레이션_요약.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(f"\n[(2)(3) 호출 시뮬레이션] k별 {NREP}회, 거리 km / 시간 분")
print(agg.to_string(index=False))
# DRT 효율: 고정노선 대비 감소율
piv = agg.pivot_table(index=["k", "안"], columns="방식", values=["운행거리_평균", "소요시간_평균", "승차시간_평균"])
eff = pd.DataFrame({f"{m}_DRT감소율_%": ((1 - piv[(m, "DRT")] / piv[(m, "고정노선")]) * 100).round(1)
                    for m in ["운행거리_평균", "소요시간_평균", "승차시간_평균"]}).reset_index()
eff.to_csv(f"{OUT}/DRT_vs_고정노선_효율.csv", index=False, encoding="utf-8-sig")
print("\n[DRT가 고정노선보다 줄이는 비율(평균 기준)]")
print(eff.to_string(index=False))

# ---------------------------------------------------------------- 지도
bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")
cl = bnd[bnd.ADM_CD.isin(["11180570", "11180590", "11180600", "11180610"])]
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CA, CB, CS, FIX = "#2a78d6", "#1baf7a", "#2a78d6", "#eb6834"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sx = np.array([s["sx"] for s in snaps])
sy = np.array([s["sy"] for s in snaps])


PATH_PTS = []


def draw(ax, order, color, lw=2.2):
    for a, c in zip(order[:-1], order[1:]):
        pts = np.array(roads.drive_path(snaps[a], snaps[c]))
        PATH_PTS.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=0.85, zorder=3, solid_capstyle="round")
        mid = len(pts) // 2
        if len(pts) > 2:
            ax.annotate("", xy=pts[mid + 1 if mid + 1 < len(pts) else mid], xytext=pts[mid],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=11), zorder=4)


def base(ax, title):
    ax.set_facecolor(SURF)
    cl.boundary.plot(ax=ax, color="#c3c2b7", lw=1, zorder=1)
    for r in cl.itertuples():
        pt = r.geometry.representative_point()
        ax.text(pt.x - 250, pt.y + 150, r.ADM_NM, fontsize=10, color=MUTED, ha="center", zorder=2,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)


def stops_layer(ax, colors):
    for i in BI:
        ax.scatter(sx[i], sy[i], s=60, color=colors[stops.권역[i]], edgecolor=SURF, lw=1.5, zorder=6)
        ax.annotate(stops.정류소ID[i], (sx[i], sy[i]), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=INK, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])
    for i in DI:
        ax.scatter(sx[i], sy[i], s=170, marker="*" if i != depot else "s", color=FIX, edgecolor=SURF, lw=1, zorder=6)
        ax.annotate(stops.이름[i].replace(".시흥4동주민센터", ""), (sx[i], sy[i]), xytext=(7, -11), textcoords="offset points",
                    fontsize=8, color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=2.5, foreground=SURF)])


r1 = tdf.set_index("안")
fig, axes = plt.subplots(1, 2, figsize=(14, 9), dpi=150, facecolor=SURF)
base(axes[0], f"(1) 단일 노선 전체 순회: {r1.loc['단일(1대)', '거리_km']}km, {r1.loc['단일(1대)', '소요시간_분']}분\n"
              f"(승차 14 + 목적지 5, 시흥사거리 출발·도착)")
draw(axes[0], tours["단일(1대)"], CS)
stops_layer(axes[0], {"A": CS, "B": CS})
base(axes[1], f"(3) 권역 분리 전체 순회: A {r1.loc['분리 A', '거리_km']}km·{r1.loc['분리 A', '소요시간_분']}분 / "
              f"B {r1.loc['분리 B', '거리_km']}km·{r1.loc['분리 B', '소요시간_분']}분\n(A: 시흥사거리+석수역, B: 현대시장+시흥사거리)")
draw(axes[1], tours["분리 A"], CA)
draw(axes[1], tours["분리 B"], CB)
stops_layer(axes[1], {"A": CA, "B": CB})
allp = np.vstack(PATH_PTS + [np.c_[sx, sy]])
ext = np.r_[np.minimum(cl.total_bounds[:2], allp.min(0)) - 200, np.maximum(cl.total_bounds[2:], allp.max(0)) + 200]
for ax in axes:
    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
axes[1].legend(handles=[Line2D([], [], color=CA, lw=2.5, label="권역 A 차량 (시흥3동)"), Line2D([], [], color=CB, lw=2.5, label="권역 B 차량 (시흥4·5동)"),
                        Line2D([], [], marker="s", ls="", color=FIX, ms=9, label="대기 장소(시흥사거리)"),
                        Line2D([], [], marker="*", ls="", color=FIX, ms=12, label="목적지 정류소")], frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "도로거리: OSM 주행망(일방통행 반영). 가정: 20km/h, 정류소당 1분 정차. 정류소 번호는 DRT_정류소19.csv 참조. © OpenStreetMap contributors",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/DRT_노선지도_전체순회_단일vs분리.png", facecolor=SURF)
plt.close(fig)
print(f"\n저장: {OUT}/DRT_정류소19.csv, DRT_도로거리행렬_19x19_m.csv, DRT_권역별_목적지거리_14곳안.csv, DRT_승차to목적지_14곳안_m.csv, "
      f"DRT_전체순회.csv, DRT_호출시뮬레이션_요약.csv, DRT_vs_고정노선_효율.csv, DRT_노선지도_전체순회_단일vs분리.png")
