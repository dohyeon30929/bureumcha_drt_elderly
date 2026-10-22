"""금천구 핀포인트 분석 (70세 이상 기준, 독립 심화분석). 방법론: docs/methodology.md 10번.

실행: 프로젝트 루트에서 `python scripts/금천구_핀포인트.py`
입력:
  - data/raw/ 생활이동 수단별 zip(seoul_trans_admdong1_in_*.csv, 2026년 8월 31일치)
  - data/raw/ 생활이동 목적별 zip(seoul_purpose_admdong1_in_*.csv, 같은 기간)
  - data/verified/행정동_코드매핑_생활이동_경계.csv (조인키)
  - data/verified/접근성_지수_v3_387동.csv
  - data/verified/소득_가구수_행정동_정리.csv (소득 필터 생략 검증용)
출력 (outputs/):
  - 금천구_70세_귀가시간대_분포.csv      2단계: 귀가(move_purpose=3) 도착 시간대 분포
  - 금천구_핀포인트_70세기준_결과.csv     3단계: 10개 동 최종 테이블
  - 금천구_우선순위점수_막대.png, 금천구_차량의존x접근성_산점도.png

정의
  - 고령층 = male_70_cnt + feml_70_cnt (70세 이상). 서울 9곳(60세 이상)과 순위 직접비교 금지
  - 귀가시간대 = 16~21시(hour 16~20). time_cd는 정시(0~23)와 러시아워 20분 단위(700~1940)가 섞여 hour로 합침
  - move_trans: 4=광역버스 5=일반버스 6=지하철 7=도보 8=차량, 분모 = 1~9 전체
  - 우선순위점수 = 차량비율 - (버스+지하철)비율
  - 표본 컷 = 서울 426개 동(경계 기준) 70세 이상 16~21시 31일 합의 하위 10% (np.percentile, 선형보간)
    -> 컷 미만 동은 제외하지 않고 '표본 상대적으로 적음' 플래그
  - 선별(A안) = 금천 내 점수 상위 50%(5개) x v3 서울 하위 25%. 소득 필터는 생략(결과 불변을 아래에서 검증)
"""
import csv
import glob
import io
import zipfile
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

GU = "금천구"
HOURS = range(16, 21)
E70 = ("male_70_cnt", "feml_70_cnt")
E60 = ("male_60_cnt", "feml_60_cnt")
CUT_PCT = 10
TOP_N = 5
INCOME_LOW_MAX = 6
OUT = "outputs"


def hour(time_cd):
    t = int(time_cd)
    return t // 100 if t >= 100 else t


def rows(zip_kw):
    zp = [p for p in glob.glob("data/raw/*.zip") if "생활이동" in p and zip_kw in p][0]
    with zipfile.ZipFile(zp) as z:
        for name in sorted(z.namelist()):
            with z.open(name) as f:
                yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))


def val(r, cols):
    return sum(float(r[c] or 0) for c in cols)


xw = {r["mois_admdong_cd"]: (r["bnd_admdong_cd"], r["sigungu"], r["admdong_nm"])
      for r in csv.DictReader(open("data/verified/행정동_코드매핑_생활이동_경계.csv", encoding="utf-8-sig"))}
names = {b: (sgg, nm) for b, sgg, nm in xw.values()}

# ---- 수단별 파일 1회 순회: 서울 전체 표본 분포, 금천 동별 수단, 금천 동별·일별 수단
seoul70 = defaultdict(float)                                # 경계코드 -> 70+ 16~21시 31일 합
mode = defaultdict(lambda: defaultdict(float))              # 경계코드 -> move_trans -> 70+
daily = defaultdict(lambda: defaultdict(float))             # (경계코드, 일자) -> move_trans -> 70+
for r in rows("수단"):
    if r["d_admdong_cd"] not in xw or hour(r["time_cd"]) not in HOURS:
        continue
    b, sgg, _ = xw[r["d_admdong_cd"]]  # 경계 코드 기준 합산(용신동 두 코드 합쳐짐)
    v = val(r, E70)
    seoul70[b] += v
    if sgg == GU:
        mode[b][r["move_trans"]] += v
        daily[(b, r["etl_ymd"])][r["move_trans"]] += v

# ---- 목적별 파일 1회 순회: 귀가 도착 시간대 분포, 시간대별 귀가비중
home_gu, all_gu, home_s70, home_s60 = (defaultdict(float) for _ in range(4))
for r in rows("성 연령별"):
    if r["d_admdong_cd"] not in xw:
        continue
    h = hour(r["time_cd"])
    v70 = val(r, E70)
    is_gu = xw[r["d_admdong_cd"]][1] == GU
    if is_gu:
        all_gu[h] += v70
    if r["move_purpose"] != "3":
        continue
    home_s70[h] += v70
    home_s60[h] += v70 + val(r, E60)
    if is_gu:
        home_gu[h] += v70

# ---- 2단계: 귀가 도착 시간대 분포
dist = pd.DataFrame({"금천_70+": pd.Series(home_gu), "서울_70+": pd.Series(home_s70),
                     "서울_60+": pd.Series(home_s60)}).sort_index().fillna(0)
share = dist / dist.sum()
out_dist = pd.concat([dist.round(0), (share * 100).round(1).add_suffix("_비중%")], axis=1)
out_dist.index.name = "시"
out_dist.to_csv(f"{OUT}/금천구_70세_귀가시간대_분포.csv", encoding="utf-8-sig")
h_gu, a_gu = pd.Series(home_gu).sort_index(), pd.Series(all_gu).sort_index()
print(f"[2단계] 금천 70+ 귀가 피크 {share['금천_70+'].idxmax()}시, 16~21시 포함 "
      f"{h_gu.loc[16:20].sum() / h_gu.sum():.1%}, 구간 내 귀가비중 {h_gu.loc[16:20].sum() / a_gu.loc[16:20].sum():.1%}")

# ---- 1단계: 표본 컷
cut = np.percentile(list(seoul70.values()), CUT_PCT)
print(f"[1단계] 서울 {len(seoul70)}개 동 70+ 16~21시 31일 합 하위 {CUT_PCT}% 컷 = {cut:,.0f}")


# ---- 3단계: 수단비율, 우선순위점수
def shares(m):
    tot = sum(m.values())
    g = lambda *k: sum(m.get(x, 0) for x in k) / tot
    return tot, g("8"), g("4", "5"), g("6"), g("7"), g("1", "2", "3", "9")


rec = []
for b, m in mode.items():
    tot, car, bus, sub, walk, etc = shares(m)
    rec.append({"경계코드": b, "동이름": names[b][1], "70세이상_16-21시_합계": round(tot, 1), "차량비율": car, "버스비율": bus,
                "지하철비율": sub, "도보비율": walk, "기타비율(1·2·3·9)": etc})
d = pd.DataFrame(rec)
d["대중교통비율"] = d.버스비율 + d.지하철비율
d["우선순위점수"] = d.차량비율 - d.대중교통비율
d = d.sort_values("우선순위점수", ascending=False).reset_index(drop=True)
d["우선순위_순위"] = d.index + 1

# 일별 점수 안정성
dl = pd.DataFrame([{"경계코드": b, "일자": day, "점수": (lambda t: (t[1] - t[2] - t[3]) * 100)(shares(m))}
                   for (b, day), m in daily.items()])
dl["일별순위"] = dl.groupby("일자")["점수"].rank(ascending=False)
st = dl.groupby("경계코드").agg(일별점수_중앙값=("점수", "median"), 일별점수_표준편차=("점수", "std"),
                              일별점수_최소=("점수", "min"), 일별점수_최대=("점수", "max"),
                              일별_1위일수=("일별순위", lambda x: int((x == 1).sum())),
                              일별_양수일수=("점수", lambda x: int((x > 0).sum()))).reset_index()
d = d.merge(st, on="경계코드")

v3 = pd.read_csv("data/verified/접근성_지수_v3_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
d = d.merge(v3[["경계코드", "지수_v3", "하위%_v3", "순위_v3", "분류_v3", "고령1천명당_정류소"]], on="경계코드")
d["접근성_금천내순위(1=최악)"] = d.지수_v3.rank().astype(int)

d["점수_금천내상위50%"] = np.where(d.우선순위_순위 <= TOP_N, "Y", "")
d["v3_서울하위25%"] = np.where(d.분류_v3 == "하위25%", "Y", "")
d["선정여부"] = np.where((d["점수_금천내상위50%"] == "Y") & (d["v3_서울하위25%"] == "Y"), "선정", "")
d["표본플래그"] = np.where(d["70세이상_16-21시_합계"] < cut, "표본 상대적으로 적음", "")
d["플래그설명"] = ""
all_or = lambda k, n: f"{n}일 모두" if k == n else f"{n}일 중 {k}일"
for i, r in d[d.표본플래그 != ""].iterrows():
    x = dl[dl.경계코드 == r.경계코드].sort_values("점수").점수
    d.loc[i, "플래그설명"] = (
        f"70세이상 16~21시 31일 합 {r['70세이상_16-21시_합계']:,.0f} < 서울 70+ 하위10% 컷 {cut:,.0f}(차이 {round(cut) - round(r['70세이상_16-21시_합계']):,}). "
        f"일별 점수 {all_or(r.일별_양수일수, len(x))} 양수(최소 {x.min():+.1f}, 중앙값 {x.median():+.1f}, 사분위 {x.quantile(.25):+.1f}~{x.quantile(.75):+.1f}, "
        f"표준편차 {x.std():.1f}로 타 동 {st[st.경계코드 != r.경계코드].일별점수_표준편차.min():.1f}~{st[st.경계코드 != r.경계코드].일별점수_표준편차.max():.1f}과 비슷), "
        f"{len(x)}일 중 {r.일별_1위일수}일 금천 내 1위·{all_or(int((dl[dl.경계코드 == r.경계코드].일별순위 <= TOP_N).sum()), len(x))} 상위 {TOP_N}위 이내, "
        f"상위 3일 제외 평균 {x.iloc[:-3].mean():+.1f} → 특정일에 튄 값 아님, 안정적으로 높음")
d["각주"] = ("고령층=70세 이상. v3 접근성지수의 '고령 1천명당 정류소' 요소는 65세 이상 인구 기준이라 70세 정의와 완전히 일치하지 않음. "
            "서울 9곳(60세 이상 기준)과 직접 순위비교 불가")

# 소득 필터 생략 검증: 필터를 먼저/나중에 적용해도 선정 동이 같은지
inc = pd.read_csv("data/verified/소득_가구수_행정동_정리.csv", encoding="utf-8-sig")
inc = inc[inc.자치구 == GU].set_index("행정동명").소득분위
d["_소득"] = d.동이름.map(inc)
sel = set(d.loc[d.선정여부 == "선정", "동이름"])
low = d[d._소득 <= INCOME_LOW_MAX].sort_values("우선순위점수", ascending=False)
pre = low.head(len(low) // 2)
post = d[(d.선정여부 == "선정") & (d._소득 <= INCOME_LOW_MAX)]
same = sel == set(pre.loc[pre["v3_서울하위25%"] == "Y", "동이름"]) == set(post.동이름)
print(f"[소득 검증] 금천 소득분위 {int(inc.min())}~{int(inc.max())}, 선정 동 소득분위 {sorted(d.loc[d.선정여부 == '선정', '_소득'].astype(int).unique())}, "
      f"필터 적용 여부와 무관하게 선정 동 동일: {same}")
d = d.drop(columns="_소득")

pct = ["차량비율", "버스비율", "지하철비율", "도보비율", "기타비율(1·2·3·9)", "대중교통비율", "우선순위점수"]
out = d.copy()
for c in pct:
    out[c] = (out[c] * 100).round(2)
for c in ["일별점수_중앙값", "일별점수_표준편차", "일별점수_최소", "일별점수_최대"]:
    out[c] = out[c].round(2)
out = out.rename(columns={c: f"{c}(%)" for c in pct[:-1]} | {"우선순위점수": "우선순위점수(%p)",
                 "순위_v3": "순위_v3(서울387, 1=최악)", "하위%_v3": "v3_서울하위%"})
cols = ["우선순위_순위", "경계코드", "동이름", "70세이상_16-21시_합계", "차량비율(%)", "버스비율(%)", "지하철비율(%)", "도보비율(%)",
        "기타비율(1·2·3·9)(%)", "대중교통비율(%)", "우선순위점수(%p)", "일별점수_중앙값", "일별점수_표준편차", "일별점수_최소",
        "일별점수_최대", "일별_1위일수", "일별_양수일수", "지수_v3", "v3_서울하위%", "순위_v3(서울387, 1=최악)", "분류_v3",
        "접근성_금천내순위(1=최악)", "고령1천명당_정류소", "점수_금천내상위50%", "v3_서울하위25%", "선정여부", "표본플래그",
        "플래그설명", "각주"]
out[cols].sort_values("우선순위_순위").to_csv(f"{OUT}/금천구_핀포인트_70세기준_결과.csv", index=False, encoding="utf-8-sig")
print(out[cols].sort_values("우선순위_순위")[["우선순위_순위", "동이름", "우선순위점수(%p)", "지수_v3", "v3_서울하위%", "선정여부", "표본플래그"]].to_string(index=False))

# ---- 그래프 (선정=blue, 미선정=회색, 텍스트는 잉크색)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SEL, OTH, INK2, MUTED, SURF, GRID = "#2a78d6", "#c3c2b7", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
d = d.sort_values("우선순위_순위")
flag_nm = set(d.loc[d.표본플래그 != "", "동이름"])


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2)


# 막대그래프
fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
ax.bar(d.동이름, d.우선순위점수 * 100, color=[SEL if s == "선정" else OTH for s in d.선정여부], width=0.6, edgecolor=SURF, linewidth=2)
ax.axhline(0, color=INK2, lw=1)
for i, (v, nm) in enumerate(zip(d.우선순위점수 * 100, d.동이름)):
    lab = f"{v:+.1f}" + ("\n표본 적음*" if nm in flag_nm else "")
    ax.text(i, v + (0.4 if v >= 0 else -0.4), lab, ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
ax.set_ylim(-7.5, 20)
ax.axvline(TOP_N - 0.5, color=MUTED, lw=1, ls="--")
ax.text(TOP_N - 0.4, 19.5, "← 금천 내 점수 상위 50%", fontsize=8.5, color=MUTED, va="top")
ax.set_ylabel("우선순위점수 (%p) = 차량비율 - (버스+지하철)비율", color=INK2)
ax.set_title("금천구 10개 동 우선순위점수 (70세 이상, 16~21시 도착, 2026년 8월)", fontsize=12, loc="left")
ax.legend(handles=[Patch(color=SEL, label="선정 4곳 (점수 상위 50% × v3 서울 하위 25%)"), Patch(color=OTH, label="미선정")],
          frameon=False, fontsize=9, loc="lower left")
style(ax)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.text(0.01, 0.01, f"* 시흥3동: 서울 70+ 하위10% 컷({cut:,.0f})에 {round(cut) - round(d.loc[d.표본플래그 != '', '70세이상_16-21시_합계'].iloc[0]):,} 미달. 31일 일별 점수 모두 양수, 25일 1위로 안정적.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/금천구_우선순위점수_막대.png", facecolor=SURF)
plt.close(fig)

# 산점도
v25 = v3.loc[v3.분류_v3 == "하위25%", "지수_v3"].max()
ymid = (d.loc[d.우선순위_순위 == TOP_N, "우선순위점수"].iloc[0] + d.loc[d.우선순위_순위 == TOP_N + 1, "우선순위점수"].iloc[0]) / 2 * 100
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
ax.set_ylim(-7.5, 20)
ax.set_xlim(-2, 78)
ax.fill_between([-2, v25], ymid, 20, color="#eef4fc", zorder=0)
ax.axvline(v25, color=MUTED, lw=1, ls="--")
ax.axhline(ymid, color=MUTED, lw=1, ls="--")
for _, r in d.iterrows():
    x, y = r.지수_v3, r.우선순위점수 * 100
    sel_, flag = r.선정여부 == "선정", r.동이름 in flag_nm
    ax.scatter(x, y, s=150 if flag else 90, marker="D" if flag else "o", color=SEL if sel_ else OTH, edgecolor=SURF, linewidth=2, zorder=3)
    if flag:
        ax.scatter(x, y, s=320, marker="D", facecolor="none", edgecolor=INK2, linewidth=1.2, zorder=2)
    ax.annotate(r.동이름 + (" (표본 적음)" if flag else ""), (x, y), xytext=(8, -13) if r.동이름 == "시흥1동" else (12, 4),
                textcoords="offset points", fontsize=9, color=INK2 if sel_ else MUTED)
ax.set_xlabel("접근성지수 v3 (높을수록 좋음)", color=INK2)
ax.set_ylabel("우선순위점수 (%p, 높을수록 차량의존)", color=INK2)
ax.set_title("금천구 차량의존 × 대중교통 접근성 (70세 이상)", fontsize=12, loc="left")
yl = ax.get_ylim()
ax.text(v25 + 0.8, yl[0] + 0.3, f"서울 v3 하위 25% 경계 ({v25:.1f})", ha="left", va="bottom", fontsize=8.5, color=MUTED)
ax.text(ax.get_xlim()[1], ymid + 0.3, "금천 내 점수 상위 50% 경계", ha="right", va="bottom", fontsize=8.5, color=MUTED)
ax.text(4, 19.2, "선정 영역 (점수 상위 50% × v3 하위 25%)", va="top", ha="left", fontsize=9, color="#1c5cab")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SEL, ms=9, label="선정 4곳"),
                   Line2D([], [], marker="o", ls="", color=OTH, ms=9, label="미선정"),
                   Line2D([], [], marker="D", ls="", markerfacecolor=SEL, markeredgecolor=INK2, ms=9, label="시흥3동 (표본 상대적으로 적음)")],
          frameon=False, fontsize=9, loc="upper right")
style(ax)
fig.text(0.01, 0.01, "주: v3의 '고령 1천명당 정류소' 요소는 65세 이상 인구 기준으로, 70세 이상 정의와 완전히 일치하지 않음.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/금천구_차량의존x접근성_산점도.png", facecolor=SURF)
plt.close(fig)
print("저장 완료:", OUT)
