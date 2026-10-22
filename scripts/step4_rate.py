"""Step 4: 반납률 = 자진반납건수(PDF) / 총면허소지자수_B(CSV, 2025.12.31 단일 시점).

주의: 분모는 2025.12.31 스냅샷 하나뿐이라 전 연도에 동일 적용.
  -> 2025년 반납률 = 정확한 값, 2022~2024년 반납률 = 그 해 당시 소지자수가 아닌 2025년 말 소지자수로 근사한 참고용 값.
추가 점검: 반납률 > 100% (분자>분모) 인 경우가 있으면 원자면허 반납자 혼입 등 분모 정의 문제 가능성을 먼저 확인한다.
출력: outputs/surrender_rate_by_gu_year.csv (연도x구, overall 65+ rate + 연령대별)
      outputs/count_vs_rate_rank_2025.csv (건수순위 vs 반납률순위 비교)
      outputs/금천구_점검.csv
      outputs/fig_2025_rate_bar.png, outputs/fig_trend_line.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = "data/interim/재현확인"

den = pd.read_csv(f"{OUT}/license_holders_65plus_by_gu_agegroup.csv", encoding="utf-8-sig")
num = pd.read_csv(f"{OUT}/self_surrender_long.csv", encoding="utf-8-sig")
assert set(den["구"]) == set(num["구"]) and set(den["연령구간"]) == set(num["연령구간"])

m = num.merge(den[["구", "연령구간", "총면허소지자수_B"]], on=["구", "연령구간"], how="left")
assert m["총면허소지자수_B"].notna().all()
m["반납률(%)"] = (m["자진반납건수"] / m["총면허소지자수_B"] * 100).round(3)
m["주석"] = m["연도"].map(lambda y: "정확(2025.12.31 기준)" if y == 2025 else "참고용 근사(분모=2025년말 소지자수)")
over100 = m[m["반납률(%)"] > 100]
print(f"반납률 > 100%인 (연도,구,연령구간) 조합: {len(over100)}건")
if len(over100):
    print(over100[["연도", "구", "연령구간", "자진반납건수", "총면허소지자수_B", "반납률(%)"]].to_string(index=False))

# ---------- 구x연도 종합(65+ 전체) 반납률
agg = m.groupby(["연도", "구"])[["자진반납건수", "총면허소지자수_B"]].sum().reset_index()
agg["반납률(%)"] = (agg["자진반납건수"] / agg["총면허소지자수_B"] * 100).round(3)
agg["주석"] = agg["연도"].map(lambda y: "정확(2025.12.31 기준)" if y == 2025 else "참고용 근사(분모=2025년말 소지자수)")
wide_rate = agg.pivot(index="구", columns="연도", values="반납률(%)")
wide_rate.to_csv(f"{OUT}/surrender_rate_by_gu_year.csv", encoding="utf-8-sig")
m.to_csv(f"{OUT}/surrender_rate_by_gu_year_agegroup.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("\n== 구x연도 반납률(%) (65+ 전체, 소계)")
print(wide_rate.round(2).to_string())

# ---------- Step5 1: 상/하위 5개 (2025 기준)
d25 = agg[agg["연도"] == 2025].sort_values("반납률(%)", ascending=False).reset_index(drop=True)
d25["반납률_순위"] = d25.index + 1
print(f"\n2025년 반납률 평균 {d25['반납률(%)'].mean():.3f}% / 중위값 {d25['반납률(%)'].median():.3f}%")
print("\n상위 5:"); print(d25.head(5)[["구", "반납률(%)", "반납률_순위"]].to_string(index=False))
print("하위 5:"); print(d25.tail(5)[["구", "반납률(%)", "반납률_순위"]].to_string(index=False))

# ---------- Step5 2: 건수 순위 vs 반납률 순위 (2025)
cnt25 = d25.copy()
cnt25["건수_순위"] = cnt25["자진반납건수"].rank(ascending=False, method="min").astype(int)
cnt25["순위변동(건수-반납률)"] = cnt25["건수_순위"] - cnt25["반납률_순위"]
cnt25 = cnt25.sort_values("순위변동(건수-반납률)", key=lambda s: s.abs(), ascending=False)
cols = ["구", "자진반납건수", "건수_순위", "반납률(%)", "반납률_순위", "순위변동(건수-반납률)"]
cnt25[cols].to_csv(f"{OUT}/count_vs_rate_rank_2025.csv", index=False, encoding="utf-8-sig")
print("\n== 건수 순위 vs 반납률 순위 (2025, 변동 큰 순)")
print(cnt25[cols].to_string(index=False))

# ---------- Step5 3: 금천구 점검
g = "금천"
rank_row = d25[d25["구"] == g].iloc[0]
print(f"\n== 금천구 점검: 2025 반납률 {rank_row['반납률(%)']:.3f}%, 25개 구 중 {int(rank_row['반납률_순위'])}위"
      f" (평균 {d25['반납률(%)'].mean():.3f}%, 중위값 {d25['반납률(%)'].median():.3f}% 대비 "
      f"{'위' if rank_row['반납률(%)'] >= d25['반납률(%)'].median() else '아래'})")
trend = agg[agg["구"] == g].sort_values("연도")[["연도", "자진반납건수", "총면허소지자수_B", "반납률(%)"]]
print(trend.to_string(index=False))
# 다른 구와 비교: 연도별 반납률의 전체 평균/금천구의 전년대비 변화율 vs 서울 평균 변화율
piv = agg.pivot(index="구", columns="연도", values="반납률(%)")
chg = piv.pct_change(axis=1) * 100
print("\n연도별 전체 평균 반납률(%):", piv.mean().round(3).to_dict())
print("금천구 YoY 변화율(%):", chg.loc[g].round(1).to_dict())
print("서울 평균 YoY 변화율(%) (구별 변화율의 평균):", chg.mean().round(1).to_dict())
print("금천구 매년 순위:", {y: int(piv[y].rank(ascending=False)[g]) for y in piv.columns})
chg.loc[[g]].to_csv(f"{OUT}/금천구_점검.csv", encoding="utf-8-sig")

# ---------- 시각화
fig, ax = plt.subplots(figsize=(11, 6))
d25_sorted = d25.sort_values("반납률(%)", ascending=False)
colors = ["#d62728" if g2 == g else "#4472C4" for g2 in d25_sorted["구"]]
ax.bar(d25_sorted["구"], d25_sorted["반납률(%)"], color=colors)
ax.axhline(d25["반납률(%)"].mean(), color="gray", ls="--", lw=1, label=f"평균 {d25['반납률(%)'].mean():.2f}%")
ax.set_title("2025년 자치구별 고령층(65+) 운전면허 자진반납률")
ax.set_ylabel("반납률(%)")
ax.set_xticklabels(d25_sorted["구"], rotation=60, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig_2025_rate_bar.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(10, 6))
for g2 in piv.index:
    ax2.plot(piv.columns, piv.loc[g2], color="#cccccc", lw=1, alpha=0.7)
ax2.plot(piv.columns, piv.loc[g], color="#d62728", lw=2.5, marker="o", label="금천구")
ax2.plot(piv.columns, piv.mean(), color="#1f1f1f", lw=2, ls="--", marker="s", label="25개구 평균")
ax2.set_title("연도별 반납률 추세 (2022~2024 참고용 근사, 2025만 정확)")
ax2.set_ylabel("반납률(%)")
ax2.set_xticks(list(piv.columns))
ax2.legend()
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig_trend_line.png", dpi=150)
print("\n그래프 저장: fig_2025_rate_bar.png, fig_trend_line.png")
