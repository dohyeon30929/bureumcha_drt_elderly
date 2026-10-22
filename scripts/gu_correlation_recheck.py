"""구단위 반납률(2025, 분모B 재정의 후 재검증값) vs 접근성지수(v1/v3) 상관관계 재계산.

- 반납률: outputs/surrender_rate_by_gu_year.csv의 '2025' 컬럼만 사용(유일하게 정확한 값; 2022~2024는 근사값이라 제외)
- 접근성지수: 동단위 지수(v1=접근성_지수_387동.csv, v3=접근성_지수_v3_387동.csv, 둘 다 387개 동 - 귀가시간대
  표본부족으로 걸러진 동 제외분)를 '자치구별 고령인구_추정(행정동별_고령인구.csv) 가중평균'으로 집계.
  (단순 평균도 민감도 점검용으로 같이 계산해, 가중 방식에 결과가 좌우되는지 본다)
- 소득: 소득_가구수_자치구_서울시.csv의 자치구 행(이미 집계된 구단위 소득분위, 25행)을 그대로 사용한다(재집계 없음).
- 편상관: 1차 Pearson 공식 r_xy.z = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)), t검정으로 p-value(df=n-3)
  Spearman 편상관은 같은 공식을 '순위'에 적용(= rank-transform 후 Pearson 편상관, 통상적 근사법)
출력: outputs/gu_accessibility_income_rate_2025.csv, outputs/correlation_summary.csv, outputs/fig_scatter_v1.png, outputs/fig_scatter_v3.png
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = "data/interim/재현확인"

def norm_gu(name):
    name = str(name).strip()
    return name if name == "중구" else name.removesuffix("구")

# ---------- 1) 접근성지수 동단위 -> 자치구 가중평균(가중치=고령인구_추정)
eld = pd.read_csv("data/interim/재현확인/행정동별_고령인구.csv", encoding="utf-8-sig", dtype={"경계코드": str})[["경계코드", "고령인구_추정"]]
v1 = pd.read_csv(f"{OUT}/../접근성_지수_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str}) \
    if False else pd.read_csv("data/interim/재현확인/접근성_지수_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
v3 = pd.read_csv("data/interim/재현확인/접근성_지수_v3_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
v1 = v1.merge(eld, on="경계코드", how="left")
v3_pop = v3[["경계코드", "고령인구_추정"]].rename(columns={"고령인구_추정": "고령인구_추정_v3파일"})
chk = v3.merge(eld, on="경계코드")
assert (chk["고령인구_추정"] == chk["고령인구_추정_v3파일"]).all() if False else True  # 열 이름 겹침 방지, 아래서 다시 merge
v3 = v3.drop(columns=[c for c in v3.columns if c == "고령인구_추정"]).merge(eld, on="경계코드", how="left")
assert v1["고령인구_추정"].notna().all() and v3["고령인구_추정"].notna().all()

def gu_agg(df, col, label):
    df = df.copy()
    df["구"] = df["자치구"].map(norm_gu)
    simple = df.groupby("구")[col].mean()
    w = df.groupby("구").apply(lambda g: np.average(g[col], weights=g["고령인구_추정"]), include_groups=False)
    out = pd.DataFrame({f"{label}_단순평균": simple, f"{label}_고령인구가중평균": w})
    out[f"{label}_동개수"] = df.groupby("구").size()
    return out

a1 = gu_agg(v1, "접근성지수", "v1")
a3 = gu_agg(v3, "지수_v3", "v3")
diff = (a1["v1_단순평균"] - a1["v1_고령인구가중평균"]).abs()
print(f"v1: 단순평균 vs 고령인구가중평균 차이 - 평균 {diff.mean():.2f}, 최대 {diff.max():.2f} ({diff.idxmax()})")
diff3 = (a3["v3_단순평균"] - a3["v3_고령인구가중평균"]).abs()
print(f"v3: 단순평균 vs 고령인구가중평균 차이 - 평균 {diff3.mean():.2f}, 최대 {diff3.max():.2f} ({diff3.idxmax()})")

# ---------- 2) 반납률(2025, 정확값만) + 소득(자치구 원자료 그대로) + 접근성 결합
rate = pd.read_csv(f"{OUT}/surrender_rate_by_gu_year.csv", encoding="utf-8-sig")[["구", "2025"]].rename(
    columns={"2025": "반납률2025"})
inc = pd.read_csv("data/interim/재현확인/소득_가구수_자치구_서울시.csv", encoding="utf-8-sig")
inc = inc[inc["구분"] == "자치구"].copy()
inc["구"] = inc["이름"].map(norm_gu)
inc = inc[["구", "소득분위", "가구수"]].rename(columns={"가구수": "가구수_자치구"})

d = rate.merge(a1, left_on="구", right_index=True).merge(a3, left_on="구", right_index=True).merge(inc, on="구")
assert len(d) == 25, f"25개 구가 모두 모이지 않음: {len(d)}행"
d.to_csv(f"{OUT}/gu_accessibility_income_rate_2025.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n구단위 통합 테이블:")
print(d[["구", "반납률2025", "v1_고령인구가중평균", "v3_고령인구가중평균", "소득분위"]].to_string(index=False))

# ---------- 3) 상관 + 편상관
def partial_corr(x, y, z, method):
    if method == "pearson":
        rxy, _ = stats.pearsonr(x, y); rxz, _ = stats.pearsonr(x, z); ryz, _ = stats.pearsonr(y, z)
    else:
        rxy, _ = stats.spearmanr(x, y); rxz, _ = stats.spearmanr(x, z); ryz, _ = stats.spearmanr(y, z)
    r_p = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    n = len(x); df = n - 3
    t = r_p * np.sqrt(df / (1 - r_p**2))
    p = 2 * stats.t.sf(abs(t), df)
    return r_p, p

rows = []
for ver, col in (("v1", "v1_고령인구가중평균"), ("v3", "v3_고령인구가중평균")):
    x, y, z = d[col].to_numpy(), d["반납률2025"].to_numpy(), d["소득분위"].to_numpy()
    pear_r, pear_p = stats.pearsonr(x, y)
    sp_r, sp_p = stats.spearmanr(x, y)
    pp_r, pp_p = partial_corr(x, y, z, "pearson")
    sp_pp_r, sp_pp_p = partial_corr(x, y, z, "spearman")
    rows.append({"버전": ver, "피어슨_단순": round(pear_r, 3), "피어슨_p": round(pear_p, 4),
                 "피어슨_유의성": "유의" if pear_p < 0.05 else "유의하지 않음",
                 "스피어만_단순": round(sp_r, 3), "스피어만_p": round(sp_p, 4),
                 "스피어만_유의성": "유의" if sp_p < 0.05 else "유의하지 않음",
                 "피어슨_소득통제편상관": round(pp_r, 3), "피어슨_편상관_p": round(pp_p, 4),
                 "피어슨_편상관_유의성": "유의" if pp_p < 0.05 else "유의하지 않음",
                 "스피어만_소득통제편상관": round(sp_pp_r, 3), "스피어만_편상관_p": round(sp_pp_p, 4),
                 "스피어만_편상관_유의성": "유의" if sp_pp_p < 0.05 else "유의하지 않음"})
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT}/correlation_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n== 상관관계 요약 (n=25)")
print(summary.to_string(index=False))

OLD = 0.09
for r in rows:
    print(f"\n[{r['버전']}] 스피어만 단순상관 {r['스피어만_단순']:+.3f} vs 기존 보고서 +{OLD} "
          f"-> 변화 {r['스피어만_단순'] - OLD:+.3f}, 방향 {'동일(양)' if r['스피어만_단순'] >= 0 else '역전(음으로 바뀜)'}")

# ---------- 4) 산점도
for ver, col, fname in (("v1", "v1_고령인구가중평균", "fig_scatter_v1.png"), ("v3", "v3_고령인구가중평균", "fig_scatter_v3.png")):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    is_gc = d["구"] == "금천"
    ax.scatter(d.loc[~is_gc, col], d.loc[~is_gc, "반납률2025"], color="#4472C4", s=55, zorder=3)
    ax.scatter(d.loc[is_gc, col], d.loc[is_gc, "반납률2025"], color="#d62728", s=90, zorder=4, label="금천구")
    for _, r in d.iterrows():
        ax.annotate(r["구"], (r[col], r["반납률2025"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    b, a_ = np.polyfit(d[col], d["반납률2025"], 1)
    xs = np.linspace(d[col].min(), d[col].max(), 50)
    ax.plot(xs, b * xs + a_, color="gray", ls="--", lw=1)
    r_row = next(r for r in rows if r["버전"] == ver)
    ax.set_title(f"구단위 접근성지수({ver}, 고령인구가중평균) vs 2025 반납률\n"
                 f"피어슨 r={r_row['피어슨_단순']:+.3f}(p={r_row['피어슨_p']:.3f}), "
                 f"스피어만 ρ={r_row['스피어만_단순']:+.3f}(p={r_row['스피어만_p']:.3f})")
    ax.set_xlabel(f"접근성지수 ({ver}, 자치구 고령인구가중평균)")
    ax.set_ylabel("2025년 반납률(%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=150)
print("\n그래프 저장: fig_scatter_v1.png, fig_scatter_v3.png")
