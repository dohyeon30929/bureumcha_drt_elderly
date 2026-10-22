"""Step 2: 운전면허소지자 CSV(1세 단위)를 65세 이상 5세 구간(65-69/70-74/75-79/80+)으로 재구성.

분모 정의(후보 B, 확정): 총면허소지자수 = 1종소계 + 2종소계 - 2종 원자
  (2종 원자=원동기장치자전거免許만으로는 자동차를 몰 수 없어 "차량의존" 모수에서 제외)
빠진 (구,나이) 조합은 해당 값이 전부 0이라 원본에서 생략된 것으로 확인하였다(Step 1). groupby sum에서 자연히 0으로 처리된다.

추가 확인: 65+ 구간에서 구별 2종원자 비중(= 2종원자 / (1종소계+2종소계))을 비교해
분모 정의(B)의 영향이 구별로 얼마나 다른지 점검.

출력: outputs/license_holders_65plus_by_gu_agegroup.csv (long), outputs/license_65plus_wide.csv (wide, 분모 B)
      outputs/원자면허_비중_점검.csv
"""
import pandas as pd

IN = "data/raw/경찰청_운전면허소지자현황_서울_시군구_연령별_대장별_20251231.csv"
OUT_DIR = "data/interim/재현확인"
AGE_BANDS = [(65, 69, "65-69"), (70, 74, "70-74"), (75, 79, "75-79"), (80, 999, "80+")]

df = pd.read_csv(IN, encoding="utf-8-sig")
df["나이"] = df["나이"].astype(int)
for c in ("1종소계", "2종소계", "2종 원자"):
    df[c] = pd.to_numeric(df[c])
df["구"] = df["시군구"].str.replace("서울 ", "", regex=False)

def band(age):
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return None

df["연령구간"] = df["나이"].map(band)
d65 = df[df["연령구간"].notna()].copy()
assert set(d65["구"]) == set(df["구"])  # 25개 구 모두 65+ 데이터 보유 확인

g = d65.groupby(["구", "연령구간"])[["1종소계", "2종소계", "2종 원자"]].sum().reset_index()
g["총면허소지자수_B"] = g["1종소계"] + g["2종소계"] - g["2종 원자"]
g["총면허소지자수_A(참고)"] = g["1종소계"] + g["2종소계"]
g["연령구간"] = pd.Categorical(g["연령구간"], categories=[b[2] for b in AGE_BANDS], ordered=True)
g = g.sort_values(["구", "연령구간"])
g.to_csv(f"{OUT_DIR}/license_holders_65plus_by_gu_agegroup.csv", index=False, encoding="utf-8-sig")

wide = g.pivot(index="구", columns="연령구간", values="총면허소지자수_B")
wide["합계(65+)"] = wide.sum(axis=1)
wide.to_csv(f"{OUT_DIR}/license_65plus_wide.csv", encoding="utf-8-sig")

# ---------- 원자면허 비중 점검 (분모 정의 B의 영향 크기 확인)
chk = g.groupby("구").agg(**{
    "1종소계": ("1종소계", "sum"), "2종소계": ("2종소계", "sum"), "2종원자": ("2종 원자", "sum"),
    "총면허_A": ("총면허소지자수_A(참고)", "sum"), "총면허_B": ("총면허소지자수_B", "sum")}).reset_index()
chk["원자비중(%)"] = (chk["2종원자"] / chk["총면허_A"] * 100).round(3)
chk["A대비B_감소율(%)"] = ((chk["총면허_A"] - chk["총면허_B"]) / chk["총면허_A"] * 100).round(3)
chk = chk.sort_values("원자비중(%)", ascending=False)
chk.to_csv(f"{OUT_DIR}/원자면허_비중_점검.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("65+ 전체 wide (총면허소지자수_B):")
print(wide.round(0).to_string())
print(f"\n서울 전체 65+ : 1종 {g['1종소계'].sum():,} / 2종 {g['2종소계'].sum():,} / 2종원자 {g['2종 원자'].sum():,} "
      f"/ 총면허_A {g['총면허소지자수_A(참고)'].sum():,} / 총면허_B {g['총면허소지자수_B'].sum():,} "
      f"(B가 A보다 {g['2종 원자'].sum():,}명 적음, {g['2종 원자'].sum()/g['총면허소지자수_A(참고)'].sum()*100:.2f}%)")
print(f"\n== 65+ 원자면허 비중 구별 비교 (상위 5 / 하위 5, 서울 전체 평균 {chk['원자비중(%)'].mean():.3f}%)")
print(chk.head(5)[["구", "원자비중(%)", "A대비B_감소율(%)", "총면허_A", "총면허_B"]].to_string(index=False))
print("...")
print(chk.tail(5)[["구", "원자비중(%)", "A대비B_감소율(%)", "총면허_A", "총면허_B"]].to_string(index=False))
print(f"\n구간 범위: 최대 {chk['원자비중(%)'].max():.3f}% ~ 최소 {chk['원자비중(%)'].min():.3f}% "
      f"(편차 {chk['원자비중(%)'].max() - chk['원자비중(%)'].min():.3f}%p)")
