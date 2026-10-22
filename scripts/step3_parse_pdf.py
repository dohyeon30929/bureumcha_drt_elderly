"""Step 3: 자진반납 PDF 4개(22~25년) 파싱 -> 자치구 x 연령구간 x 연도 long 테이블.

구조: 각 PDF는 1페이지, 표 1개(27행=제목행+헤더행+25개구, 6열=구분+합계+65-69/70-74/75-79/80이상).
종별(1종/2종) 구분 없음 -> 자진반납 건수는 면허 종류와 무관한 '사람 수' 합계로 추정(확정은 아님).
검증: 2025년 전체 합계가 33,084와 일치하는지 확인한다.
출력: outputs/self_surrender_long.csv, outputs/self_surrender_wide.csv
"""
import re

import pandas as pd
import pdfplumber

FILES = {2022: "data/raw/22년 자진반납.pdf", 2023: "data/raw/23년 자진반납.pdf", 2024: "data/raw/24년 자진반납.pdf", 2025: "data/raw/25년 자진반납.pdf"}
EXPECTED_2025_TOTAL = 33084
OUT_DIR = "data/interim/재현확인"

rows = []
for year, path in FILES.items():
    with pdfplumber.open(path) as pdf:
        table = pdf.pages[0].extract_tables()[0]
    header = table[1]
    assert header == ["연령대\n자치구별", "합계", "65-69", "70-74", "75-79", "80이상"], (year, header)
    for r in table[2:]:
        gu_raw = r[0]
        if gu_raw is None:
            continue
        gu = re.sub(r"^서울\s*", "", gu_raw.strip())
        vals = [int(x.replace(",", "")) for x in r[1:]]
        rows.append({"연도": year, "구": gu, "합계": vals[0], "65-69": vals[1], "70-74": vals[2],
                     "75-79": vals[3], "80+": vals[4]})

df = pd.DataFrame(rows)
assert df.groupby("연도").size().eq(25).all(), "일부 연도에 25개 구가 안 모임"
# 합계 열 vs 연령대 4칸 합 일치 검증
chk = (df["65-69"] + df["70-74"] + df["75-79"] + df["80+"] - df["합계"]).abs()
print("합계열 vs 연령대 4칸 합 불일치 행:", int((chk != 0).sum()))

by_year = df.groupby("연도")["합계"].sum()
print("\n연도별 서울 전체 합계:")
print(by_year.to_string())
y25 = by_year.loc[2025]
print(f"\n2025년 합계 {y25:,} vs 기대값 {EXPECTED_2025_TOTAL:,}  "
      f"{'일치' if y25 == EXPECTED_2025_TOTAL else f'불일치(차이 {y25 - EXPECTED_2025_TOTAL:+,})'}")

long_ = df.melt(id_vars=["연도", "구"], value_vars=["65-69", "70-74", "75-79", "80+"],
                var_name="연령구간", value_name="자진반납건수")
long_.to_csv(f"{OUT_DIR}/self_surrender_long.csv", index=False, encoding="utf-8-sig")
wide = df.set_index(["연도", "구"]).sort_index()
wide.to_csv(f"{OUT_DIR}/self_surrender_wide.csv", encoding="utf-8-sig")

pd.set_option("display.width", 250)
print("\n2025년 자치구별 합계 (상위 5 / 하위 5):")
d25 = df[df["연도"] == 2025].sort_values("합계", ascending=False)
print(pd.concat([d25.head(5), d25.tail(5)])[["구", "합계"]].to_string(index=False))
