"""서울시 소득_가구수.xlsx 정리 -> 우선순위 결과에 소득분위/가구수 병합 -> 상관 계산.

xlsx 구조: 머리글 5행 뒤에 [이름+소득분위+가구수] 행과 [소득구간 텍스트] 행이 번갈아 나온다.
3개 분기(2024/2025/2026 2분기) 값이 동일해서 2026년 2분기(col 5,6)만 사용한다.
자치구 행 아래에 그 구의 동 행이 이어지므로 순서로 자치구를 붙인다(신사동처럼 구를 넘는 동명 중복 때문).

출력
  - 소득_가구수_행정동_정리.csv        (행정동명, 자치구, 소득분위, 가구수)
  - 소득_가구수_자치구_서울시.csv      (구분, 이름, 소득분위, 가구수)  <- 행정동 매칭에는 쓰지 않음
  - 귀가시간대_우선순위_교차검증.csv   (소득분위, 가구수, 소득구분, 유형 컬럼을 추가해 다시 저장)
"""
import csv
import glob
import re

import pandas as pd

MAIN = "data/interim/재현확인/귀가시간대_우선순위_교차검증.csv"
LOW_MAX, HIGH_MIN = 6, 8      # 소득 낮음 <=6분위 < 중간(7, 서울 전체와 동일) < 소득 높음 >=8분위
SMALL_HH = 500                # 가구수가 이보다 적으면 소득분위 신뢰 낮음 표시
TOP = 30
# 후보/순위에서 '제외' 표시만 하는 동 (삭제하지 않고 이유를 남긴다)
EXCLUDE = {("강동구", "둔촌1동"): "가구수 11로 통계적 신뢰 불가(재건축 이주 추정)"}
VULN_OUT = "data/interim/재현확인/취약지대_후보_9곳.csv"

# ---------- 1) xlsx 정리
p = glob.glob("data/raw/서울시 소득_가구수*.xlsx")[0]
raw = pd.read_excel(p, header=None, dtype=object).iloc[5:].reset_index(drop=True)
rows = raw.iloc[0::2].reset_index(drop=True)
gu_names = {r["sigungu"] for r in csv.DictReader(open("data/interim/재현확인/행정동_코드매핑_생활이동_경계.csv", encoding="utf-8-sig"))}

dong, upper, cur_gu = [], [], None
for _, r in rows.iterrows():
    name = str(r[0]).strip()
    q = int(re.search(r"\d+", str(r[5])).group())
    hh = int(float(r[6]))
    if name == "서울시":
        upper.append(("서울시", name, q, hh))
    elif name in gu_names:
        cur_gu = name
        upper.append(("자치구", name, q, hh))
    else:
        dong.append({"행정동명": name, "자치구": cur_gu, "소득분위": q, "가구수": hh})

inc = pd.DataFrame(dong)
inc.to_csv("data/interim/재현확인/소득_가구수_행정동_정리.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(upper, columns=["구분", "이름", "소득분위", "가구수"]).to_csv(
    "data/interim/재현확인/소득_가구수_자치구_서울시.csv", index=False, encoding="utf-8-sig")
print(f"정리: 행정동 {len(inc)}개, 자치구 {sum(1 for u in upper if u[0] == '자치구')}개, 서울시 1행")

# ---------- 2) 병합 (자치구 + 정규화한 동명)
norm = lambda s: re.sub(r"[·ㆍ.\s]", "", str(s))
inc["_k"] = inc["자치구"] + "|" + inc["행정동명"].map(norm)
main = pd.read_csv(MAIN, encoding="utf-8-sig", dtype={"경계코드": str})
main = main.drop(columns=[c for c in ("소득분위", "가구수", "소득구분", "유형") if c in main.columns])
main["_k"] = main["자치구"] + "|" + main["동이름"].map(norm)
main = main.merge(inc[["_k", "소득분위", "가구수"]], on="_k", how="left")

unmatched = main[main["소득분위"].isna()]
print(f"\n매칭: {main['소득분위'].notna().sum()}/{len(main)}   실패 {len(unmatched)}개:",
      [f"{r.자치구} {r.동이름}" for r in unmatched.itertuples()])
matched_keys = set(main["_k"])
notin = inc[~inc["_k"].isin(matched_keys)]
print("xlsx에만 있는 동(우선순위 결과에 없음):", len(notin), "개")

def band(q):
    return "낮음" if q <= LOW_MAX else ("높음" if q >= HIGH_MIN else "중간")
main["소득구분"] = main["소득분위"].map(lambda q: band(q) if pd.notna(q) else "")
top_cut = main["우선순위_순위"] <= TOP
main["유형"] = ""
main.loc[top_cut & (main["소득구분"] == "낮음"), "유형"] = "취약지대 후보(소득낮음+우선순위높음)"
main.loc[top_cut & (main["소득구분"] == "높음"), "유형"] = "부촌 차량의존(소득높음+우선순위높음)"
main.loc[top_cut & (main["소득구분"] == "중간"), "유형"] = "중간소득(서울평균 7분위)"
main.loc[top_cut & main["소득분위"].isna(), "유형"] = "소득 미매칭"

main["제외"] = ""
main["제외사유"] = ""
for (gu, nm), why in EXCLUDE.items():
    hit = (main["자치구"] == gu) & (main["동이름"] == nm)
    assert hit.sum() == 1, f"제외 대상 동을 찾지 못함: {gu} {nm}"
    main.loc[hit, ["제외", "유형", "제외사유"]] = ["제외", "제외", why]
main.drop(columns="_k").to_csv(MAIN, index=False, encoding="utf-8-sig")

# 취약지대 후보: 상위 30 중 소득 낮음(<=6분위)이면서 제외 표시가 없는 동
vuln = main[top_cut & (main["소득구분"] == "낮음") & (main["제외"] == "")]
vuln_cols = ["우선순위_순위", "동이름", "자치구", "소득분위", "가구수", "우선순위점수", "차량비율", "버스비율",
             "지하철비율", "도보비율", "대중교통비율(버스+지하철)", "귀가시간대고령층합계", "귀가비중",
             "귀가비중_낮음", "경계코드"]
vuln[vuln_cols].to_csv(VULN_OUT, index=False, encoding="utf-8-sig")
print(f"\n취약지대 후보 {len(vuln)}곳 저장 -> {VULN_OUT}:", ", ".join(vuln["동이름"]))

# ---------- 3) 상관
m = main.dropna(subset=["소득분위"])
pear = m["우선순위점수"].corr(m["소득분위"])
spear = m["우선순위점수"].rank().corr(m["소득분위"].rank())  # scipy 없이 순위 상관
print(f"\n피어슨 r(우선순위점수, 소득분위) = {pear:+.3f}  (n={len(m)})   [참고 스피어만 {spear:+.3f}]")
big = m[m["가구수"] >= SMALL_HH]
print(f"가구수 {SMALL_HH} 이상 동만: 피어슨 {big['우선순위점수'].corr(big['소득분위']):+.3f} (n={len(big)})")
print("소득분위별 우선순위점수 평균:", {int(k): round(v, 3) for k, v in m.groupby("소득분위")["우선순위점수"].mean().items()},
      " 동수:", m["소득분위"].astype(int).value_counts().sort_index().to_dict())

print(f"\n== 우선순위 상위 {TOP} + 소득 (낮음<= {LOW_MAX}분위 / 높음>= {HIGH_MIN}분위)")
for r in main[top_cut].itertuples():
    flag = " 귀가낮음" if getattr(r, "귀가비중_낮음") == "낮음" else ""
    small = " 가구적음" if pd.notna(r.가구수) and r.가구수 < SMALL_HH else ""
    inc_txt = (f"소득{int(r.소득분위)}분위({r.소득구분}) 가구{int(r.가구수):>6,}"
               if pd.notna(r.소득분위) else "소득 미매칭")
    print(f"{r.우선순위_순위:>2} {r.자치구} {r.동이름:<9} 점수{r.우선순위점수:+.3f} {inc_txt} "
          f"귀가비중{r.귀가비중:.0%}{flag}{small}")
for t in ("취약지대 후보", "부촌 차량의존", "중간소득", "소득 미매칭"):
    g = main[top_cut & main["유형"].str.startswith(t)]
    print(f"[{t}] {len(g)}개:", ", ".join(g["동이름"]))
print("\nxlsx에만 있는 동 예시(앞 12):", [f"{r.자치구} {r.행정동명}" for r in notin.head(12).itertuples()])
