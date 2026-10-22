"""면허 자진반납자 기준 DRT 잠재 수요 추정 (시흥3·4·5동, 70세 이상 반납자).

실행: 프로젝트 루트에서 `python scripts/금천구_반납자수요.py`  (다음: 금천구_DRT_v4_보정.py가 조합별 필요 차량 수 계산)
- 반납자: 금천구 70세 이상 자진반납(70-74 + 75-79 + 80+), 2022~2025 4년 누적. 원자료 data/raw/self_surrender_long.csv
  (기존 프로젝트 outputs에서 복사, 자진반납 PDF 파싱). 1단계 자치구 반납률(65세 이상, data/verified/)은 그대로 둠
  - 65-69세는 2025년 금천 반납의 약 4%(37명/859명)라 정책 대상을 70세 이상으로 정함(methodology 10번 고령층 정의와 같은 기준)
- 세 동 몫: 주민등록 2026.9 기준 세 동 70세 이상 8,817명 ÷ 금천구 70세 이상 33,014명 = 26.7%
  (data/raw/202609_202609_연령별인구현황_월간.xlsx 금천구 '연령구간인구수'. 이전 SGIS 2024 면적배분 28.1%는 쓰지 않음)
- 조합: 잔존율(생존·거주) 100/80% x 이용 참여율 30/50/100% x 주 1/2회 x 평일 5일/주 7일 운행
  하루 이용자 = 누적 반납자 x 세 동 몫 x 잔존율 x 참여율 x 주당 이용 횟수 / 운행일수 (1명 = 하루 1회 왕복)
- 참여율 100%는 상한에 가까움. 반대로 반납하지 않은(또는 원래 운전하지 않던) 고령자도 이용할 수 있어 실제 수요는 이 추정 밖에서도 생김
"""
import pandas as pd

OUT = "outputs"
SRC = "data/raw/self_surrender_long.csv"
POP3, POPGU = 8817, 33014  # 주민등록 2026.9 70세 이상: 시흥3·4·5동 합 / 금천구
AGES70 = ["70-74", "75-79", "80+"]
LIMIT = {"DRT 1대": 26, "DRT 2대(새 권역)": 62}  # 한계 이용자: 평균 대기 15분·거절률 5% 이하를 지키는 마지막 계산 지점(v4)

l = pd.read_csv(SRC, encoding="utf-8-sig")
g = l[l.구 == "금천"].pivot_table(index="연도", columns="연령구간", values="자진반납건수", aggfunc="sum")
by70 = g[AGES70].sum(axis=1)
cum = int(by70.loc[2022:2025].sum())
share = POP3 / POPGU
pot = cum * share
assert by70.loc[2025] == 822, by70.loc[2025]

rows = []
for surv in (1.0, 0.8):
    for part in (0.3, 0.5, 1.0):
        for f in (1, 2):
            for days, lab in ((5, "평일 5일"), (7, "주 7일")):
                n = pot * surv * part * f / days
                # 계산식이 재현되도록 모든 인수를 열로 남김: 하루 이용자 = 누적 x 비중 x 잔존 x 참여 x 주당 왕복 / 운행일
                rows.append({"잔존율_%": int(surv * 100), "참여율_%": int(part * 100), "이용빈도": f"주 {f}회", "운행일": lab,
                             "누적반납자_명(2022~2025, 70세 이상)": cum, "세 동 비중": round(share, 4), "잔존율": surv, "참여율": part,
                             "주당 왕복_회": f, "운행일_일/주": days, "잠재 이용자_명": round(pot * surv * part),
                             "하루 이용자(소수)": round(n, 2), "하루 이용자_명": round(n), "하루 편도 호출_건": 2 * round(n),
                             "1대 한계(26명) 대비": f"{n / LIMIT['DRT 1대']:.1f}배", "2대 한계(62명) 대비": f"{n / LIMIT['DRT 2대(새 권역)']:.1f}배",
                             "이용률_%(세 동 70세 이상 8,817명 대비)": round(n / POP3 * 100, 2)})
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/DRT_반납자기준_수요추정.csv", index=False, encoding="utf-8-sig")
yr = pd.DataFrame({"65-69": g["65-69"], "70세 이상": by70, "65세 이상 계": g.sum(axis=1)})
yr["65-69 비중_%"] = (yr["65-69"] / yr["65세 이상 계"] * 100).round(1)
yr.to_csv(f"{OUT}/DRT_반납자_연도별_연령구간.csv", encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("금천구 자진반납 연도별:\n", yr.to_string())
print(f"70세 이상 2022~2025 누적 {cum}명 x 세 동 몫 {share:.3f} ({POP3:,}/{POPGU:,}) -> 잠재 이용자(잔존·참여 100%) {pot:.0f}명")
print(t.to_string(index=False))
