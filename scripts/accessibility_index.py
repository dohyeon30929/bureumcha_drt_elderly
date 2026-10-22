"""2단계: 387개 동 종합 접근성 지수(A안 = 버스 0.5 : 지하철 0.5) 산출, 3단계: 취약지대 후보 9곳 검증.

지수 = 세부지표의 387개 동 내 백분위(0~100, 높을수록 접근성 좋음)의 가중합
  버스밀도 0.35, 마을버스제외밀도 0.15, 지하철최단거리(가까울수록 좋음) 0.35, 1km내역수 0.15
가중치 민감도(accessibility_sensitivity.py): 실측 이용비율 58:42(B안)과 지수 상관 0.99, 9곳 순위 변동 최대 14위 -> 강건
'하위 %' = 지수 오름차순 백분위(작을수록 접근성이 나쁨). 순위 1 = 접근성이 가장 나쁜 동.
출력: 접근성_지수_387동.csv, 접근성_검증_9곳.csv
"""
import pandas as pd
from math import comb

pd.set_option("display.width", 250)
W = {"버스밀도": 0.35, "마을버스제외밀도": 0.15, "지하철최단거리": 0.35, "1km내역수": 0.15}
LOW_Q = 25  # 하위 25% 이내면 '접근성 하위권'

main = pd.read_csv("data/interim/재현확인/귀가시간대_우선순위_교차검증.csv", encoding="utf-8-sig", dtype={"경계코드": str})
acc = pd.read_csv("data/interim/재현확인/접근성_지표_행정동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
d = main[["경계코드", "동이름", "자치구", "우선순위_순위", "우선순위점수", "유형"]].merge(
    acc.drop(columns=["동이름", "자치구"]), on="경계코드", how="left")
assert len(d) == 387 and d["버스정류소수"].notna().all()
d["마을버스제외밀도"] = (d["버스정류소수_마을버스제외"] / d["면적_km2"]).round(3)

pct = pd.DataFrame({
    "버스밀도": d["버스정류소밀도_개per_km2"].rank(pct=True),
    "마을버스제외밀도": d["마을버스제외밀도"].rank(pct=True),
    "지하철최단거리": (-d["지하철최단거리_m"]).rank(pct=True),
    "1km내역수": d["지하철역수_1km"].rank(pct=True)}) * 100
for k in W:
    d[f"백분위_{k}"] = pct[k].round(1)
d["접근성지수"] = sum(pct[k] * w for k, w in W.items()).round(2)
d["접근성순위"] = d["접근성지수"].rank(method="min").astype(int)
d["접근성_하위%"] = (d["접근성지수"].rank(pct=True) * 100).round(1)

cols = ["경계코드", "동이름", "자치구", "접근성지수", "접근성순위", "접근성_하위%", "버스정류소밀도_개per_km2", "마을버스제외밀도",
        "지하철최단거리_m", "최근접역", "지하철역수_1km"] + [f"백분위_{k}" for k in W]
d.sort_values("접근성지수")[cols].to_csv("data/interim/재현확인/접근성_지수_387동.csv", index=False, encoding="utf-8-sig")
print("== 2) 387개 동 접근성 지수: 최소 %.1f / 중앙 %.1f / 최대 %.1f" % (d["접근성지수"].min(), d["접근성지수"].median(), d["접근성지수"].max()))
print("   접근성 최하위 10개:", [f"{r.자치구} {r.동이름}({r.접근성지수:.0f})" for r in d.nsmallest(10, "접근성지수").itertuples()])

# ---------- 3) 9곳
v = pd.read_csv("data/interim/재현확인/취약지대_후보_9곳.csv", encoding="utf-8-sig", dtype={"경계코드": str})
r = v[["경계코드"]].merge(d, on="경계코드").sort_values("접근성지수")
def judge(x):
    if x <= 10: return "하위 10% (인프라 매우 부족)"
    if x <= LOW_Q: return "하위 25% (인프라 부족)"
    if x <= 50: return "하위 50% (하위권 아님, 중간 이하)"
    return "중간 이상"
r["판정"] = r["접근성_하위%"].map(judge)
r["인프라있는데_안탐_후보"] = (r["접근성_하위%"] > LOW_Q).map({True: "Y", False: ""})
out = r[["우선순위_순위", "자치구", "동이름", "접근성지수", "접근성순위", "접근성_하위%", "판정", "인프라있는데_안탐_후보",
         "버스정류소밀도_개per_km2", "마을버스제외밀도", "지하철최단거리_m", "최근접역", "지하철역수_1km"]]
out.to_csv("data/interim/재현확인/접근성_검증_9곳.csv", index=False, encoding="utf-8-sig")
print("\n== 3) 취약지대 후보 9곳 (접근성 나쁜 순)")
print(out.to_string(index=False))
n_low = int((r["접근성_하위%"] <= LOW_Q).sum())
print(f"\n하위 25% 이내 {n_low}/9곳, 하위 50% 이내 {int((r['접근성_하위%'] <= 50).sum())}/9곳, 9곳 평균 하위% {r['접근성_하위%'].mean():.1f}")
p = sum(comb(9, k) * 0.25**k * 0.75**(9 - k) for k in range(n_low, 10))
print(f"참고: 무작위 9곳 중 {n_low}곳 이상이 하위 25%일 이항확률 {p:.4f} (금천·양천 인접 동은 상관이 있어 실제 우연 확률은 이보다 큼)")

# ---------- 참고: 전체 상관, 유형별 비교
print(f"\n[참고] 387개 동 우선순위점수 vs 접근성지수: 피어슨 {d['우선순위점수'].corr(d['접근성지수']):+.3f}, "
      f"스피어만 {d['우선순위점수'].rank().corr(d['접근성지수'].rank()):+.3f}  (음수 = 점수 높을수록 접근성 나쁨)")
t = d[d["우선순위_순위"] <= 30]
t = t[t["유형"] != "제외"]
for name, g in t.groupby(t["유형"].str.extract(r"^(\S+?)[\(\s]")[0].fillna(t["유형"])):
    print(f"   상위30 {name:<8} {len(g):>2}곳  평균 접근성 하위% {g['접근성_하위%'].mean():5.1f}  하위25% 이내 {(g['접근성_하위%'] <= LOW_Q).sum()}곳")
