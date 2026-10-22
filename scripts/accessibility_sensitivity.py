"""2단계 착수 전 점검: (1) 세부지표 상관 (2) 가중치 민감도(50:50 vs 실측 이용비율 58:42) (3) 강건성 판정.

대상: 귀가시간대_우선순위_교차검증.csv의 387개 동 (접근성_지표_행정동.csv를 경계코드로 결합)
종합지수 = 세부지표의 387개 동 내 백분위(0~100, 높을수록 접근성 좋음)의 가중합
  버스밀도(+), 마을버스제외밀도(+), 지하철최단거리(-, 가까울수록 좋음), 1km내역수(+)
  모드 내부 배분은 밀도:마을버스제외 = 0.7:0.3, 거리:1km역수 = 0.7:0.3
  A안: 버스 0.5 : 지하철 0.5 (원래 제안)
  B안: 버스 0.416 : 지하철 0.584 (귀가시간대 서울 고령층 수단 구성 버스 13.6% : 지하철 19.1%)
강건성 판정 기준(결과를 보기 전에 고정):
  통과 = 9곳 모두 (a) 순위 변동폭 <= 5%(387개 중 19위) 이고 (b) 하위 사분위(하위 25%) 소속 여부가 A/B에서 같다
"""
import pandas as pd

pd.set_option("display.width", 250)
main = pd.read_csv("data/interim/재현확인/귀가시간대_우선순위_교차검증.csv", encoding="utf-8-sig", dtype={"경계코드": str})
acc = pd.read_csv("data/interim/재현확인/접근성_지표_행정동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
d = main[["경계코드", "동이름", "자치구"]].merge(acc.drop(columns=["동이름", "자치구"]), on="경계코드", how="left")
assert len(d) == 387 and d["버스정류소수"].notna().all()

M = {"버스밀도": "버스정류소밀도_개per_km2", "마을버스제외밀도": None, "지하철최단거리": "지하철최단거리_m", "1km내역수": "지하철역수_1km"}
d["마을버스제외밀도"] = d["버스정류소수_마을버스제외"] / d["면적_km2"]
cols = {"버스밀도": "버스정류소밀도_개per_km2", "마을버스제외밀도": "마을버스제외밀도",
        "지하철최단거리": "지하철최단거리_m", "1km내역수": "지하철역수_1km"}
X = d[list(cols.values())].copy()
X.columns = list(cols)

print("== 1) 피어슨 상관행렬 (387개 동, 원값)")
print(X.corr().round(3).to_string())
print("\n   (참고) 스피어만 상관 - 종합지수는 순위 기반이라 이쪽이 더 직접적")
print(X.rank().corr().round(3).to_string())
print("\n   참고: 지하철최단거리는 클수록 나쁨이라 음의 상관이 '같은 방향'을 뜻함")

# 백분위(높을수록 좋음)
P = pd.DataFrame({
    "버스밀도": X["버스밀도"].rank(pct=True), "마을버스제외밀도": X["마을버스제외밀도"].rank(pct=True),
    "지하철최단거리": (-X["지하철최단거리"]).rank(pct=True), "1km내역수": X["1km내역수"].rank(pct=True)}) * 100

def index(bus_w, sub_w):
    w = {"버스밀도": bus_w * 0.7, "마을버스제외밀도": bus_w * 0.3, "지하철최단거리": sub_w * 0.7, "1km내역수": sub_w * 0.3}
    return sum(P[k] * v for k, v in w.items()), w

A, wA = index(0.5, 0.5)
B, wB = index(0.416, 0.584)
print("\n== 2) 가중치")
print("   A안:", {k: round(v, 3) for k, v in wA.items()})
print("   B안:", {k: round(v, 3) for k, v in wB.items()})
d["지수A"], d["지수B"] = A.round(2), B.round(2)
d["순위A"] = d["지수A"].rank(method="min").astype(int)   # 1 = 접근성 가장 나쁨
d["순위B"] = d["지수B"].rank(method="min").astype(int)
d["하위%A"] = (d["지수A"].rank(pct=True) * 100).round(1)
d["하위%B"] = (d["지수B"].rank(pct=True) * 100).round(1)
d["변동"] = d["순위B"] - d["순위A"]

print(f"\n   전체 387개 동: A-B 지수 피어슨 {A.corr(B):+.4f}, 순위 스피어만 {A.rank().corr(B.rank()):+.4f}, "
      f"순위 변동 |Δ| 평균 {d['변동'].abs().mean():.1f} / 최대 {d['변동'].abs().max()} / 90분위 {d['변동'].abs().quantile(.9):.0f}")

v = pd.read_csv("data/interim/재현확인/취약지대_후보_9곳.csv", encoding="utf-8-sig", dtype={"경계코드": str})
r = v[["우선순위_순위", "경계코드"]].merge(d, on="경계코드")
show = r[["우선순위_순위", "자치구", "동이름", "지수A", "지수B", "순위A", "순위B", "변동", "하위%A", "하위%B"]]
print("\n== 9곳 (순위 1 = 접근성 가장 나쁨, 하위% 작을수록 나쁨)")
print(show.to_string(index=False))
show.to_csv("data/interim/재현확인/접근성_가중치민감도_9곳.csv", index=False, encoding="utf-8-sig")

cut = 0.05 * len(d)
ok_shift = (r["변동"].abs() <= cut).all()
q = lambda x: x <= 25
ok_quart = (q(r["하위%A"]) == q(r["하위%B"])).all()
print(f"\n== 3) 판정 기준: 변동폭 <= {cut:.0f}위(5%) & 하위 25% 소속 동일")
print(f"   최대 변동폭 {r['변동'].abs().max()}위 -> {'충족' if ok_shift else '미충족'} | "
      f"하위25% 소속 불일치 {(q(r['하위%A']) != q(r['하위%B'])).sum()}곳 -> {'충족' if ok_quart else '미충족'}")
print(f"   9곳 평균 하위% : A {r['하위%A'].mean():.1f}  B {r['하위%B'].mean():.1f}   (전체 평균은 50)")
print(f"   하위25% 소속: A {int(q(r['하위%A']).sum())}곳, B {int(q(r['하위%B']).sum())}곳 / 하위50%: A {int((r['하위%A']<=50).sum())}곳, B {int((r['하위%B']<=50).sum())}곳")
print("   결과:", "강건(가중치 선택에 민감하지 않음)" if ok_shift and ok_quart else "민감 - 추가 판단 필요")
