"""접근성 지수 v3: 기존 버스:지하철 내부비율(B안, 실측 고령층 이용비율 버스41.6%:지하철58.4%)을 유지한 채
전체를 0.8로 축소하고, 새 요소(고령인구당_버스정류소)에 0.2를 배정.

가중치 산출
  버스 총비중 0.416 x 0.8 = 0.3328, 지하철 총비중 0.584 x 0.8 = 0.4672, 고령당정류소 = 0.2  (합 1.0)
  모드 내부 배분은 기존과 동일하게 밀도:마을버스제외 = 0.7:0.3, 거리:1km역수 = 0.7:0.3 유지
    버스밀도 .23296, 마을버스제외밀도 .09984, 지하철최단거리 .32704, 1km내역수 .14016, 고령당정류소 .2

비교 대상
  지수_기존   = 원래 축①(A안, 버스0.5:지하철0.5), 고령 요소 없음 (접근성_지수_387동.csv)
  지수_v3     = 이번 지시 가중치 (본 스크립트)
출력: 접근성_지수_v3_387동.csv, 접근성_9곳_v3.csv
"""
import pandas as pd

pd.set_option("display.width", 250)
LOW_Q = 25

main = pd.read_csv("data/interim/재현확인/귀가시간대_우선순위_교차검증.csv", encoding="utf-8-sig", dtype={"경계코드": str})
acc = pd.read_csv("data/interim/재현확인/접근성_지표_행정동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
old_idx = pd.read_csv("data/interim/재현확인/접근성_지수_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
eld = pd.read_csv("data/interim/재현확인/행정동별_고령인구.csv", encoding="utf-8-sig", dtype={"경계코드": str})
fin = pd.read_csv("data/interim/재현확인/최종_이동취약지대.csv", encoding="utf-8-sig", dtype={"경계코드": str})

d = main[["경계코드", "동이름", "자치구", "가구수"]].merge(acc.drop(columns=["동이름", "자치구"]), on="경계코드").merge(
    eld[["경계코드", "고령인구_추정", "고령화율", "고령인구밀도_명per_km2"]], on="경계코드")
assert len(d) == 387
d["마을버스제외밀도"] = d["버스정류소수_마을버스제외"] / d["면적_km2"]
d["고령1천명당_정류소"] = d["버스정류소수"] / d["고령인구_추정"] * 1000

pc = lambda s: s.rank(pct=True) * 100
P = pd.DataFrame({
    "버스밀도": pc(d["버스정류소밀도_개per_km2"]), "마을버스제외밀도": pc(d["마을버스제외밀도"]),
    "고령당정류소": pc(d["고령1천명당_정류소"]), "지하철최단거리": pc(-d["지하철최단거리_m"]), "1km내역수": pc(d["지하철역수_1km"])})

BUS, SUB, NEW = 0.416 * 0.8, 0.584 * 0.8, 0.2
W = {"버스밀도": BUS * 0.7, "마을버스제외밀도": BUS * 0.3, "지하철최단거리": SUB * 0.7, "1km내역수": SUB * 0.3, "고령당정류소": NEW}
assert abs(sum(W.values()) - 1.0) < 1e-9
print("가중치:", {k: round(v, 5) for k, v in W.items()}, " 합", round(sum(W.values()), 6))

d["지수_v3"] = sum(P[k] * v for k, v in W.items()).round(2)
d["하위%_v3"] = (d["지수_v3"].rank(pct=True) * 100).round(1)
d["순위_v3"] = d["지수_v3"].rank(method="min").astype(int)
d = d.merge(old_idx[["경계코드", "접근성지수", "접근성_하위%", "접근성순위"]].rename(
    columns={"접근성지수": "지수_기존", "접근성_하위%": "하위%_기존", "접근성순위": "순위_기존"}), on="경계코드")
d["분류_기존"] = (d["하위%_기존"] <= LOW_Q).map({True: "하위25%", False: "하위권밖"})
d["분류_v3"] = (d["하위%_v3"] <= LOW_Q).map({True: "하위25%", False: "하위권밖"})
d["순위변동"] = d["순위_v3"] - d["순위_기존"]

print(f"\n기존 vs v3: 피어슨 {d['지수_기존'].corr(d['지수_v3']):+.4f}, 스피어만 {d['지수_기존'].rank().corr(d['지수_v3'].rank()):+.4f}, "
      f"순위변동 |Δ| 평균 {d['순위변동'].abs().mean():.1f} / 최대 {d['순위변동'].abs().max()}, 하위25% 소속 변경 {int((d['분류_기존'] != d['분류_v3']).sum())}개 동")

out_cols = ["경계코드", "동이름", "자치구", "지수_v3", "하위%_v3", "순위_v3", "분류_v3", "지수_기존", "하위%_기존", "순위_기존",
            "분류_기존", "순위변동", "고령인구_추정", "고령화율", "고령1천명당_정류소", "버스정류소수", "버스정류소밀도_개per_km2",
            "지하철최단거리_m", "최근접역", "지하철역수_1km"]
d[out_cols].sort_values("지수_v3").to_csv("data/interim/재현확인/접근성_지수_v3_387동.csv", index=False, encoding="utf-8-sig")

v = fin[["경계코드", "구분"]].merge(d, on="경계코드").sort_values("지수_v3")
cols9 = ["구분", "자치구", "동이름", "고령인구_추정", "고령1천명당_정류소", "지수_v3", "하위%_v3", "분류_v3", "순위_v3",
         "지수_기존", "하위%_기존", "분류_기존", "순위_기존", "순위변동"]
print("\n== 9곳 (v3 지수 나쁜 순, 하위%는 작을수록 접근성 나쁨)")
print(v[cols9].to_string(index=False))
v[cols9].to_csv("data/interim/재현확인/접근성_9곳_v3.csv", index=False, encoding="utf-8-sig")

n_low = int((v["하위%_v3"] <= LOW_Q).sum())
print(f"\n9곳 중 v3 하위25% {n_low}/9 (기존 7/9). 구성요소 백분위 참고:")
print(P.loc[v.index, ["버스밀도", "마을버스제외밀도", "고령당정류소", "지하철최단거리", "1km내역수"]].round(1)
      .set_index(v["동이름"]).to_string())
