"""접근성 지수 v3의 DRT 도입 전후 비교 (방안 A + 민감도 a·b·c) + MCLP 기준 '기존 정류소 400m 밖 → DRT로 새로 400m 안' 인원.

실행: 프로젝트 루트에서 `python scripts/금천구_접근성_DRT전후.py`
입력(원 계산과 같은 파일): data/raw/접근성_지표_행정동.csv, data/raw/귀가시간대_우선순위_교차검증.csv((초안) 프로젝트에서 복사),
      data/verified/행정동별_고령인구.csv, data/verified/접근성_지수_v3_387동.csv(재현 검증용), outputs/MCLP_v4_최종승차정류소.csv
지수 v3(원 정의 그대로, scripts/accessibility_index_v3.py): 387개 동 백분위(rank pct x 100, 높을수록 좋음) 가중합
  ① 버스정류소 밀도(마을버스 포함 정류소 수 ÷ 면적) .23296  ② 마을버스 제외 밀도 .09984  ④ 지하철 최단거리(부호 반대) .32704
  ⑤ 1km 내 역 수 .14016  ③ 고령 1천명당 정류소(정류소 수 ÷ 65세 이상 x 1000) .2 / 하위 % = 지수의 백분위, 순위 1 = 가장 나쁨
DRT 반영(방안 A, 기본): 기존 버스정류소가 아닌 신규 정차 지점 8곳만, 정류소 1개로 세어 ①·③의 정류소 수에 더함. B01 제외
  민감도 a = 신규 지점을 방향 쌍 2개로 셈 / b = ②(마을버스 제외)에도 더함 / c = B01 포함(위치한 동에 +1)
  387개 동 전체 백분위를 다시 계산한다(상대 지표라 원지표가 그대로인 동도 순위가 움직일 수 있음)
기존 결과 파일은 수정하지 않고 outputs/접근성_DRT전후_*.csv 로만 저장
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

OUT = "outputs"
W = {"버스밀도": 0.416 * 0.8 * 0.7, "마을버스제외밀도": 0.416 * 0.8 * 0.3, "지하철최단거리": 0.584 * 0.8 * 0.7, "1km내역수": 0.584 * 0.8 * 0.3, "고령당정류소": 0.2}
FOCUS = ["시흥1동", "시흥3동", "시흥4동", "시흥5동", "독산2동"]
pc = lambda s: s.rank(pct=True) * 100


def base_table():
    main = pd.read_csv("data/raw/귀가시간대_우선순위_교차검증.csv", encoding="utf-8-sig", dtype={"경계코드": str})
    acc = pd.read_csv("data/raw/접근성_지표_행정동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
    eld = pd.read_csv("data/verified/행정동별_고령인구.csv", encoding="utf-8-sig", dtype={"경계코드": str})
    d = main[["경계코드", "동이름", "자치구"]].merge(acc.drop(columns=["동이름", "자치구"]), on="경계코드").merge(
        eld[["경계코드", "고령인구_추정"]], on="경계코드")
    assert len(d) == 387
    return d


def index(d, add_all=None, add_nv=None):
    """add_all: 경계코드 -> ①·③에 더할 정류소 수, add_nv: ②에 더할 수"""
    a_all = d["경계코드"].map(add_all or {}).fillna(0)
    n_all = d["버스정류소수"] + a_all
    n_nv = d["버스정류소수_마을버스제외"] + d["경계코드"].map(add_nv or {}).fillna(0)
    # 원 스크립트와 같게: 버스밀도는 저장된 밀도 열(소수 3자리)을 쓰고, 정류소를 더한 동만 같은 방식(수 ÷ 면적, 3자리 반올림)으로 다시 계산
    dens = d["버스정류소밀도_개per_km2"].where(a_all == 0, (n_all / d["면적_km2"]).round(3))
    raw = pd.DataFrame({"버스밀도": dens, "마을버스제외밀도": n_nv / d["면적_km2"], "고령당정류소": n_all / d["고령인구_추정"] * 1000,
                        "지하철최단거리": -d["지하철최단거리_m"], "1km내역수": d["지하철역수_1km"]})
    P = raw.apply(pc)
    idx = sum(P[k] * v for k, v in W.items())
    out = pd.DataFrame({"경계코드": d.경계코드, "동이름": d.동이름, "자치구": d.자치구, "정류소수": n_all, "지수": idx.round(2)})
    out["하위%"] = (out["지수"].rank(pct=True) * 100).round(1)
    out["순위"] = out["지수"].rank(method="min").astype(int)
    return out, P


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = base_table()
    before, Pb = index(d)
    # ---------------- 재현 검증
    ref = pd.read_csv("data/verified/접근성_지수_v3_387동.csv", encoding="utf-8-sig", dtype={"경계코드": str})
    m = before.merge(ref[["경계코드", "지수_v3", "하위%_v3", "순위_v3"]], on="경계코드")
    err = (m["지수"] - m["지수_v3"]).abs().max()
    rk = int((m["순위"] != m["순위_v3"]).sum())
    print(f"[재현 검증] 387개 동 매칭 {len(m)}, 지수 최대 오차 {err:.4f}, 순위 불일치 {rk}개, 하위% 최대 오차 {(m['하위%'] - m['하위%_v3']).abs().max():.2f}")
    if len(m) != 387 or err > 0.01 or rk:
        sys.exit("v3 재현 실패 -> 중단")

    # ---------------- DRT 정차 지점이 놓인 동
    st = pd.read_csv(f"{OUT}/MCLP_v4_최종승차정류소.csv", encoding="utf-8-sig")
    g = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.x_5179, st.y_5179), crs=5179)
    bnd = gpd.read_file("zip://data/raw/bnd_dong_11_2025_2Q.zip")[["ADM_CD", "ADM_NM", "geometry"]]
    g = gpd.sjoin(g, bnd, how="left", predicate="within")
    g["신규"] = g.유형 != "기존 버스정류소"
    loc = g[["ID", "정류소명", "유형", "동", "ADM_CD", "ADM_NM", "신규"]].rename(columns={"ADM_CD": "위치_경계코드", "ADM_NM": "위치_행정동(경계)"})
    loc.to_csv(f"{OUT}/접근성_DRT전후_정차지점_위치.csv", index=False, encoding="utf-8-sig")
    print(loc.to_string(index=False))
    new = g[g.신규 & (g.ID != "B01")]
    add1 = new.groupby("ADM_CD").size().to_dict()
    b01 = g[g.ID == "B01"].ADM_CD.iloc[0]
    scen = {"A(기본): 신규 8곳 1개씩, ①·③, B01 제외": (add1, None),
            "민감도 a: 신규 지점 2개씩(방향 쌍)": ({k: 2 * v for k, v in add1.items()}, None),
            "민감도 b: ②(마을버스 제외)에도 더함": (add1, add1),
            "민감도 c: B01 포함": ({**add1, b01: add1.get(b01, 0) + 1}, None)}
    rows, dec = [], []
    for name, (aa, an) in scen.items():
        after, Pa = index(d, aa, an)
        mm = before.merge(after, on=["경계코드", "동이름", "자치구"], suffixes=("_전", "_후"))
        for _, r in mm[mm.동이름.isin(FOCUS) & (mm.자치구 == "금천구")].iterrows():
            rows.append({"안": name, "동": r.동이름, "정류소수 전→후": f"{int(r.정류소수_전)} → {int(r.정류소수_후)}",
                         "지수_전": r.지수_전, "지수_후": r.지수_후, "지수 변화": r.지수_후 - r.지수_전,
                         "하위%_전": r["하위%_전"], "하위%_후": r["하위%_후"], "하위% 변화": r["하위%_후"] - r["하위%_전"],
                         "순위_전": r.순위_전, "순위_후": r.순위_후, "순위 변화(+ = 개선)": r.순위_후 - r.순위_전,
                         "하위25% 소속 전→후": f"{'하위25%' if r['하위%_전'] <= 25 else '밖'} → {'하위25%' if r['하위%_후'] <= 25 else '밖'}",
                         "하위25% 벗어남": r["하위%_전"] <= 25 < r["하위%_후"]})
        if name.startswith("A"):
            i3 = d.index[(d.동이름 == "시흥3동") & (d.자치구 == "금천구")][0]
            for k in W:
                dec.append({"지표": k, "가중치": W[k], "백분위_전": Pb.loc[i3, k], "백분위_후": Pa.loc[i3, k], "백분위 변화": Pa.loc[i3, k] - Pb.loc[i3, k],
                            "지수 기여 변화(가중치 x 변화)": W[k] * (Pa.loc[i3, k] - Pb.loc[i3, k])})
            moved = mm[(mm.순위_전 != mm.순위_후) & ~mm.동이름.isin(FOCUS)]
            print(f"\n방안 A: 순위가 바뀐 다른 동 {len(moved)}개(상대 지표 효과), 하위25% 소속이 바뀐 동:",
                  mm[(mm["하위%_전"] <= 25) != (mm["하위%_후"] <= 25)][["자치구", "동이름", "하위%_전", "하위%_후"]].to_dict("records"))
    R = pd.DataFrame(rows)
    R.round(2).to_csv(f"{OUT}/접근성_DRT전후_비교.csv", index=False, encoding="utf-8-sig")
    DEC = pd.DataFrame(dec)
    DEC.loc[len(DEC)] = {"지표": "합계", "가중치": 1.0, "지수 기여 변화(가중치 x 변화)": DEC["지수 기여 변화(가중치 x 변화)"].sum()}
    DEC.round(3).to_csv(f"{OUT}/접근성_DRT전후_시흥3동_분해.csv", index=False, encoding="utf-8-sig")
    print("\n[전후 비교]")
    print(R.round(2).to_string(index=False))
    print("\n[시흥3동 지표별 백분위 변화, 방안 A]")
    print(DEC.round(2).to_string(index=False))
    a_c, _ = index(d, scen["민감도 c: B01 포함"][0])
    nh = before.merge(a_c, on=["경계코드", "동이름", "자치구"], suffixes=("_전", "_후"))
    print("\n[민감도 c에서 바뀌는 관악구 난향동]")
    print(nh[nh.경계코드 == b01][["자치구", "동이름", "정류소수_전", "정류소수_후", "지수_전", "지수_후", "하위%_전", "하위%_후", "순위_전", "순위_후"]].to_string(index=False))

    # ---------------- MCLP 기준: 기존 버스정류소 보행 400m 밖 -> DRT 정류소로 새로 400m 안 (동별 70세 이상 인원)
    sys.path.insert(0, "scripts")
    from osm_roads import Roads
    dem = pd.read_csv(f"{OUT}/MCLP_v3_수요점_100m보정.csv", encoding="utf-8-sig")
    w, dong = dem.가중치_70세이상.to_numpy(), dem.동.to_numpy()
    cand = pd.read_csv(f"{OUT}/MCLP_후보정류소.csv", encoding="utf-8-sig")
    ex = cand[cand.후보유형 == "기존 버스정류소"]
    roads = Roads()  # MCLP v4와 같은 보행망·같은 함수
    dxy = dem[["x_5179", "y_5179"]].to_numpy()
    cov_ex = np.isfinite(roads.walk_dist(ex[["x", "y"]].to_numpy(), dxy, 400)).any(axis=1)
    drt12 = st[st.ID != "B01"]
    cov_d12 = np.isfinite(roads.walk_dist(drt12[["x_5179", "y_5179"]].to_numpy(), dxy, 400)).any(axis=1)
    cov_d13 = np.isfinite(roads.walk_dist(st[["x_5179", "y_5179"]].to_numpy(), dxy, 400)).any(axis=1)
    rows = []
    for dn in ["시흥3동", "시흥4동", "시흥5동", "합계"]:
        mk = (dong == dn) if dn != "합계" else np.ones(len(w), bool)
        tot = w[mk].sum()
        rows.append({"동": dn, "70세 이상_명": tot, "기존 버스정류소 97곳 400m 안_명": w[mk & cov_ex].sum(), "기존 커버율_%": w[mk & cov_ex].sum() / tot * 100,
                     "기존 400m 밖_명": w[mk & ~cov_ex].sum(),
                     "그중 DRT 12곳(B01 제외)으로 새로 400m 안_명": w[mk & ~cov_ex & cov_d12].sum(),
                     "그중 DRT 13곳(B01 포함)으로 새로 400m 안_명": w[mk & ~cov_ex & cov_d13].sum(),
                     "기존+DRT 12곳 커버율_%": w[mk & (cov_ex | cov_d12)].sum() / tot * 100,
                     "둘 다 400m 밖으로 남는_명(B01 제외)": w[mk & ~cov_ex & ~cov_d12].sum()})
    NC = pd.DataFrame(rows)
    NC.round(1).to_csv(f"{OUT}/접근성_DRT전후_신규400m커버.csv", index=False, encoding="utf-8-sig")
    print(f"\n[기존 버스정류소 {len(ex)}곳(클러스터 내부+경계 300m, 같은 동·같은 이름 방향 쌍 병합) 기준 400m 밖 -> DRT로 새로 400m 안]")
    print(NC.round(1).to_string(index=False))
