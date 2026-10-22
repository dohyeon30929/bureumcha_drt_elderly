"""MCLP·목적지 가정 검증: 생활이동 O-D(2026-08-31 월요일, 60plus_cnt)로 시흥3·4·5동 출발 60세 이상 이동의 도착지 분포.

실행: 프로젝트 루트에서 `python scripts/금천구_OD검증.py`
입력: data/raw/seoul_purpose_admdong4_in_20260831.zip (o/d = 행안부 행정동코드, 내국인), data/raw/전국_행정동_코드정보.zip(ADMI_202608)
한계: 서울 밖 도착지는 시군구 단위로만 제공(안양 석수동 등 동 단위 불가), 하루치(월요일 1일), 60세 이상(70세 이상 아님), 소량 이동은 0으로 처리된 것으로 보임(3·4·5동 출발 행의 62%가 0, 최소 양수 1.15),
      목적 구분 없음(귀가·병원 등 혼재), 같은 동 안 이동은 일부만 잡힘
시간대: st_time_cd(출발 시각). 러시아워는 20분 단위(700, 720 …)라 hour = time_cd // 100 으로 합침
"""
import io
import zipfile

import pandas as pd

OUT = "outputs"
ORIG = {"11545690": "시흥3동", "11545700": "시흥4동", "11545710": "시흥5동"}
S1 = "11545670"

z = zipfile.ZipFile("data/raw/seoul_purpose_admdong4_in_20260831.zip")
od = pd.read_csv(z.open(z.namelist()[0]), dtype={"o_admdong_cd": str, "d_admdong_cd": str})
nm = pd.read_csv(io.BytesIO(zipfile.ZipFile("data/raw/전국_행정동_코드정보.zip").read("SEOUL_ADMI/ADMI_202608.csv")),
                 encoding="utf-8-sig", dtype=str).set_index("ADMI_CD")
od["hour"] = od.st_time_cd.where(od.st_time_cd < 100, od.st_time_cd // 100)


def group(o, d):
    if o == d:
        return "① 같은 동 안"
    if d in ORIG:
        return "② 클러스터 내 다른 동(시흥3·4·5)"
    if d == S1:
        return "③ 시흥1동"
    if d == "11545680":
        return "④ 시흥2동"
    if d in ("11545610", "11545620", "11545630", "11545640"):
        return "⑤ 독산1~4동"
    if d == "11545510":
        return "⑥ 가산동"
    if d.startswith("11"):
        return "⑦ 금천구 밖 서울"
    if d.startswith("4117"):
        return "⑧ 안양시"
    if d.startswith("4121"):
        return "⑨ 광명시"
    return "⑩ 기타(경기·인천 등)"


o = od[od.o_admdong_cd.isin(ORIG)].copy()
o["출발동"] = o.o_admdong_cd.map(ORIG)
o["묶음"] = [group(a, b) for a, b in zip(o.o_admdong_cd, o.d_admdong_cd)]
sgg = nm.assign(k=nm.index.str[:5]).drop_duplicates("k").set_index("k")
# 서울 밖 도착지는 O-D에 시군구 단위(뒷자리 000)로만 들어 있음 -> 시군구 이름으로 표시
o["도착동"] = o.d_admdong_cd.map(lambda c: f"{nm.SGG_NM[c]} {nm.ADMI_NM[c]}" if c in nm.index
                                 else f"{sgg.SIDO_NM.get(c[:5], '?')} {sgg.SGG_NM.get(c[:5], c)}(시군구 전체)")

# ---- 1. 묶음별 비율
g = o.groupby("묶음")[["60plus_cnt", "total_cnt"]].sum()
t1 = pd.DataFrame({"60세이상_명": g["60plus_cnt"].round(0), "60세이상_%": (g["60plus_cnt"] / g["60plus_cnt"].sum() * 100).round(1),
                   "동밖이동중_%": (g["60plus_cnt"] / g.drop("① 같은 동 안")["60plus_cnt"].sum() * 100).round(1),
                   "전연령_%(참고)": (g.total_cnt / g.total_cnt.sum() * 100).round(1)})
t1.loc["① 같은 동 안", "동밖이동중_%"] = None
byo = o.pivot_table(index="묶음", columns="출발동", values="60plus_cnt", aggfunc="sum").fillna(0)
byo = (byo / byo.sum() * 100).round(1).add_suffix("_%")
t1 = t1.join(byo)
t1.loc["합계"] = [round(g["60plus_cnt"].sum()), 100.0, 100.0, 100.0] + [100.0] * len(byo.columns)
t1.to_csv(f"{OUT}/OD검증_도착지묶음_60세이상.csv", encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(f"[1] 시흥3·4·5동 출발 60세 이상 이동 {g['60plus_cnt'].sum():,.0f}건(추정 통행량, 2026-08-31 월)")
print(t1.to_string())
sub = o[~o.d_admdong_cd.isin(ORIG) & ~o.d_admdong_cd.str.startswith("11545")]
print("    금천구 밖 서울 상위 구:", sub[sub.d_admdong_cd.str.startswith("11")].assign(구=lambda x: x.d_admdong_cd.map(nm.SGG_NM))
      .groupby("구")["60plus_cnt"].sum().sort_values(ascending=False).head(5).round(0).to_dict())

# ---- 2·3. 도착 동 순위 (출발 동 자신 제외)
x = o[o.o_admdong_cd != o.d_admdong_cd]
top = x.groupby("도착동")["60plus_cnt"].sum().sort_values(ascending=False)
tt = pd.DataFrame({"60세이상_명": top.round(0), "동밖이동중_%": (top / top.sum() * 100).round(1)})
tt["순위"] = range(1, len(tt) + 1)
tt["누적_%"] = tt["동밖이동중_%"].cumsum().round(1)
tt.head(30).to_csv(f"{OUT}/OD검증_도착동순위_60세이상.csv", encoding="utf-8-sig")
print("\n[2·3] 도착 동 순위 (출발 동 안 이동 제외, 상위 15):")
print(tt.head(15).to_string())
for d, g_ in x.groupby("출발동"):
    s = g_.groupby("도착동")["60plus_cnt"].sum().sort_values(ascending=False)
    print(f"    {d} 출발 상위 5:", (s / s.sum() * 100).round(1).head(5).to_dict())

# ---- 4. 시간대별 가는/오는 방향
go = od[od.o_admdong_cd.isin(ORIG) & (od.d_admdong_cd == S1)]
back = od[(od.o_admdong_cd == S1) & od.d_admdong_cd.isin(ORIG)]
allout = od[od.o_admdong_cd.isin(ORIG) & ~od.d_admdong_cd.isin(ORIG)]
allin = od[~od.o_admdong_cd.isin(ORIG) & od.d_admdong_cd.isin(ORIG)]
hr = pd.DataFrame({"가는_3·4·5동→시흥1동": go.groupby("hour")["60plus_cnt"].sum(),
                   "오는_시흥1동→3·4·5동": back.groupby("hour")["60plus_cnt"].sum(),
                   "참고_3·4·5동→클러스터밖 전체": allout.groupby("hour")["60plus_cnt"].sum(),
                   "참고_클러스터밖→3·4·5동 전체": allin.groupby("hour")["60plus_cnt"].sum()}).reindex(range(24)).fillna(0)
for d, c in ORIG.items():
    hr[f"가는_{c}"] = go[go.o_admdong_cd == d].groupby("hour")["60plus_cnt"].sum().reindex(range(24)).fillna(0)
    hr[f"오는_{c}"] = back[back.d_admdong_cd == d].groupby("hour")["60plus_cnt"].sum().reindex(range(24)).fillna(0)
hr.index.name = "출발시"
hr.round(1).to_csv(f"{OUT}/OD검증_시간대별_시흥1동_왕복_60세이상.csv", encoding="utf-8-sig")
print("\n[4] 시간대별(출발 시각) 60세 이상 이동량")
print(hr.iloc[:, :4].round(0).astype(int).T.to_string())
print(f"    합계: 가는 {hr.iloc[:, 0].sum():,.0f} / 오는 {hr.iloc[:, 1].sum():,.0f}")
print(f"    가는 방향 오전(6~12시) 비중 {hr.iloc[6:12, 0].sum() / hr.iloc[:, 0].sum():.0%} / 오는 방향 오후(12~19시) 비중 {hr.iloc[12:19, 1].sum() / hr.iloc[:, 1].sum():.0%}")
mt = go.assign(w=go["60plus_cnt"]).query("w > 0")
print(f"    가는 방향 평균 이동거리 {(mt.move_dist * mt.w).sum() / mt.w.sum():,.0f}m, 평균 이동시간 {(mt.move_time * mt.w).sum() / mt.w.sum():.0f}분 (단위는 원자료 기준)")
print("\n저장: OD검증_도착지묶음_60세이상.csv, OD검증_도착동순위_60세이상.csv, OD검증_시간대별_시흥1동_왕복_60세이상.csv")
