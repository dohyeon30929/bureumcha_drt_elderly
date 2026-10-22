"""금천구 고령층 목적지 후보 전체 목록: 지오코딩(카카오 로컬 API) + 행정동 공간조인.

실행: 프로젝트 루트에서 `python scripts/금천구_목적지후보.py`
API 키: 프로젝트 루트 .env 파일(KAKAO_REST_API_KEY=...)을 python-dotenv로 읽음(.gitignore 등록). 코드에 키를 쓰지 않는다.
  - 키가 없으면 지오코딩 대상 목록(data/interim/geocode_input.csv)만 만들고 종료한다.
  - 응답은 data/interim/geocode_cache.json 에 캐시해서 재실행 시 API를 다시 부르지 않는다.

입력 (data/raw/)
  - 서울시 병의원 위치 정보.csv            좌표 있음(병원경도/병원위도, WGS84). 금천구 의료기관은 이 파일만 사용
  - 서울특별시 금천구_주민센터 현황_*.csv   좌표 있음(위도/경도)
  - 서울시 금천구 사회복지시설 목록.csv     주소만 -> 지오코딩. 고령층 관련 5종만 사용
  - (서울시경로당)현황3644(25.6월말 기준).xlsx  주소만(도로명, 시군구 생략) -> 지오코딩. 운영상태 컬럼 없음
  - 서울시버스정류소위치정보(*.xlsx)       좌표 있음(X/Y=경도/위도). 분석 범위 내부 + 경계 300m 이내, 한강선착장 제외
                                           (서울시 파일이라 시계 밖 안양·광명 정류소는 없음)
  - 서울시 역사마스터 정보.csv             좌표 있음(위도/경도). 분석 범위 내부 + 경계 1km 이내(밖이면 동="금천구 밖(인접)")
  - bnd_dong_11_2025_2Q.zip                 행정동 경계(EPSG:5179, ADM_CD=통계청 코드)
출력
  - outputs/금천구_목적지후보.csv          시흥1·3·4·5동 클러스터. 시설명, 유형, 동, 위도, 경도
  - outputs/독산2동_2단계후보.csv           독산2동(2단계 참고용). 같은 컬럼·같은 규칙
  - data/interim/금천구_목적지후보_상세.csv  금천구 전체 + 범위 포함 여부·경계거리·주소·좌표출처·비고 상세본
  - data/interim/geocode_result.csv        지오코딩 상세(질의 주소, 매칭 방식, 카카오 반환 주소)

지오코딩 순서: 주소 검색(address.json) -> 실패 시 키워드 검색(keyword.json, "금천구 시설명").
검증: 좌표가 금천구 경계 밖이면 "좌표_금천구밖" 표시(상세본에는 남기되 범위별 요약본에서는 빠짐).
"""
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

RAW = "data/raw"
INTERIM = "data/interim"
CACHE = f"{INTERIM}/geocode_cache.json"
OUT_DETAIL = "data/interim/금천구_목적지후보_상세.csv"  # 주소·좌표출처·5179좌표·비고 포함
GU_PREFIX = "11180"  # 금천구 통계청 시군구 코드
# 분석 범위: 일반 시설은 범위 내부만, 버스정류소는 범위 경계 BUS_BUFFER_M 이내, 지하철역은 STATION_BUFFER_M 이내
SCOPES = {
    "시흥클러스터": {"11180570": "시흥1동", "11180590": "시흥3동", "11180600": "시흥4동", "11180610": "시흥5동"},
    "독산2동": {"11180530": "독산2동"},
}
SCOPE_OUT = {"시흥클러스터": "outputs/금천구_목적지후보.csv", "독산2동": "outputs/독산2동_2단계후보.csv"}
BUS_BUFFER_M = 300
STATION_BUFFER_M = 1000
WELFARE_KINDS = ["재가노인복지시설", "장기요양시설", "노인의료복지시설", "노인여가복지시설", "노인주거복지시설"]
# 고령층 5종 121개 전부 지오코딩(불완전 주소도 시도 -> 실패는 '지오코딩 실패'로 보고). 금천 밖 주소만 제외
EXCLUDE_INCOMPLETE = []
NOTE_KYUNGRO = "경로당 현황은 운영·운영예정·미운영·휴지·폐지 포함이며 운영상태 컬럼이 없어 폐지 경로당을 구분할 수 없음"


def load_key():
    load_dotenv(".env")  # .env 값을 환경변수로 로드(이미 설정된 환경변수가 우선)
    return os.environ.get("KAKAO_REST_API_KEY", "").strip()


def clean_addr(a, prefix=""):
    """괄호(법정동·건물명)와 쉼표 뒤 층·호를 떼고, 'N 번지 M 호' -> 'N-M'으로 정규화."""
    a = str(a).split("(")[0].split(",")[0].strip()
    a = re.sub(r"(\d+)\s*번지\s*(\d+)\s*호", r"\1-\2", a)
    a = re.sub(r"(\d+)\s*번지", r"\1", a)
    return re.sub(r"\s+", " ", prefix + a).strip()


# ---- 행정동 경계
bnd = gpd.read_file(f"zip://{RAW}/bnd_dong_11_2025_2Q.zip")
gc = bnd[bnd.ADM_CD.astype(str).str.startswith(GU_PREFIX)][["ADM_CD", "ADM_NM", "geometry"]]

# ---- 좌표가 이미 있는 데이터
h = pd.read_csv(f"{RAW}/서울시 병의원 위치 정보.csv", encoding="cp949", low_memory=False)  # 의료기관 현황 CSV는 이 파일과 중복이라 미사용
h = h[h.주소.str.contains("금천구", na=False)]
hosp = pd.DataFrame({"유형": "의료기관", "세부유형": h.병원분류명, "시설명": h.기관명, "원주소": h.주소, "원본ID": h.기관ID,
                     "경도": h.병원경도, "위도": h.병원위도, "좌표출처": "원본"})
cp = glob.glob(f"{RAW}/서울특별시 금천구_주민센터 현황_*.csv")[0]
c = pd.read_csv(cp, encoding="utf-8-sig")
center = pd.DataFrame({"유형": "주민센터", "세부유형": "주민센터", "시설명": c.주민센터명, "원주소": c.도로명주소,
                       "원본ID": "주민센터-" + c.주민센터명, "경도": c.경도, "위도": c.위도, "좌표출처": "원본"})
# 주소는 금천구인데 원본 좌표가 금천구 밖인 의료기관(예: 현대연합의원 -> 강서구 공항동) -> 원본 좌표 버리고 지오코딩
hpt = gpd.GeoSeries(gpd.points_from_xy(hosp.경도, hosp.위도), crs=4326, index=hosp.index).to_crs(bnd.crs)
bad_xy = ~hpt.within(gc.union_all())
print(f"의료기관: 금천 주소 {len(hosp)}개 중 원본 좌표가 금천구 밖 {bad_xy.sum()}개 -> 재지오코딩", hosp.loc[bad_xy, "시설명"].tolist())
h_geo = hosp[bad_xy].drop(columns=["경도", "위도", "좌표출처"]).assign(질의주소=lambda x: x.원주소.map(clean_addr))
hosp = hosp[~bad_xy]

# ---- 지오코딩 대상
welfare = pd.read_csv(f"{RAW}/서울시 금천구 사회복지시설 목록.csv", encoding="cp949")
welfare = welfare[welfare["시설종류상세명(시설종류)"].isin(WELFARE_KINDS)]
n_kind = len(welfare)
welfare = welfare[welfare.시설주소.fillna("").str.startswith("서울특별시 금천구") & ~welfare.시설명.isin(EXCLUDE_INCOMPLETE)]
print(f"사회복지시설: 고령층 5종 {n_kind}개 -> 금천 밖 주소 제외 후 {len(welfare)}개")
w = pd.DataFrame({"유형": "사회복지시설", "세부유형": welfare["시설종류상세명(시설종류)"], "시설명": welfare.시설명,
                  "원주소": welfare.시설주소, "질의주소": welfare.시설주소.fillna("").map(clean_addr), "원본ID": welfare.시설코드})

kp = glob.glob(f"{RAW}/(서울시경로당)*.xlsx")[0]
k = pd.read_excel(kp, header=None, skiprows=6).iloc[:, :9]
k.columns = ["연번", "시도명", "시군구명", "시설종류", "시설명", "주소", "지자체명", "담당부서", "전화번호"]
k = k[k.연번.notna() & (k.시군구명 == "금천구")]
print(f"경로당: 금천 {len(k)}개 (운영상태 미확인, 전부 포함)")
kk = pd.DataFrame({"유형": "경로당", "세부유형": "경로당(노인여가복지시설)", "시설명": k.시설명, "원주소": k.주소,
                   "질의주소": k.주소.map(lambda a: clean_addr(a, "서울특별시 금천구 ")), "원본ID": "경로당-" + k.연번.astype(int).astype(str)})
geo_in = pd.concat([w, kk, h_geo], ignore_index=True)
# 같은 시설이 종류별로 따로 등록된 경우(시설코드만 다름): 행은 남기고 표시, 고유 지점 수는 별도 집계
geo_in["동일시설중복"] = geo_in.duplicated(["시설명", "질의주소"], keep=False).map({True: "Y", False: ""})
os.makedirs(INTERIM, exist_ok=True)
geo_in.to_csv(f"{INTERIM}/geocode_input.csv", index=False, encoding="utf-8-sig")

KEY = load_key()
if not KEY:
    print(f"\nKAKAO_REST_API_KEY 없음 -> 지오코딩 대상 {len(geo_in)}개를 {INTERIM}/geocode_input.csv 에 저장하고 종료.")
    print(".env.example 을 .env 로 복사해 키를 넣거나 환경변수로 설정한 뒤 다시 실행하세요.")
    sys.exit(0)

# ---- 카카오 지오코딩 (캐시)
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def kakao(kind, query):
    ck = f"{kind}|{query}"
    if ck in cache:
        return cache[ck]
    url = f"https://dapi.kakao.com/v2/local/search/{kind}.json?" + urllib.parse.urlencode({"query": query, "size": 5})
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            docs = json.load(r).get("documents", [])
    except urllib.error.HTTPError as e:
        sys.exit(f"카카오 API 오류 {e.code}: {e.read().decode('utf-8', 'ignore')[:200]} (키·앱 설정 확인)")
    cache[ck] = docs
    time.sleep(0.05)
    return docs


res = []
for r in geo_in.itertuples():
    docs, how = kakao("address", r.질의주소), "주소검색"
    docs = [d for d in docs if "금천구" in d.get("address_name", "")]
    if not docs:
        docs, how = kakao("keyword", f"금천구 {r.시설명}"), "키워드검색"
        docs = [d for d in docs if "금천구" in d.get("address_name", "")]
    if docs:
        d = docs[0]
        res.append({"경도": float(d["x"]), "위도": float(d["y"]), "좌표출처": f"카카오 {how}",
                    "카카오주소": d.get("road_address_name") or d.get("address_name"), "후보수": len(docs),
                    "매칭유형": d.get("address_type", "POI")})
    else:
        res.append({"경도": None, "위도": None, "좌표출처": "지오코딩 실패", "카카오주소": None, "후보수": 0, "매칭유형": None})
json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
geo = pd.concat([geo_in, pd.DataFrame(res)], axis=1)
geo.to_csv(f"{INTERIM}/geocode_result.csv", index=False, encoding="utf-8-sig")
print("\n지오코딩 결과:", geo.groupby(["유형", "좌표출처"]).size().to_dict())

# 지하철역: 같은 역사명은 호선별 행을 1개로(좌표 평균). 금천구 경계 1km 이내 전부 수집 -> 범위별로 다시 거름
st = pd.read_csv(f"{RAW}/서울시 역사마스터 정보.csv", encoding="cp949")
st = st.groupby("역사명").agg(호선=("호선", lambda x: ",".join(sorted(set(x)))), 위도=("위도", "mean"), 경도=("경도", "mean"),
                            원본ID=("역사_ID", lambda x: ",".join(map(str, x)))).reset_index()
spt = gpd.GeoSeries(gpd.points_from_xy(st.경도, st.위도), crs=4326, index=st.index).to_crs(bnd.crs)
st = st[spt.distance(gc.union_all()) <= STATION_BUFFER_M]
station = pd.DataFrame({"유형": "지하철역", "세부유형": st.호선, "시설명": st.역사명 + "역", "원주소": None, "원본ID": st.원본ID,
                        "경도": st.경도, "위도": st.위도, "좌표출처": "원본"})
print(f"지하철역: 금천구 경계 {STATION_BUFFER_M}m 이내 {len(station)}개", station.시설명.tolist())

# 버스정류소(교통연계점): 모든 범위의 경계 300m 이내 수집. 한강선착장 제외(accessibility_metrics.py와 동일)
scope_poly = {k: gc[gc.ADM_CD.astype(str).isin(v)].union_all() for k, v in SCOPES.items()}
bs = pd.read_excel(glob.glob(f"{RAW}/서울시버스정류소위치정보*.xlsx")[0], dtype={"NODE_ID": str, "ARS_ID": str})
bs = bs[bs.정류소타입 != "한강선착장"]
bpt = gpd.GeoSeries(gpd.points_from_xy(bs.X좌표, bs.Y좌표), crs=4326, index=bs.index).to_crs(bnd.crs)
bs = bs[np.min([bpt.distance(pl) for pl in scope_poly.values()], axis=0) <= BUS_BUFFER_M]
bus = pd.DataFrame({"유형": "버스정류소", "세부유형": bs.정류소타입, "시설명": bs.정류소명, "원주소": None,
                    "원본ID": "ARS-" + bs.ARS_ID + "/NODE-" + bs.NODE_ID, "경도": bs.X좌표, "위도": bs.Y좌표, "좌표출처": "원본"})
print(f"버스정류소: 분석 범위 경계 {BUS_BUFFER_M}m 이내 {len(bus)}개 수집")

allf = pd.concat([hosp, center, station, bus, geo.drop(columns=["질의주소"])], ignore_index=True)

# ---- 행정동 공간조인
ok = allf.경도.notna()
g = gpd.GeoDataFrame(allf[ok], geometry=gpd.points_from_xy(allf.loc[ok, "경도"], allf.loc[ok, "위도"]), crs=4326).to_crs(bnd.crs)
g = gpd.sjoin(g, gc, how="left", predicate="within").drop(columns="index_right")
g = g[~g.index.duplicated(keep="first")]
# 동 경계 틈에 걸린 버스정류소는 100m 이내 최근접 동으로 배정(accessibility_metrics.py와 같은 규칙).
# 버스만 적용: 지하철역은 서울 밖 인접역(석수역 등)이 있어 적용하면 금천 안으로 잘못 끌려옴
gap = g.ADM_CD.isna() & (g.유형 == "버스정류소")
if gap.any():
    near = gpd.sjoin_nearest(g.loc[gap, ["geometry"]], gc, how="left", max_distance=100)
    near = near[~near.index.duplicated(keep="first")]
    g.loc[near.index, ["ADM_CD", "ADM_NM"]] = near[["ADM_CD", "ADM_NM"]].values
g["x_5179"], g["y_5179"] = g.geometry.x.round(1), g.geometry.y.round(1)
for k, pl in scope_poly.items():
    g[f"{k}_경계거리_m"] = g.geometry.distance(pl).round(0)
allf = pd.concat([pd.DataFrame(g.drop(columns="geometry")), allf[~ok]], ignore_index=True)
allf = allf.rename(columns={"ADM_CD": "행정동코드", "ADM_NM": "행정동"})
allf["행정동코드"] = allf.행정동코드.astype("string").str.replace(r"\.0$", "", regex=True)

# 범위 포함 여부: 일반 시설은 범위 내부, 버스는 경계 300m, 지하철은 경계 1km 이내
for k, dongs in SCOPES.items():
    dist = allf[f"{k}_경계거리_m"]
    inside = allf.행정동코드.isin(dongs)
    allf[k] = np.select([inside, (allf.유형 == "버스정류소") & (dist <= BUS_BUFFER_M),
                         (allf.유형 == "지하철역") & (dist <= STATION_BUFFER_M)], ["내부", "경계인접", "경계인접"], "")

allf["비고"] = ""
outside = allf.경도.notna() & allf.행정동.isna()
allf.loc[outside, "비고"] = np.where(allf.loc[outside, "유형"] == "지하철역", "금천구 밖 인접역", "좌표_금천구밖")
allf.loc[allf.좌표출처 == "지오코딩 실패", "비고"] = "지오코딩 실패"
# 정확도 낮은 좌표 표시: 번지 없는 도로명(ROAD=도로 대표점), 주소 매칭 실패 후 시설명 키워드검색(POI)
allf.loc[allf.매칭유형 == "ROAD", "비고"] = "번지 없음: 도로 대표점 좌표(위치 부정확)"
kw = allf.좌표출처 == "카카오 키워드검색"
allf.loc[kw, "비고"] = "주소 검색 실패 -> 시설명 키워드검색 좌표(카카오주소와 원주소 비교 확인 필요)"
dup = allf.동일시설중복 == "Y"
allf.loc[dup, "비고"] = (allf.loc[dup, "비고"] + " 동일 시설 다중 등록(종류별)").str.strip()
allf.loc[allf.유형 == "경로당", "비고"] = (allf.loc[allf.유형 == "경로당", "비고"] + " " + NOTE_KYUNGRO).str.strip()
# 지하철역·버스정류소는 목적지가 아니라 DRT가 연결할 교통연계점
allf["유형"] = allf.유형.replace({"의료기관": "병의원", "지하철역": "지하철역(연계)", "버스정류소": "버스정류소(연계)"})
allf = allf.sort_values(["행정동", "유형", "시설명"])
cols = ["유형", "세부유형", "시설명", "원주소", "행정동코드", "행정동", *SCOPES, *[f"{k}_경계거리_m" for k in SCOPES],
        "경도", "위도", "x_5179", "y_5179", "좌표출처", "매칭유형", "카카오주소", "원본ID", "동일시설중복", "비고"]
allf[cols].to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
print(f"\n상세 저장: {OUT_DETAIL} ({len(allf)}행, 금천구 전체)")

# ---- 범위별 요약본: 시설명, 유형, 동, 위도, 경도
for k, out in SCOPE_OUT.items():
    sc = allf[allf[k] != ""].copy()
    sc["동"] = sc.행정동.fillna("금천구 밖(인접)")
    sc[["시설명", "유형", "동", "위도", "경도"]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n===== {k} -> {out} ({len(sc)}행)")
    print("[유형 x 동] (범위 밖 동 = 버스 300m·지하철 1km 경계인접)")
    print(pd.crosstab(sc.유형, sc.동, margins=True, margins_name="합계").to_string())
    bt = sc[sc.유형 == "버스정류소(연계)"]
    print("[버스정류소: 내부/경계인접 x 정류소타입]")
    print(pd.crosstab(bt[k], bt.세부유형, margins=True, margins_name="합계").to_string())
    print("[지하철역]", sc.loc[sc.유형 == "지하철역(연계)", ["시설명", "세부유형", "동", f"{k}_경계거리_m"]].to_string(index=False))
    u = sc.drop_duplicates(["유형", "시설명", "동"])
    print("[같은 이름 1개로 센 수(버스는 정류소명 기준)]", u.groupby("유형").size().to_dict())

bad = allf[allf.비고.str.contains("실패|금천구밖|부정확|키워드")]
if len(bad):
    print("\n[확인 필요]"); print(bad[["유형", "시설명", "원주소", "카카오주소", "행정동", "비고"]].to_string(index=False))
