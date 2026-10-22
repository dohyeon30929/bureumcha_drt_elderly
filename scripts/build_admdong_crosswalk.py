"""생활이동 행정동코드(행안부 8자리) -> 경계파일 행정동코드(통계청 8자리) 매핑표 생성.

입력
  - 국가데이터처_법정동 연계정보_20250602.csv (cp949): 서울 2025-04-01 개정본만 사용
  - seoul_purpose_admdong4_in_20260831.zip: 생활이동 O-D 코드 목록(2026-08 기준)
  - bnd_dong_11_2025_2Q.zip: 경계 파일 코드 검증용
출력
  - 행정동_코드매핑_생활이동_경계.csv (utf-8-sig)

2025-07-01 용신동이 신설동+용두동으로 분동됨. 연계표(2025-04-01)와 경계(2025-06-30)는 분동 전이라
생활이동의 11230515, 11230533 을 모두 옛 용신동(경계 11060810)에 합산 대응시킨다.
두 코드 중 어느 쪽이 신설동/용두동인지는 확인하지 못했다.
"""
import csv
import glob
import io
import struct
import zipfile

EXTRA = {"11230515": "11060810", "11230533": "11060810"}  # 분동 후 코드 -> 옛 용신동
OUT = "data/interim/재현확인/행정동_코드매핑_생활이동_경계.csv"

link = {}  # 행안부 8자리 -> (시군구, 행정동명, 통계청 8자리)
for r in csv.DictReader(open(glob.glob("data/raw/국가데이터처_법정동 연계*.csv")[0], encoding="cp949")):
    if (r["시도명"].startswith("서울") and r["개정일자"] == "2025-04-01"
            and len(r["행정구역코드"]) == 8):
        link[r["행정동코드"][:8]] = (r["시군구명"], r["행정동명"], r["행정구역코드"])

z = zipfile.ZipFile("data/raw/bnd_dong_11_2025_2Q.zip")
dbf = z.read("bnd_dong_11_2025_2Q.dbf")
n, hl, rl = struct.unpack("<xxxxIHH", dbf[:12])
bnd = {dbf[hl + i * rl + 9:hl + i * rl + 17].decode().strip() for i in range(n)}

od = set()
zo = zipfile.ZipFile("data/raw/seoul_purpose_admdong4_in_20260831.zip")
with zo.open(zo.namelist()[0]) as f:
    rd = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
    next(rd)
    for row in rd:
        od.update((row[0], row[1]))
od = sorted(c for c in od if c.startswith("11"))

names = {v[2]: (v[0], v[1]) for v in link.values()}
rows, unmatched = [], []
for c in od:
    if c in link:
        sgg, nm, t = link[c]
        note = ""
    elif c in EXTRA:
        t = EXTRA[c]
        sgg, nm = names[t]
        note = "2025-07-01 용신동 분동(신설동/용두동) - 옛 용신동 폴리곤에 합산"
    else:
        unmatched.append(c)
        continue
    rows.append((c, t, sgg, nm, note))

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mois_admdong_cd", "bnd_admdong_cd", "sigungu", "admdong_nm", "note"])
    w.writerows(rows)

print("매핑 행:", len(rows), " 미매핑:", unmatched)
print("경계 코드 커버:", len({r[1] for r in rows} & bnd), "/", len(bnd),
      " 경계에 없는 대응코드:", sorted({r[1] for r in rows} - bnd))
