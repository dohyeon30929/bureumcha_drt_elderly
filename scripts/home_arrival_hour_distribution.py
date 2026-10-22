"""1단계: 귀가 도착 이동량의 시간대별 분포 (서울, 8월 한 달, 전 연령 vs 고령층).

귀가 코드는 3이다(인수인계 문서의 6은 오류). 근거: 코드 3만 저녁 17~22시에 몰리고 427개 동 전부에 등장하며
상위 동이 주거지(진관동 등)다. 코드 6은 상위 동이 대형병원 소재지(풍납2동·신촌동·이화동 등)라 병원이다.

time_cd 는 정시(0~23)와 러시아워 20분 단위(700, 720, ... 1940)가 섞여 있어 시(hour)로 합친다.
고령층 = male_60 + male_70 + feml_60 + feml_70
"""
import csv
import glob
import io
import zipfile
from collections import defaultdict

ELDER = ("male_60_cnt", "male_70_cnt", "feml_60_cnt", "feml_70_cnt")
HOME = "3"
OUT = "data/interim/재현확인/귀가(코드3)_시간대별_분포.csv"

zpath = [p for p in glob.glob("data/raw/*.zip") if "성 연령별, 도착지" in p][0]
tot = defaultdict(float)
eld = defaultdict(float)
raw_tc = set()
days = 0
with zipfile.ZipFile(zpath) as z:
    for name in z.namelist():
        days += 1
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
                if r["move_purpose"] != HOME or not r["d_admdong_cd"].startswith("11"):
                    continue
                t = int(r["time_cd"])
                raw_tc.add(t)
                h = t // 100 if t >= 100 else t
                tot[h] += float(r["total_cnt"] or 0)
                eld[h] += sum(float(r[c] or 0) for c in ELDER)

T, E = sum(tot.values()), sum(eld.values())
print(f"파일 {days}개(일), 서울 귀가 도착량 전 연령 {T:,.0f} / 고령층 {E:,.0f} ({E / T:.1%})")
print("time_cd 종류:", sorted(raw_tc))

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시", "전연령_귀가", "전연령_비중", "고령층_귀가", "고령층_비중"])
    for h in range(24):
        w.writerow([h, round(tot[h], 1), round(tot[h] / T, 4), round(eld[h], 1), round(eld[h] / E, 4)])

print("\n시   전연령비중  고령층비중  고령층 막대(1칸=0.5%p)")
for h in range(24):
    print(f"{h:>2}  {tot[h] / T:>8.1%}  {eld[h] / E:>8.1%}  " + "#" * round(eld[h] / E * 200))

def best(d, S, k):
    return max(((sum(d[(h + i) % 24] for i in range(k)) / S, h) for h in range(24)))

for k in (2, 3, 4, 6):
    a, ha = best(tot, T, k)
    b, hb = best(eld, E, k)
    print(f"연속 {k}시간 최대 집중: 전연령 {ha}~{(ha + k) % 24}시 {a:.1%} | 고령층 {hb}~{(hb + k) % 24}시 {b:.1%}")
print("고령층 상위 5개 시:", [(h, f"{eld[h] / E:.1%}") for h in sorted(eld, key=eld.get, reverse=True)[:5]])
