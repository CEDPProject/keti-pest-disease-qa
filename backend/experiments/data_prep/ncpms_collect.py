import os
import re
import sys
import time
import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from backend.core.classes import CLASSES, NCPMS_IMAGE_DIR
from backend.core.settings import NCPMS_API_KEY

BASE = "http://ncpms.rda.go.kr/npmsAPI/service"
CROP = "감귤"
SVC_LIST = "SVC01"
SVC_DETAIL = "SVC05"
SERVICE_TYPE = "AA001"
CITRUS_DISEASES = {info["key"]: label for label, info in CLASSES.items()}

IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
REQ_SLEEP = 0.7
HEADERS = {"User-Agent": "Mozilla/5.0 (research; citrus-pest-collect)"}
KEY_HELP = "먼저 환경변수 NCPMS_API_KEY 를 설정하세요 (https://ncpms.rda.go.kr/npms/OpenApiInfo.np)"


# 순서를 유지하며 중복을 뺀다
def _unique(items: list) -> list:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# XML 태그에서 네임스페이스를 뗀 소문자 이름을 돌려준다
def _localname(tag: str) -> str:
    return tag.split("}")[-1].lower()


# 상대 경로 이미지 주소를 NCPMS 절대 URL 로 바꾼다
def _normalize(u: str):
    u = u.strip()
    if not u:
        return None
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        return "http://ncpms.rda.go.kr" + u
    return "http://ncpms.rda.go.kr/" + u


# 병 목록 응답 XML 에서 sickKey 를 순서대로 모은다
def find_sick_keys(xml_bytes: bytes) -> list:
    keys = []
    try:
        for el in ET.fromstring(xml_bytes).iter():
            if el.tag.lower().endswith("sickkey") and el.text and el.text.strip():
                keys.append(el.text.strip())
    except ET.ParseError:
        keys = re.findall(r"<sickKey>\s*([^<]+?)\s*</sickKey>", xml_bytes.decode("utf-8", "ignore"))
    return _unique(keys)


# 병 상세 XML 에서 사진 URL 을 모은다
def extract_image_urls(xml_bytes: bytes) -> list:
    text = xml_bytes.decode("utf-8", "ignore")
    urls = []
    try:
        root = ET.fromstring(xml_bytes)
        for il in root.iter():
            if _localname(il.tag) == "imagelist":
                for ch in il.iter():
                    if _localname(ch.tag) == "image" and (ch.text or "").strip():
                        urls.append(ch.text.strip())
        if not urls:
            for el in root.iter():
                if _localname(el.tag) == "image" and (el.text or "").strip():
                    urls.append(el.text.strip())
    except ET.ParseError:
        urls += re.findall(r"<image>\s*([^<]+?)\s*</image>", text)

    for m in re.findall(r"https?://[^\s<>\"']+", text):
        if m.lower().split("?")[0].endswith(IMG_EXT):
            urls.append(m)
    return _unique([full for full in map(_normalize, urls) if full])


# 파일 앞부분 바이트로 실제 이미지인지 판별해 확장자를 돌려준다
def image_ext_from_bytes(data: bytes):
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    return None


# NCPMS OpenAPI 의 병 목록·상세 조회와 사진 내려받기를 맡는 클라이언트
class NcpmsClient:
    # API 키와 요청 간격을 받는다
    def __init__(self, api_key: str, sleep: float = REQ_SLEEP):
        self.api_key = api_key
        self.sleep = sleep

    # URL 을 요청해 응답 바이트를 돌려준다
    @staticmethod
    def get(url: str) -> bytes:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()

    # 감귤 작물의 병명으로 병 목록 조회 URL 을 만든다
    def list_url(self, sick_name: str) -> str:
        q = urllib.parse.urlencode({
            "apiKey": self.api_key, "serviceCode": SVC_LIST, "serviceType": SERVICE_TYPE,
            "cropName": CROP, "sickNameKor": sick_name,
            "displayCount": "100", "startPoint": "1",
        })
        return f"{BASE}?{q}"

    # sickKey 로 병 상세 조회 URL 을 만든다
    def detail_url(self, sick_key: str) -> str:
        q = urllib.parse.urlencode({
            "apiKey": self.api_key, "serviceCode": SVC_DETAIL,
            "serviceType": SERVICE_TYPE, "sickKey": sick_key,
        })
        return f"{BASE}?{q}"

    # 요청 후 서버 부담을 줄이도록 잠시 쉰다
    def get_polite(self, url: str) -> bytes:
        data = self.get(url)
        time.sleep(self.sleep)
        return data

    # 병 하나의 사진을 중복 없이 내려받고 저장 수를 돌려준다
    def collect_disease(self, key: str, sick_name: str, out_dir: str) -> int:
        sick_keys = find_sick_keys(self.get_polite(self.list_url(sick_name)))
        print(f"  검색된 병 항목(sickKey): {len(sick_keys)}개")
        seen_hash, saved = set(), 0
        for sk in sick_keys:
            try:
                detail = self.get_polite(self.detail_url(sk))
            except Exception as e:
                print(f"    상세 조회 실패 {sk}: {e}")
                continue
            for img_url in extract_image_urls(detail):
                try:
                    data = self.get_polite(img_url)
                except Exception:
                    continue
                ext = image_ext_from_bytes(data)
                if ext is None or len(data) < 2000:
                    continue
                h = hashlib.md5(data).hexdigest()
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                with open(os.path.join(out_dir, f"{key}_{saved+1:03d}{ext}"), "wb") as f:
                    f.write(data)
                saved += 1
        return saved


# 궤양병 목록·상세 응답 원문 일부를 출력해 API 구조를 확인한다
def probe(client: NcpmsClient):
    print(f"[SVC01] {CROP} 궤양병 목록 조회 …")
    xml = client.get(client.list_url("궤양병"))
    print(xml.decode("utf-8", "ignore")[:1500])
    keys = find_sick_keys(xml)
    print(f"\n→ sickKey {len(keys)}개: {keys[:5]}")
    if keys:
        print(f"\n[SVC05] sickKey={keys[0]} 상세 조회 …")
        d = client.get(client.detail_url(keys[0]))
        print(d.decode("utf-8", "ignore")[:1500])
        imgs = extract_image_urls(d)
        print(f"\n→ 추출된 이미지 URL {len(imgs)}개:")
        for u in imgs[:10]:
            print("   ", u)


# 질병 7종의 NCPMS 사진을 내려받는다
def collect(client: NcpmsClient):
    os.makedirs(NCPMS_IMAGE_DIR, exist_ok=True)
    summary = {}
    for key, sick_name in CITRUS_DISEASES.items():
        out_dir = os.path.join(NCPMS_IMAGE_DIR, key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== [{key}] {CROP} {sick_name} ===")
        summary[key] = client.collect_disease(key, sick_name, out_dir)
        print(f"  → 저장 {summary[key]}장")

    print("\n" + "=" * 50)
    print("  수집 요약")
    print("=" * 50)
    for key, n in summary.items():
        print(f"  {CITRUS_DISEASES[key]:14s} ({key:12s}) {n:3d}장")
    print(f"  {'합계':14s}              {sum(summary.values()):3d}장")
    print(f"\n  저장 위치: {NCPMS_IMAGE_DIR}")
    print("  학습 사용 전 각 이미지 라벨을 눈으로 확인하세요(도감엔 유사병 참고컷이 섞일 수 있음).")


# --probe 면 API 구조만 확인하고 아니면 전체 수집을 실행한다
def main():
    if not NCPMS_API_KEY:
        print(KEY_HELP)
        return
    client = NcpmsClient(NCPMS_API_KEY)
    if "--probe" in sys.argv:
        probe(client)
    else:
        collect(client)


if __name__ == "__main__":
    main()
