import os
import re
import sys
import time
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


BASE = "http://ncpms.rda.go.kr/npmsAPI/service"
CROP = "감귤"                    # 대상작물 고정 → 종 차이(오렌지 등) 혼입 방지
SVC_LIST = "SVC01"              # 병 목록 검색 (sickKey 획득)
SVC_DETAIL = "SVC05"           # 병 상세 (이미지 포함)
SERVICE_TYPE = "AA001"         # XML

# 클래스 키(우리 classes.py와 동일) → NCPMS 검색 병명
CITRUS_DISEASES = {
    "canker":      "궤양병",
    "scab":        "더뎅이병",
    "gray_mold":   "잿빛곰팡이병",
    "gummosis":    "수지병",
    "black_spot":  "검은점무늬병",
    "green_mold":  "녹색곰팡이병",
    "anthracnose": "탄저병",
}

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from settings import NCPMS_API_KEY as API_KEY  # noqa: E402  (.env 의 NCPMS_API_KEY)
OUT_ROOT = os.path.join(REPO, "data", "raw", "ncpms_수집")

IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
REQ_SLEEP = 0.7                # 서버 예의상 요청 간 간격(초)
HEADERS = {"User-Agent": "Mozilla/5.0 (research; citrus-pest-collect)"}


def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _list_url(sick_name):
    q = urllib.parse.urlencode({
        "apiKey": API_KEY, "serviceCode": SVC_LIST, "serviceType": SERVICE_TYPE,
        "cropName": CROP, "sickNameKor": sick_name,
        "displayCount": "100", "startPoint": "1",
    })
    return f"{BASE}?{q}"


def _detail_url(sick_key):
    q = urllib.parse.urlencode({
        "apiKey": API_KEY, "serviceCode": SVC_DETAIL,
        "serviceType": SERVICE_TYPE, "sickKey": sick_key,
    })
    return f"{BASE}?{q}"


def _find_sick_keys(xml_bytes):
    """SVC01 응답 XML에서 sickKey 목록 추출(태그명 변형에 견고)."""
    keys = []
    try:
        root = ET.fromstring(xml_bytes)
        for el in root.iter():
            if el.tag.lower().endswith("sickkey") and el.text and el.text.strip():
                keys.append(el.text.strip())
    except ET.ParseError:
        keys = re.findall(r"<sickKey>\s*([^<]+?)\s*</sickKey>", xml_bytes.decode("utf-8", "ignore"))
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


def _localname(tag):
    return tag.split("}")[-1].lower()


def _normalize(u):
    u = u.strip()
    if not u:
        return None
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        return "http://ncpms.rda.go.kr" + u
    return "http://ncpms.rda.go.kr/" + u


def _extract_image_urls(xml_bytes):
    """상세 XML에서 이미지 URL 추출.
    명세: imageList/image (0~N개) = 병피해 사진의 NCPMS URL 링크.
    URL이 .jpg 로 안 끝나는 뷰어 링크일 수 있으므로 태그 기반으로 우선 수집한다."""
    text = xml_bytes.decode("utf-8", "ignore")
    urls = []

    try:
        root = ET.fromstring(xml_bytes)
        # 1) 명세대로 imageList 하위의 image 태그
        for il in root.iter():
            if _localname(il.tag) == "imagelist":
                for ch in il.iter():
                    if _localname(ch.tag) == "image" and (ch.text or "").strip():
                        urls.append(ch.text.strip())
        # 2) imageList 없이 image 태그만 있는 경우
        if not urls:
            for el in root.iter():
                if _localname(el.tag) == "image" and (el.text or "").strip():
                    urls.append(el.text.strip())
    except ET.ParseError:
        urls += re.findall(r"<image>\s*([^<]+?)\s*</image>", text)

    # 3) 확장자로 끝나는 절대 URL(백업)
    for m in re.findall(r"https?://[^\s<>\"']+", text):
        if m.lower().split("?")[0].endswith(IMG_EXT):
            urls.append(m)

    out, seen = [], set()
    for u in urls:
        full = _normalize(u)
        if full and full not in seen:
            seen.add(full); out.append(full)
    return out


def _img_ext_from_bytes(data):
    """매직바이트로 실제 이미지 여부·확장자 판별(HTML/에러페이지 저장 방지)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    return None


def probe():
    """구조 확인용: 감귤 궤양병의 목록/상세 원문을 일부 출력."""
    if not API_KEY:
        print("먼저 환경변수 NCPMS_API_KEY 를 설정하세요 (https://ncpms.rda.go.kr/npms/OpenApiInfo.np)")
        return
    print(f"[SVC01] {CROP} 궤양병 목록 조회 …")
    xml = _get(_list_url("궤양병"))
    print(xml.decode("utf-8", "ignore")[:1500])
    keys = _find_sick_keys(xml)
    print(f"\n→ sickKey {len(keys)}개: {keys[:5]}")
    if keys:
        print(f"\n[SVC05] sickKey={keys[0]} 상세 조회 …")
        d = _get(_detail_url(keys[0]))
        print(d.decode("utf-8", "ignore")[:1500])
        imgs = _extract_image_urls(d)
        print(f"\n→ 추출된 이미지 URL {len(imgs)}개:")
        for u in imgs[:10]:
            print("   ", u)


def collect():
    if not API_KEY:
        print("먼저 환경변수 NCPMS_API_KEY 를 설정하세요 (https://ncpms.rda.go.kr/npms/OpenApiInfo.np)")
        return
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = {}
    for key, sick_name in CITRUS_DISEASES.items():
        out_dir = os.path.join(OUT_ROOT, key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== [{key}] {CROP} {sick_name} ===")

        xml = _get(_list_url(sick_name)); time.sleep(REQ_SLEEP)
        sick_keys = _find_sick_keys(xml)
        print(f"  검색된 병 항목(sickKey): {len(sick_keys)}개")

        seen_hash = set()
        saved = 0
        for sk in sick_keys:
            try:
                detail = _get(_detail_url(sk)); time.sleep(REQ_SLEEP)
            except Exception as e:
                print(f"    상세 조회 실패 {sk}: {e}"); continue
            for img_url in _extract_image_urls(detail):
                try:
                    data = _get(img_url); time.sleep(REQ_SLEEP)
                except Exception:
                    continue
                ext = _img_ext_from_bytes(data)   # 실제 이미지인지 매직바이트로 확인
                if ext is None or len(data) < 2000:   # HTML/에러페이지/깨진 파일 제외
                    continue
                h = hashlib.md5(data).hexdigest()
                if h in seen_hash:          # 중복 제거
                    continue
                seen_hash.add(h)
                fp = os.path.join(out_dir, f"{key}_{saved+1:03d}{ext}")
                with open(fp, "wb") as f:
                    f.write(data)
                saved += 1
        summary[key] = saved
        print(f"  → 저장 {saved}장")

    print("\n" + "=" * 50)
    print("  수집 요약")
    print("=" * 50)
    total = 0
    for key, n in summary.items():
        total += n
        print(f"  {CITRUS_DISEASES[key]:14s} ({key:12s}) {n:3d}장")
    print(f"  {'합계':14s}              {total:3d}장")
    print(f"\n  저장 위치: {OUT_ROOT}")
    print("  학습 사용 전 각 이미지 라벨을 눈으로 확인하세요(도감엔 유사병 참고컷이 섞일 수 있음).")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        collect()
