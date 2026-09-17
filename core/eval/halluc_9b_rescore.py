"""환각 재채점 — 영명(English name) 오인 제거.

저장된 답변/라틴에서 학명만 판정. 라틴 정규식이 'Scab diseases'/'Green mold' 등
영명을 학명으로 잡던 오검출을 제거한 뒤 재집계.
입력=core/results/p1_2_hallucination_9b.json (덮어씀).
"""
import os, sys, re, json
VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from classes import CLASSES
from label_extract import LABEL_TO_KEY
from kb_parser import parse_kb

KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
K2SEC = {m["key"]: m["kb_section"] for m in CLASSES.values()}
IN = os.path.join(VLM_FT, "results", "p1_2_hallucination_9b.json")
LATIN = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})")


def lset(t):
    return set(f"{m.group(1)} {m.group(2)}" for m in LATIN.finditer(t or ""))


kb = parse_kb()
sci_latin = {k: lset(kb[K2SEC[k]]["scientific"]) for k in K2SEC}   # 진짜 학명
eng_latin = set()                                                 # 영명(제외 대상)
for k in K2SEC:
    eng_latin |= lset(kb[K2SEC[k]]["english"])


def real_hakmyeong(found):
    return {n for n in found if n not in eng_latin}


def classify(found_all, gold_key):
    found = real_hakmyeong(found_all)
    if not found:
        return "none"
    gold = sci_latin.get(gold_key, set())
    types = []
    for n in found:
        if n in gold:
            types.append("correct")
        elif any(n in sci_latin[k] for k in sci_latin):
            types.append("misattribution")
        else:
            types.append("fabrication")
    for lv in ("fabrication", "misattribution", "correct"):
        if lv in types:
            return lv
    return "none"


d = json.load(open(IN, encoding="utf-8"))
KO2K = {v: k for k, v in KO.items()}
for key in ("rowsA", "rowsB"):
    for r in d[key]:
        gk = KO2K[r["gold"]]
        r["latin_all"] = r.get("latin", [])
        r["latin"] = sorted(real_hakmyeong(set(r["latin_all"])))
        r["halluc"] = classify(set(r["latin_all"]), gk)


def agg(rows):
    n = len(rows)
    fab = sum(r["halluc"] == "fabrication" for r in rows)
    mis = sum(r["halluc"] == "misattribution" for r in rows)
    cor = sum(r["halluc"] == "correct" for r in rows)
    non = sum(r["halluc"] == "none" for r in rows)
    return {"n": n, "fabrication": fab, "misattribution": mis, "correct": cor, "none": non,
            "halluc_n": fab + mis, "halluc_rate": round((fab + mis) / n, 4)}


d["summary"]["condition_A_noRAG"] = agg(d["rowsA"])
d["summary"]["condition_B_RAG"] = agg(d["rowsB"])
d["summary"]["A_halluc_examples"] = [{"image": r["image"], "gold": r["gold"], "q": r["q"], "halluc": r["halluc"],
                                      "latin": r["latin"], "answer": r.get("answer", "")[:180]}
                                     for r in d["rowsA"] if r["halluc"] in ("fabrication", "misattribution")][:8]
d["summary"]["B_halluc_examples"] = [r for r in d["rowsB"] if r["halluc"] in ("fabrication", "misattribution")]
d["summary"]["rescore_note"] = "영명(Scab diseases/Green mold 등) 오인 제거 후 재채점. 정규식이 영명을 학명으로 잡던 버그 수정(4B p1_2_rescore와 동일)."
json.dump(d, open(IN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
A = d["summary"]["condition_A_noRAG"]; B = d["summary"]["condition_B_RAG"]
print("=" * 56); print("  [M5 재채점] 9B 환각 (영명 제외, 학명만)")
print(f"  조건A(9B직접): 환각 {A['halluc_n']}/{A['n']} = {A['halluc_rate']*100:.1f}% (fab{A['fabrication']}+mis{A['misattribution']}, 정답{A['correct']}, 미언급{A['none']})")
print(f"  조건B(9B+RAG): 환각 {B['halluc_n']}/{B['n']} = {B['halluc_rate']*100:.1f}% {'구조상 0 확인' if B['halluc_n']==0 else '0아님 → 사례 첨부'}")
if B["halluc_n"] > 0:
    for r in d["summary"]["B_halluc_examples"]:
        print("   ", r)
