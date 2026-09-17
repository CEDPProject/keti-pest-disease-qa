import os
import re
import sys
import json
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))       # core/
sys.path.insert(0, os.path.join(os.path.dirname(CORE), "backend", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(CORE), "backend", "rag"))
os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
from sklearn.metrics import f1_score, accuracy_score
from classes import CLASSES  # noqa
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from kb_retriever import KBRetriever  # noqa
CAPS = os.path.join(CORE, "results", "fusion_captions.json")
STAGE1 = os.path.join(CORE, "results", "b_stage1_top3.json")
OUT = os.path.join(CORE, "results", "exp1_llm_reranker.json")

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"
KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
KEY_TO_SECTION = {m["key"]: m["kb_section"] for lab, m in CLASSES.items()}
VISUAL_QUERY = ("병반 반점 딱지 부스럼 융기 돌출 코르크 분화구 검은점 "
                "부패 곰팡이 균사 포자 색깔 표면 모양 크기 테두리")
NORMAL_RULE = "병징에 해당하는 이상 소견(반점·병반·변색·곰팡이 등)이 없으면 정상."


def build_guides(kb):
    g = {}
    for k in CLASS_KEYS:
        if k == "normal":
            g[k] = NORMAL_RULE
        else:
            lines = kb.retrieve_lines(KEY_TO_SECTION[k], VISUAL_QUERY, "병징", top_n=3)
            g[k] = " ".join(l["text"] for l in lines)
    return g


def ask(caption, order_keys, guides):
    blocks = "\n".join(f"[{KO[k]}] {guides[k]}" for k in order_keys)
    prompt = ("아래는 감귤 사진의 시각적 특징 서술이다. 이어서 후보 병해 3개의 병징 설명이 있다.\n"
              "서술과 가장 부합하는 후보 **하나만** 병명으로 답하라. 다른 설명 없이 병명만.\n\n"
              f"[관찰된 시각 특징]\n{caption}\n\n[후보]\n{blocks}\n\n답(병명 하나):")
    req = urllib.request.Request(OLLAMA, data=json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 20}}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())["response"].strip()
    except Exception as e:
        return None, str(e)[:40]
    return extract_label(r), r


def _source(img, key):
    if key == "normal":
        return "잎" if "잎" in img else ("열매" if "열매" in img else "기타")
    return "NCPMS" if img.startswith("ncpms_") else "원본"


def _metrics(golds, preds, srcs):
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)
    src = {}
    for s in ("원본", "NCPMS"):
        idx = [i for i in range(len(golds)) if srcs[i] == s and golds[i] != "normal"]
        src[s] = round(sum(preds[i] == golds[i] for i in idx) / len(idx), 3) if idx else None
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "src_acc": src}


def _trans(golds, base, final):
    w2c = sum(1 for i in range(len(golds)) if base[i] != golds[i] and final[i] == golds[i])
    c2w = sum(1 for i in range(len(golds)) if base[i] == golds[i] and final[i] != golds[i])
    return {"wrong2correct": w2c, "correct2wrong": c2w, "net": w2c - c2w}


def main():
    caps = json.load(open(CAPS, encoding="utf-8"))
    s1 = {r["image"]: r for r in json.load(open(STAGE1, encoding="utf-8"))["rows"]}
    kb = KBRetriever(device="cpu")
    guides = build_guides(kb)

    rows = []
    for i, (b, c) in enumerate(caps.items()):
        s = s1[b]; top1 = s["score_top1"]; top3 = s["score_top3"]
        asc = sorted(top3, key=lambda k: KO[k])
        desc = list(reversed(asc))
        k1, r1 = ask(c["caption"], asc, guides)
        k2, r2 = ask(c["caption"], desc, guides)
        v1 = k1 in top3; v2 = k2 in top3
        agree = (k1 == k2) and v1
        final = k1 if agree else top1                # 불일치→top-1 유지
        rows.append({"image": b, "gold": s["gold"], "source": _source(b, s["gold"]),
                     "top1": top1, "top3": top3, "margin": s["margin_1_2"],
                     "pred_asc": k1, "pred_desc": k2, "agree": bool(agree),
                     "valid": bool(v1 and v2), "final": final, "raw_asc": r1[:30]})
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(caps)}")

    golds = [r["gold"] for r in rows]; srcs = [r["source"] for r in rows]
    base = [r["top1"] for r in rows]
    agree_rate = sum(r["agree"] for r in rows) / len(rows)

    # (a) 전체: agree일 때만 채택(=final), 아니면 top1
    pa = [r["final"] for r in rows]
    # (b) 마진<0.9 인 케이스에만 감별 적용
    pb = [r["final"] if r["margin"] < 0.9 else r["top1"] for r in rows]

    res = {
        "model": MODEL, "n": len(rows),
        "position_agreement": round(agree_rate, 3),
        "baseline": _metrics(golds, base, srcs),
        "a_all": {**_metrics(golds, pa, srcs), **_trans(golds, base, pa)},
        "b_margin<0.9": {**_metrics(golds, pb, srcs), **_trans(golds, base, pb),
                         "n_applied": sum(1 for r in rows if r["margin"] < 0.9)},
    }
    json.dump({"summary": res, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  EXP-1 LLM-as-reranker ({MODEL})")
    print("=" * 60)
    print(f"  위치편향 일치율(가나다↔역순): {agree_rate:.3f} ({'양호' if agree_rate>=0.8 else '편향 심각'})")
    b = res["baseline"]; a = res["a_all"]; mb = res["b_margin<0.9"]
    print(f"  baseline    F1={b['macro_f1']:.3f} acc={b['accuracy']:.3f}")
    print(f"  (a) 전체    F1={a['macro_f1']:.3f} acc={a['accuracy']:.3f} | 오답→정답 {a['wrong2correct']} 정답→오답 {a['correct2wrong']} 순이득 {a['net']}")
    print(f"  (b) 마진<0.9 F1={mb['macro_f1']:.3f} acc={mb['accuracy']:.3f} | 적용 {mb['n_applied']} | 순이득 {mb['net']}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
