import os, json, csv
import sys
from sklearn.metrics import f1_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import RESULTS_DIR, PAPER_DOCS_DIR  # noqa: E402
R = RESULTS_DIR; REF = os.path.join(PAPER_DOCS_DIR, "_4b_reference", "results_json")
OUT_DIR = os.path.join(PAPER_DOCS_DIR, "tables")
os.makedirs(OUT_DIR, exist_ok=True)
KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot", "green_mold", "anthracnose", "normal"]
KO = {"canker": "궤양병", "scab": "더뎅이병", "gray_mold": "잿빛곰팡이병", "gummosis": "수지병",
      "black_spot": "검은점무늬병", "green_mold": "녹색곰팡이병", "anthracnose": "탄저병", "normal": "정상"}


def J(path): return json.load(open(path, encoding="utf-8"))


curve9 = J(f"{R}/balanced_curve_9b.json"); best9 = curve9["best"]
abl9 = J(f"{R}/ablation_9b.json")
top3_9 = J(f"{R}/b_stage1_top3_9b.json")["summary"]
sel9 = J(f"{R}/exp3_confidence_9b.json")
hal9 = J(f"{R}/p1_2_hallucination_9b.json")["summary"]
pred9 = J(f"{R}/pred_balanced_9b_ep6.json")
f1_9 = dict(zip(KEYS, f1_score([r["key"] for r in pred9], [r["pred_key"] for r in pred9], labels=KEYS, average=None, zero_division=0)))
# 4B 참고
curve4 = J(f"{REF}/balanced_curve.json"); best4 = curve4["best"]
sel4 = J(f"{REF}/exp3_confidence.json"); top3_4 = J(f"{REF}/b_stage1_top3.json")["summary"]
hal4 = J(f"{REF}/p1_2_hallucination.json")["summary"]

def w_csv(name, header, rows):
    with open(f"{OUT_DIR}/{name}", "w", newline="") as f:
        c = csv.writer(f); c.writerow(header); [c.writerow(r) for r in rows]

# ── 표1a: 진단 ablation ──
zs, ft = abl9["zero_shot"], abl9["fine_tuned"]
cond = [("a_text_only", "text-only (caption→name)"), ("b_image_only", "image-only (min. prompt)"), ("c_image_text", "image+text (fixed Q)")]
t1a = [[lab, zs[k]["acc"], zs[k]["macro_f1"], ft[k]["acc"], ft[k]["macro_f1"], ft[k]["src"]["원본"], ft[k]["src"]["NCPMS"]] for k, lab in cond]
w_csv("table1a_ablation_9b.csv", ["condition", "zs_acc", "zs_F1", "ft_acc", "ft_F1", "ft_원본", "ft_NCPMS"], t1a)

# ── 표1b: 사실성(환각) ──
A, B = hal9["condition_A_noRAG"], hal9["condition_B_RAG"]
t1b = [["A: direct generation (9B)", A["n"], f"{A['halluc_rate']*100:.1f}%", A["fabrication"], A["misattribution"], A["correct"], A["none"]],
       ["B: +RAG (KB verbatim)", B["n"], f"{B['halluc_rate']*100:.1f}%", B["fabrication"], B["misattribution"], B["correct"], B["none"]]]
w_csv("table1b_factuality_9b.csv", ["condition", "n_queries", "halluc_rate", "fabrication", "misattribution", "correct", "none"], t1b)

# ── 표2: 층화 + 클래스별 F1 ──
t2 = [["overall", best9["macro_f1"], best9["accuracy"]],
      ["source:원본(AI Hub)", "-", best9["src_acc"]["원본"]],
      ["source:NCPMS", "-", best9["src_acc"]["NCPMS"]]]
t2 += [[f"class:{KO[k]}", round(float(f1_9[k]), 3), "-"] for k in KEYS]
w_csv("table2_stratified_9b.csv", ["scope/class", "macro_F1_or_F1", "accuracy"], t2)

# ── 표3: selective ──
t3 = [[s["threshold"], s["trigger_rate"], s["wrong_coverage"], s["gold_in_top3_among_trigger"]] for s in sel9["sweep"]]
w_csv("table3_selective_9b.csv", ["margin_threshold", "trigger_rate", "wrong_coverage", "gold_in_top3"], t3)

# ── M7: 스케일 비교 4B vs 9B ──
def g(curve, sel, top3, hal, key):
    b = curve["best"]
    return {"epoch": b["epoch"], "F1": b["macro_f1"], "acc": b["accuracy"],
            "원본": b["src_acc"]["원본"], "NCPMS": b["src_acc"]["NCPMS"],
            "AURC": sel["aurc"], "rej30_acc": [r["acc"] for r in sel["reject_policy"] if r["reject_pct"] == 0.3][0],
            "top3recall": top3["top3_recall"], "margin_c": top3["margin_correct"]["median"], "margin_w": top3["margin_wrong"]["median"],
            "hallA": hal["condition_A_noRAG"]["halluc_rate"], "hallB": hal["condition_B_RAG"]["halluc_rate"]}
m4 = g(curve4, sel4, top3_4, hal4, "4B"); m9 = g(curve9, sel9, top3_9, hal9, "9B")
metrics = [("최적 epoch", "epoch"), ("macro-F1", "F1"), ("accuracy", "acc"), ("원본 acc", "원본"), ("NCPMS acc", "NCPMS"),
           ("환각 A(직접)", "hallA"), ("환각 B(+RAG)", "hallB"), ("AURC", "AURC"), ("30%보류 acc", "rej30_acc"),
           ("top-3 recall", "top3recall"), ("마진 정답med", "margin_c"), ("마진 오답med", "margin_w")]
t7 = []
for lab, k in metrics:
    v4, v9 = m4[k], m9[k]
    if k in ("hallA", "hallB"):
        v4, v9 = f"{v4*100:.1f}%", f"{v9*100:.1f}%"
    t7.append([lab, v4, v9])
w_csv("table7_scale_4b_vs_9b.csv", ["metric", "Qwen3.5-4B", "Qwen3.5-9B"], t7)

# ── 통합 md ──
def mdtable(header, rows):
    s = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows: s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s

md = f"""# 논문 표 (ICTC2026) — Qwen3.5-9B 메인. 모든 수치 측정값, 출처 명시.

> 근거: `qa_docs/experiment_results/*_9b*.json` · 4B는 `_4b_reference/`. 9B 최적 ep{best9['epoch']}.

## 표1a. 진단 Ablation — 입력 모달리티 × [zero-shot / fine-tuned] (9B)
{mdtable(["입력 조건", "ZS acc", "ZS F1", "FT acc", "FT F1", "FT 원본", "FT NCPMS"], t1a)}
> zero-shot은 이미지 조건 거의 0 (base가 8클래스 병명 미출력). 파인튜닝으로 image+text F1 **{ft['c_image_text']['macro_f1']}**(메인). image-only({ft['b_image_only']['macro_f1']})→+고정질문 image+text({ft['c_image_text']['macro_f1']}) 소폭 상승.

## 표1b. 사실성 — 생성 vs KB조회 환각률 (9B, 147 factual queries)
{mdtable(["조건", "n", "환각률", "fab", "mis", "correct", "none"], t1b)}
> 조건 B(+RAG)는 학명·약제를 **KB 원문 조립** → 환각 **구조적 0%**. 영명 오탐 재채점 반영(`halluc_9b_rescore.py`).

## 표2. 층화 평가 + 클래스별 F1 (9B ep{best9['epoch']})
{mdtable(["범위/클래스", "macro-F1 / F1", "acc"], t2)}
> 잿빛곰팡이(0.59)만 하락, 나머지 향상(탄저 1.00·검은점 0.92·정상 0.93). 원본 0.913 = 4B와 동일(포화), NCPMS 0.731(+7.7pt).

## 표3. Selective prediction (9B, 마진 기반)
{mdtable(["마진 임계", "발동률", "오답커버", "후보내정답"], t3)}
> AURC = **{sel9['aurc']}** · 보류 30% → acc **{[r['acc'] for r in sel9['reject_policy'] if r['reject_pct']==0.3][0]}** (baseline {sel9['baseline_acc']}).

## 표4. top-3 / 마진 (9B)
{mdtable(["지표", "값"], [["score top-1 acc", top3_9["score_top1_acc"]], ["생성 top-1 acc", top3_9["gen_top1_acc"]], ["score=생성 일치율", top3_9["score_vs_gen_agree"]], ["top-3 recall", f"{top3_9['top3_recall']} (원본{top3_9['top3_recall_by_source']['원본']}/NCPMS{top3_9['top3_recall_by_source']['NCPMS']})"], ["마진 정답 median", top3_9["margin_correct"]["median"]], ["마진 오답 median", top3_9["margin_wrong"]["median"]]])}

## 표7 (M7). 스케일 비교 — Qwen3.5-4B vs 9B (4B는 기존 수치 재사용)
{mdtable(["지표", "Qwen3.5-4B", "Qwen3.5-9B"], t7)}
> 9B가 전 지표 상회: F1 +5.4pt, 교차출처 +7.7pt, AURC 0.084→0.062, 30%보류 0.85→0.90. In-source(원본 0.913)는 동일(데이터 천장).
"""
open(f"{OUT_DIR}/tables_9b.md", "w", encoding="utf-8").write(md)
print("생성: tables_9b.md + csv 6종")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith("_9b.csv"): print("   ", f)
