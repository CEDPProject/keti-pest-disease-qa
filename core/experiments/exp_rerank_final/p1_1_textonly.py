"""P1-1: 텍스트-only 병명 정확도 (ablation '텍스트만' 칸).
입력=베이스 캡션(병명 토큰 제거). (a) e5 최근접  (b) Gemma 선택.
결과: core/results/p1_1_textonly.json
"""
import os,sys,re,json,urllib.request
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); CORE=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,os.path.join(CORE,"common")); sys.path.insert(0,os.path.join(CORE,"rag"))
os.environ.setdefault("HF_HOME","/opt/project_data/users/jhlee/hf_cache")
from sklearn.metrics import f1_score,accuracy_score
from classes import CLASSES
from label_extract import extract_label,LABEL_TO_KEY
from kb_parser import parse_kb
from kb_retriever import KBRetriever
CAPS=os.path.join(CORE,"results","fusion_captions.json"); S1=os.path.join(CORE,"results","b_stage1_top3.json")
OUT=os.path.join(CORE,"results","p1_1_textonly.json")
KO={v:k for k,v in LABEL_TO_KEY.items()}; KO["normal"]="정상"
KEYS=["canker","scab","gray_mold","gummosis","black_spot","green_mold","anthracnose","normal"]
K2SEC={m["key"]:m["kb_section"] for m in CLASSES.values()}
NORMAL_PASS="병징에 해당하는 이상 소견(반점 병반 변색 곰팡이 무름)이 없는 건강한 상태."
DWORDS=list(KO.values())+["궤양","더뎅이","잿빛","곰팡이","수지","검은점","탄저","멜라노스","푸른곰팡이"]
def clean(c):
    for w in DWORDS: c=re.sub(r"\S*"+re.escape(w)+r"\S*","",c)
    return re.sub(r"\s+"," ",c).strip()
def src(img,k): return ("잎" if "잎" in img else "열매") if k=="normal" else ("NCPMS" if img.startswith("ncpms_") else "원본")
def metrics(g,p):
    acc=accuracy_score(g,p); f1=f1_score(g,p,labels=KEYS,average="macro",zero_division=0)
    sd={}
    for s in ("원본","NCPMS"):
        idx=[i for i in range(len(g)) if srcs[i]==s and g[i]!="normal"]
        sd[s]=round(sum(p[i]==g[i] for i in idx)/len(idx),3) if idx else None
    return {"acc":round(acc,4),"macro_f1":round(f1,4),"src":sd}
def gemma(cap):
    names="、".join(KO[k] for k in KEYS)
    pr=(f"다음은 감귤 사진의 시각적 특징 서술이다. 아래 병명 목록 중 서술과 가장 부합하는 병명 하나만 답하라.\n"
        f"[병명 목록] {names}\n[서술] {cap}\n답(병명 하나):")
    req=urllib.request.Request("http://localhost:11434/api/generate",
        data=json.dumps({"model":"gemma3:12b","prompt":pr,"stream":False,
        "options":{"temperature":0,"seed":42,"num_predict":20}}).encode(),headers={"Content-Type":"application/json"})
    try: r=json.loads(urllib.request.urlopen(req,timeout=120).read())["response"].strip()
    except Exception as e: return "unknown"
    return extract_label(r)
caps=json.load(open(CAPS,encoding="utf-8")); s1={r["image"]:r for r in json.load(open(S1,encoding="utf-8"))["rows"]}
kb=parse_kb(); kbret=KBRetriever(device="cpu")
pass_txt={k:(NORMAL_PASS if k=="normal" else " ".join(kb[K2SEC[k]]["symptoms"])) for k in KEYS}
pemb={k:kbret.model.encode([f"passage: {pass_txt[k]}"],normalize_embeddings=True)[0] for k in KEYS}
golds=[]; srcs=[]; pred_e5=[]; pred_llm=[]
for b,c in caps.items():
    g=s1[b]["gold"]; golds.append(g); srcs.append(src(b,g)); cap=clean(c["caption"])
    ce=kbret.model.encode([f"query: {cap}"],normalize_embeddings=True)[0]
    pred_e5.append(max(KEYS,key=lambda k:float(ce@pemb[k])))
    pred_llm.append(gemma(cap))
res={"a_e5":metrics(golds,pred_e5),"b_gemma":metrics(golds,pred_llm),"n":len(golds)}
res["adopted_textonly"]=res["a_e5"] if res["a_e5"]["macro_f1"]>=res["b_gemma"]["macro_f1"] else res["b_gemma"]
json.dump(res,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("="*54); print("  P1-1 텍스트-only (캡션→병명)"); print("="*54)
print(f"  (a) e5 최근접 : acc={res['a_e5']['acc']} F1={res['a_e5']['macro_f1']} 원본{res['a_e5']['src']['원본']} NCPMS{res['a_e5']['src']['NCPMS']}")
print(f"  (b) Gemma선택 : acc={res['b_gemma']['acc']} F1={res['b_gemma']['macro_f1']} 원본{res['b_gemma']['src']['원본']} NCPMS{res['b_gemma']['src']['NCPMS']}")
print(f"  → 채택(높은쪽): F1={res['adopted_textonly']['macro_f1']}")
print(f"  저장:{OUT}")
