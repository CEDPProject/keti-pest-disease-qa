"""P1-2: 조건별 환각률. A=파인튜닝 VLM 자유생성, B=RAG build_answer(KB원문).
학명(라틴 이명) 기준 fabrication/misattribution/correct 채점. 약제는 보조.
결과: research/results/p1_2_hallucination.json
"""
import os,sys,re,json
os.environ.setdefault("HF_HOME","/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM,AutoProcessor
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); CORE=os.path.dirname(os.path.dirname(HERE)); REPO=os.path.dirname(CORE)
sys.path.insert(0,os.path.join(os.path.dirname(CORE),"backend","common")); sys.path.insert(0,os.path.join(os.path.dirname(CORE),"backend","rag"))
from classes import CLASSES
from label_extract import LABEL_TO_KEY
from kb_retriever import KBRetriever
BASE="Qwen/Qwen3.5-4B"; ADAPTER=os.path.join(REPO,"models","qwen3_5_balanced_lora","checkpoint-932")
TEST=os.path.join(REPO,"data","vlm_finetune_balanced","test.jsonl")
OUT=os.path.join(CORE,"results","p1_2_hallucination.json")
KO={v:k for k,v in LABEL_TO_KEY.items()}; KO["normal"]="정상"
K2SEC={m["key"]:m["kb_section"] for m in CLASSES.values()}
QS=[("학명","이 병의 학명은 무엇인가요?"),("약제","이 병에는 어떤 약제를 사용하나요?"),
    ("설명","이 병에 대해 자세히 설명해주세요.")]
LATIN=re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})")
def latin_set(text): return set(f"{m.group(1)} {m.group(2)}" for m in LATIN.finditer(text))

@torch.no_grad()
def vlm(model,proc,img,q):
    msgs=[{"role":"user","content":[{"type":"image","image":img,"max_pixels":768*768},{"type":"text","text":q}]}]
    try: t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    im=Image.open(img).convert("RGB"); inp=proc(text=[t],images=[im],return_tensors="pt").to("cuda")
    out=model.generate(**inp,max_new_tokens=200,do_sample=False)
    return proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()

def main():
    test=[json.loads(l) for l in open(TEST,encoding="utf-8")]
    kb=KBRetriever(device="cpu")
    # KB 학명 universe
    sci={k:kb.kb[K2SEC[k]]["scientific"] for k in K2SEC}      # key->학명원문
    sci_latin={k:latin_set(sci[k]) for k in sci}
    all_latin=set().union(*sci_latin.values())
    base=AutoModelForMultimodalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="cuda",trust_remote_code=True)
    proc=AutoProcessor.from_pretrained(BASE,trust_remote_code=True)
    model=PeftModel.from_pretrained(base,ADAPTER).eval()

    def classify(text,gold_key):
        found=latin_set(text)
        if not found: return "none",[]
        gold=sci_latin.get(gold_key,set())
        types=[]
        for name in found:
            if name in gold: types.append("correct")
            elif any(name in sci_latin[k] for k in sci_latin): types.append("misattribution")
            else: types.append("fabrication")
        # 우선순위: fabrication > misattribution > correct
        for lv in ("fabrication","misattribution","correct"):
            if lv in types: return lv, sorted(found)
        return "none",[]

    rowsA=[]; rowsB=[]
    for i,r in enumerate(test):
        gold=r["key"]; img=r["image"]
        if gold=="normal": continue         # 정상은 학명 없음 → 제외
        for tag,q in QS:
            a=vlm(model,proc,img,q); ca,fa=classify(a,gold)
            rowsA.append({"image":os.path.basename(img),"gold":KO[gold],"q":tag,"answer":a[:250],"latin":fa,"halluc":ca})
            b=kb.build_answer(K2SEC[gold],q); cb,fb=classify(b,gold)
            rowsB.append({"image":os.path.basename(img),"gold":KO[gold],"q":tag,"latin":fb,"halluc":cb})
        if (i+1)%10==0: print(f"  {i+1}/{len(test)}")

    def agg(rows):
        n=len(rows); fab=sum(1 for r in rows if r["halluc"]=="fabrication")
        mis=sum(1 for r in rows if r["halluc"]=="misattribution")
        cor=sum(1 for r in rows if r["halluc"]=="correct"); non=sum(1 for r in rows if r["halluc"]=="none")
        halluc=fab+mis
        return {"n":n,"fabrication":fab,"misattribution":mis,"correct":cor,"none":non,
                "halluc_rate":round(halluc/n,4),"halluc_n":halluc}
    res={"condition_A_noRAG":agg(rowsA),"condition_B_RAG":agg(rowsB),
         "note":"학명(라틴 이명) 기준. false-certainty는 미확정 학명 클래스 부재(검은무늬 제거)로 N/A. 정상 제외.",
         "A_halluc_examples":[r for r in rowsA if r["halluc"] in("fabrication","misattribution")][:6],
         "B_halluc_examples":[r for r in rowsB if r["halluc"] in("fabrication","misattribution")]}
    json.dump({"summary":res,"rowsA":rowsA,"rowsB":rowsB},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    A=res["condition_A_noRAG"]; B=res["condition_B_RAG"]
    print("="*56); print("  P1-2 환각률 (학명 기준)"); print("="*56)
    print(f"  조건A(RAG없음): 환각 {A['halluc_n']}/{A['n']} = {A['halluc_rate']*100:.1f}% (fab {A['fabrication']} + mis {A['misattribution']}, 정답 {A['correct']}, 미언급 {A['none']})")
    print(f"  조건B(RAG)    : 환각 {B['halluc_n']}/{B['n']} = {B['halluc_rate']*100:.1f}% {'구조상 0 확인' if B['halluc_n']==0 else '0아님 → 파이프라인 결함'}")
    print(f"  저장:{OUT}")

if __name__=="__main__": main()
