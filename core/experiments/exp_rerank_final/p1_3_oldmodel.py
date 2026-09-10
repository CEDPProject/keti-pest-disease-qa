"""P1-3: 구 3.5-4B(0.890 냈던 어댑터)를 새 test 병해 49장에 동일 프로토콜 평가.
목적: 0.890 vs 0.783은 rerank 아니라 평가셋 차이임을 표로. 둘 다 VLM top-1(no rerank).
결과: core/results/p1_3_oldmodel.json
"""
import os,sys,re,json
os.environ.setdefault("HF_HOME","/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM,AutoProcessor
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); CORE=os.path.dirname(os.path.dirname(HERE)); REPO=os.path.dirname(CORE)
sys.path.insert(0,os.path.join(CORE,"common"))
from label_extract import extract_label
from sklearn.metrics import f1_score,accuracy_score
BASE="Qwen/Qwen3.5-4B"; OLD=os.path.join(REPO,"models","qwen3_5vl_lora","checkpoint-192")   # 구 ep4
TEST=os.path.join(REPO,"data","vlm_finetune_balanced","test.jsonl")
OUT=os.path.join(CORE,"results","p1_3_oldmodel.json")
NAME_PROMPT="이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
KEYS=["canker","scab","gray_mold","gummosis","black_spot","green_mold","anthracnose"]
@torch.no_grad()
def infer(model,proc,img):
    msgs=[{"role":"user","content":[{"type":"image","image":img,"max_pixels":768*768},{"type":"text","text":NAME_PROMPT}]}]
    try: t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    im=Image.open(img).convert("RGB"); inp=proc(text=[t],images=[im],return_tensors="pt").to("cuda")
    out=model.generate(**inp,max_new_tokens=32,do_sample=False)
    return proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
def src(img): return "NCPMS" if img.startswith("ncpms_") else "원본"
def main():
    test=[json.loads(l) for l in open(TEST,encoding="utf-8") if json.loads(l)["key"]!="normal"]
    base=AutoModelForMultimodalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="cuda",trust_remote_code=True)
    proc=AutoProcessor.from_pretrained(BASE,trust_remote_code=True)
    model=PeftModel.from_pretrained(base,OLD).eval()
    g=[]; p=[]; rows=[]
    for i,r in enumerate(test):
        ans=infer(model,proc,r["image"]); pk=extract_label(ans)
        g.append(r["key"]); p.append(pk)
        rows.append({"image":os.path.basename(r["image"]),"src":src(os.path.basename(r["image"])),"gold":r["key"],"pred":pk,"ans":ans[:40]})
        if (i+1)%10==0: print(f"  {i+1}/{len(test)}")
    acc=accuracy_score(g,p); f1=f1_score(g,p,labels=KEYS,average="macro",zero_division=0)
    def sacc(s):
        idx=[i for i in range(len(g)) if rows[i]["src"]==s]
        return round(sum(p[i]==g[i] for i in idx)/len(idx),3) if idx else None
    res={"old_model_on_new_disease_test":{"n":len(g),"acc":round(acc,4),"macro_f1":round(f1,4),
         "src":{"원본":sacc("원본"),"NCPMS":sacc("NCPMS")}},
         "ref_old_on_old_test":{"acc":0.88,"macro_f1":0.890,"n":25,"src":"nameprompt_qwen35_curve.json ep4"},
         "ref_new_on_new_disease":{"acc":0.776,"macro_f1":0.783,"n":49,"src":"pred_balanced_ep4.json disease subset"},
         "confounds":"구모델=구데이터(신 NCPMS 미학습)·구클래스(검은무늬 포함/정상 없음). 백본은 동일 3.5-4B. 둘 다 VLM top-1(no rerank)."}
    json.dump({"summary":res,"rows":rows},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    o=res["old_model_on_new_disease_test"]
    print("="*56); print("  P1-3 구모델 재평가"); print("="*56)
    print(f"  구 3.5 on 구test(25): acc 0.88 F1 0.890")
    print(f"  구 3.5 on 신test병해(49): acc {o['acc']} F1 {o['macro_f1']} (원본{o['src']['원본']} NCPMS{o['src']['NCPMS']})")
    print(f"  신 균형 on 신test병해(49): acc 0.776 F1 0.783")
    print(f"  저장:{OUT}")
if __name__=="__main__": main()
