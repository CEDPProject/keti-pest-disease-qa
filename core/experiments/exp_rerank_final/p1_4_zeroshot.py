"""이미지만(zero-shot): 베이스 Qwen3.5-4B(어댑터 없음)를 신 test 57장 평가. ablation 빈칸."""
import os,sys,json
os.environ.setdefault("HF_HOME","/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM,AutoProcessor
HERE=os.path.dirname(os.path.abspath(__file__)); CORE=os.path.dirname(os.path.dirname(HERE)); REPO=os.path.dirname(CORE)
sys.path.insert(0,os.path.join(CORE,"common"))
from label_extract import extract_label
from sklearn.metrics import f1_score,accuracy_score
BASE="Qwen/Qwen3.5-4B"; TEST=os.path.join(REPO,"data","vlm_finetune_balanced","test.jsonl")
OUT=os.path.join(CORE,"results","p1_4_zeroshot.json")
NP="이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
KEYS=["canker","scab","gray_mold","gummosis","black_spot","green_mold","anthracnose","normal"]
@torch.no_grad()
def infer(m,p,img):
    ms=[{"role":"user","content":[{"type":"image","image":img,"max_pixels":768*768},{"type":"text","text":NP}]}]
    try:t=p.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:t=p.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
    im=Image.open(img).convert("RGB");inp=p(text=[t],images=[im],return_tensors="pt").to("cuda")
    return p.batch_decode(m.generate(**inp,max_new_tokens=32,do_sample=False)[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
test=[json.loads(l) for l in open(TEST,encoding="utf-8")]
m=AutoModelForMultimodalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="cuda",trust_remote_code=True).eval()
pr=AutoProcessor.from_pretrained(BASE,trust_remote_code=True)
g=[];p=[]
for i,r in enumerate(test):
    a=infer(m,pr,r["image"]);g.append(r["key"]);p.append(extract_label(a))
    if(i+1)%15==0:print(f"  {i+1}/{len(test)}")
acc=accuracy_score(g,p);f1=f1_score(g,p,labels=KEYS,average="macro",zero_division=0)
json.dump({"zero_shot":{"n":len(g),"acc":round(acc,4),"macro_f1":round(f1,4)}},open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(f"  zero-shot(베이스 3.5-4B) 신test: acc={acc:.4f} F1={f1:.4f}")
