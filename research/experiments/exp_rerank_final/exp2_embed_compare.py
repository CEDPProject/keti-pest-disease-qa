import os, sys, json, itertools
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); CORE=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,os.path.join(os.path.dirname(CORE),"backend","common")); sys.path.insert(0,os.path.join(os.path.dirname(CORE),"backend","rag"))
os.environ.setdefault("HF_HOME","/opt/project_data/users/jhlee/hf_cache")
from sentence_transformers import SentenceTransformer
from classes import CLASSES
from label_extract import LABEL_TO_KEY
from kb_parser import parse_kb
from kb_retriever import KBRetriever

CAPS=os.path.join(CORE,"results","fusion_captions.json")
S1=os.path.join(CORE,"results","b_stage1_top3.json")
OUT=os.path.join(CORE,"results","exp2_embed_compare.json")
KO={v:k for k,v in LABEL_TO_KEY.items()}; KO["normal"]="정상"
DISEASE=[m["key"] for m in CLASSES.values()]           # 7병해
K2SEC={m["key"]:m["kb_section"] for m in CLASSES.values()}
VQ="병반 반점 딱지 부스럼 융기 돌출 코르크 분화구 검은점 부패 곰팡이 균사 포자 색깔 표면 모양 크기 테두리"
NORMAL_PASS="병징에 해당하는 이상 소견(반점 병반 변색 곰팡이 무름)이 없는 건강한 상태."

def softmax(x): x=np.array(x); e=np.exp(x-x.max()); return e/e.sum()

def main():
    caps=json.load(open(CAPS,encoding="utf-8"))
    s1={r["image"]:r for r in json.load(open(S1,encoding="utf-8"))["rows"]}
    kb=parse_kb(); kbret=KBRetriever(device="cpu")   # e5
    # passages
    full={k:" ".join(kb[K2SEC[k]]["symptoms"]) for k in DISEASE}; full["normal"]=NORMAL_PASS
    vis={}
    for k in DISEASE:
        lines=kbret.retrieve_lines(K2SEC[k],VQ,"병징",top_n=3)
        vis[k]=" ".join(l["text"] for l in lines)
    vis["normal"]=NORMAL_PASS
    allk=DISEASE+["normal"]

    def encode(model,texts,is_e5,is_query):
        if is_e5:
            pref="query: " if is_query else "passage: "
            texts=[pref+t for t in texts]
        return model.encode(texts,normalize_embeddings=True,batch_size=16)

    models={"e5":kbret.model, "bge-m3":SentenceTransformer("BAAI/bge-m3",device="cpu")}
    report={}
    for mname,model in models.items():
        is_e5=(mname=="e5")
        for ptype,passages in [("full",full),("visual",vis)]:
            cond=f"{mname}/{ptype}"
            pemb={k:encode(model,[passages[k]],is_e5,False)[0] for k in allk}
            # (2a) 병해 쌍별 코사인 산포 (7병해)
            cos=[float(pemb[a]@pemb[b]) for a,b in itertools.combinations(DISEASE,2)]
            spread={"mean_pair_cos":round(float(np.mean(cos)),4),
                    "min":round(min(cos),4),"max":round(max(cos),4),
                    "std":round(float(np.std(cos)),4)}
            # (2b) 캡션 기반 진단
            stds=[]; wrong_rank1=0; wrong_n=0; all_rank1=0; all_n=0
            for b,c in caps.items():
                s=s1[b]; top3=s["score_top3"]; gold=s["gold"]
                cemb=encode(model,[c["caption"]],is_e5,True)[0]
                sc={k:float(cemb@pemb[k]) for k in top3}
                stds.append(float(np.std(softmax([sc[k] for k in top3]))))
                if gold in top3:
                    all_n+=1
                    rank=sorted(top3,key=lambda k:-sc[k]).index(gold)+1
                    if rank==1: all_rank1+=1
                    if s["score_top1"]!=gold:
                        wrong_n+=1
                        if rank==1: wrong_rank1+=1
            stds=sorted(stds); med=stds[len(stds)//2]
            gate=bool(med>=0.02 and wrong_n and wrong_rank1/wrong_n>0.333)
            report[cond]={"pair_cos":spread,"softmax_std_median":round(med,4),
                          "wrong_gold_rank1":f"{wrong_rank1}/{wrong_n}",
                          "wrong_rank1_ratio":round(wrong_rank1/wrong_n,3) if wrong_n else None,
                          "all_rank1_ratio":round(all_rank1/all_n,3) if all_n else None,
                          "gate_pass":gate}
    json.dump(report,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("="*66); print("  EXP-2 임베딩×발췌 비교"); print("="*66)
    print(f"  {'조건':<14}{'쌍cos평균':>9}{'std중앙':>9}{'오답1위':>10}{'게이트':>7}")
    for c,r in report.items():
        print(f"  {c:<14}{r['pair_cos']['mean_pair_cos']:>9}{r['softmax_std_median']:>9}"
              f"{r['wrong_rank1_ratio']:>10}{'통과' if r['gate_pass'] else '미달':>7}")
    anypass=any(r["gate_pass"] for r in report.values())
    print(f"\n  게이트 통과 조건: {'있음 → 융합 재검토' if anypass else '없음 → 임베딩·발췌 무관 확정'}")
    print(f"  저장: {OUT}")

if __name__=="__main__": main()
