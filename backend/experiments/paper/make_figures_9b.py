import os, json, csv
import sys
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType 임베딩 (IEEE: Type 3 금지)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "path"  # SVG 글자를 벡터 패스로 → 폰트 의존 없음
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import RESULTS_DIR, PAPER_DOCS_DIR  # noqa: E402
R = RESULTS_DIR
OUT_DIR = os.path.join(PAPER_DOCS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.size":8,"font.family":"DejaVu Sans","axes.linewidth":0.6,
                     "xtick.major.width":0.6,"ytick.major.width":0.6})
s1=json.load(open(f"{R}/b_stage1_top3_9b.json"))["rows"]
mar_c=[r["margin_1_2"] for r in s1 if r["score_top1"]==r["gold"]]
mar_w=[r["margin_1_2"] for r in s1 if r["score_top1"]!=r["gold"]]
med_c,med_w=float(np.median(mar_c)),float(np.median(mar_w))
with open(f"{OUT_DIR}/fig2_margins.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["margin","group"])
    [w.writerow([m,"correct"]) for m in mar_c]; [w.writerow([m,"incorrect"]) for m in mar_w]
fig,ax=plt.subplots(figsize=(3.5,2.4))
bins=np.linspace(0,2.5,11)
ax.hist(mar_c,bins=bins,color="0.75",edgecolor="black",linewidth=0.6,label=f"Correct (n={len(mar_c)})")
ax.hist(mar_w,bins=bins,color="0.35",edgecolor="black",linewidth=0.6,hatch="//",alpha=0.9,label=f"Incorrect (n={len(mar_w)})")
ax.axvline(med_c,color="0.6",ls="--",lw=1.2); ax.axvline(med_w,color="0.2",ls=":",lw=1.2)
ymax=ax.get_ylim()[1]
ax.text(med_w-0.03,ymax*0.55,f"med={med_w:.2f}",fontsize=6.5,ha="right",color="0.2")
ax.text(med_c+0.03,ymax*0.55,f"med={med_c:.2f}",fontsize=6.5,ha="left",color="0.35")
ax.set_xlabel("Confidence margin (top-1 minus top-2 log-likelihood)")
ax.set_ylabel("Count"); ax.legend(frameon=False,fontsize=7,loc="upper left"); ax.set_xlim(0,2.5)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT_DIR}/fig2_margin_dist.png",dpi=300); fig.savefig(f"{OUT_DIR}/fig2_margin_dist.pdf"); fig.savefig(f"{OUT_DIR}/fig2_margin_dist.svg"); plt.close(fig)
e3=json.load(open(f"{R}/exp3_confidence_9b.json"))
rc=e3["risk_coverage"]; cov=[p["coverage"] for p in rc]; risk=[p["risk"] for p in rc]
acc30=[r for r in e3["reject_policy"] if r["reject_pct"]==0.3][0]["acc"]
with open(f"{OUT_DIR}/risk_coverage.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["coverage","risk"]); [w.writerow([c,r]) for c,r in zip(cov,risk)]
fig,ax=plt.subplots(figsize=(3.5,2.4))
ax.plot(cov,risk,color="black",lw=1.3,marker="o",ms=2.5,mfc="white",mec="black",mew=0.6)
ax.axvline(0.70,color="0.5",ls="--",lw=1.0)
ax.annotate(f"30% deferred\ncoverage 0.70, accuracy {acc30:.2f}",xy=(0.70,risk[int(len(risk)*0.7)-1]),
            xytext=(0.40,max(risk)*0.85),fontsize=6.5,arrowprops=dict(arrowstyle="->",lw=0.6))
ax.text(0.04,max(risk)*0.05,f"AURC = {e3['aurc']:.3f}",fontsize=7)
ax.set_xlabel("Coverage (fraction of queries answered)")
ax.set_ylabel("Selective risk"); ax.set_xlim(0,1.02); ax.set_ylim(0,max(risk)*1.15)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT_DIR}/risk_coverage.png",dpi=300); fig.savefig(f"{OUT_DIR}/risk_coverage.pdf"); fig.savefig(f"{OUT_DIR}/risk_coverage.svg"); plt.close(fig)
print("Fig2/Fig3 (9B) 완료:")
print(f"  Fig2: correct med={med_c:.3f}(n={len(mar_c)}) / incorrect med={med_w:.3f}(n={len(mar_w)})")
print(f"  Fig3: AURC={e3['aurc']}, 30%보류 acc={acc30}")
