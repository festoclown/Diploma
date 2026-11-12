"""Publication figures with visible grids, paired effects and explicit sample sizes."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("--results",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--diagnostics",type=Path)
parser.add_argument("--refinement",type=Path)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,
    "axes.grid":True,"axes.axisbelow":True,"grid.alpha":.35,"grid.linestyle":":",
    "legend.fontsize":10})
names={"breast_cancer":"Breast Cancer","digits":"Digits","dry_bean":"Dry Bean",
       "ag_news":"AG News","20newsgroups":"20 Newsgroups","banking77":"Banking77"}
def save(fig,name):
    for ax in fig.axes:
        ax.grid(True,which="major",axis="both")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.out/(name+".pdf"),bbox_inches="tight")
    fig.savefig(args.out/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,6.3),sharex=True,sharey=True)
summary=pd.read_csv(args.diagnostics/"summary.csv") if args.diagnostics else None
for ax,budget in zip(axes,[20,50]):
    for position,(name,title) in enumerate(names.items()):
        data=pd.read_csv(args.results/(name+".csv"))
        gain=data[data.budget==budget].gain.to_numpy()
        # Aggregate exact repeated effects and explicitly label their multiplicities.
        values, counts = np.unique(np.round(gain, 12), return_counts=True)
        for index, (value, count) in enumerate(zip(values, counts)):
            ax.scatter(value, position + .22, s=18 + 6 * count, color="C0", alpha=.65)
            near_previous = index > 0 and value - values[index - 1] < .04
            level = (level + 1) % 3 if near_previous else 0
            ax.annotate(f"×{count}", (value, position + .22), xytext=(0, 8 + 12 * level),
                        textcoords="offset points", ha="center", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "0.55", "lw": .5})
        zeros = int(np.sum(np.abs(gain) <= 1e-12))
        ax.text(.98, position - .23, f"нулей: {zeros}",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="0.35")
        if summary is not None:
            row=summary[(summary.dataset==name)&(summary.budget==budget)].iloc[0]
            ax.plot([row.ci_low,row.ci_high],[position-.12,position-.12],color="black",lw=2)
        ax.scatter(gain.mean(),position-.12,s=35,color="C3",marker="D",zorder=4)
    ax.set_yticks(range(6),list(names.values()))
    ax.axvline(0,color="grey",lw=1)
    ax.set(title=f"{budget} меток, 20 раскрытий",xlabel="Выигрыш M6 по ARI")
save(fig,"domain_gain")
pilots=pd.read_csv(args.results/"pilots.csv")
fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharex=True)
for ax,scenario in zip(axes,["random","missing_classes"]):
    block=pilots[pilots.scenario==scenario]
    labels=[];ex=[];inside=[]
    for name,g in block.groupby("pilot"):
        titles = {"gpu_pilot": "Синтетические данные", "gpu_breast_cancer_light": "Breast Cancer",
                  "gpu_digits_light": "Digits, silhouette", "gpu_digits_db_light": "Digits, DB"}
        labels.append(titles[name]+f" (n={len(g)})")
        ex.append(g.excluded_loss.mean());inside.append(g.inside_loss.mean())
    ax.barh(labels,ex,label="Исключение",color="C0")
    ax.barh(labels,inside,left=ex,label="Выбор внутри допуска",color="C1")
    ax.set(title="Случайные метки" if scenario=="random" else "Пропущенные классы",
           xlabel="Средняя потеря ARI")
axes[1].legend(loc="lower right")
save(fig,"pilot_losses")
rect=pd.read_csv(args.results/"rectangles.csv")
fig,axes=plt.subplots(2,2,figsize=(10,6.5))
centers=np.repeat([[-3,-1],[-3,1],[3,-1],[3,1]],100,axis=0)
x=centers+np.random.default_rng(101).normal(0,.2,centers.shape)
for j in range(2):
    axes[0,j].scatter(x[:,0],x[:,1],c=x[:,j]>0,s=6,cmap="coolwarm")
    axes[0,j].set(xlabel="Первая координата",ylabel="Вторая координата",
                  title=["Слева / справа, нач. знач. 101","Сверху / снизу, нач. знач. 101"][j])
    axes[0,j].set_aspect("equal")
paired=rect.pivot(index=["seed","repeat","budget"],columns="target",values="gain")
paired["paired"]=paired["left-right"]-paired["up-down"]
for seed,g in paired.groupby("seed"):
    means=g.groupby("budget").paired.mean()
    axes[1,0].plot(means.index,means.values,marker="o",label=f"нач. знач. {seed}")
axes[1,0].set(xlabel="Число меток",ylabel="Средняя парная разность выигрышей",xticks=[4,8,20])
axes[1,0].legend()
for pos,budget in enumerate([4,8,20]):
    for color,(seed,offset) in enumerate(zip([101,102,103],[-.2,0,.2])):
        g=paired.xs((seed,budget),level=("seed","budget"))
        axes[1,1].scatter(pos+offset+np.linspace(-.035,.035,len(g)),g.paired,s=16,alpha=.5,
                          color=f"C{color}",label=f"нач. знач. {seed}" if pos==0 else None)
axes[1,1].set(xticks=[0,1,2],xticklabels=[4,8,20],xlabel="Число меток",ylabel="Парная разность по раскрытиям")
axes[1,1].legend()
for ax in axes[1]:ax.axhline(0,color="grey",lw=1)
save(fig,"rectangle_control")


if args.refinement:
    coverage = pd.read_csv(args.refinement / "coverage_draws.csv")
    categories = pd.read_csv(args.refinement / "coverage_categories.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, dataset in enumerate(["digits", "dry_bean"]):
        for color, present in enumerate([2, 4, 6]):
            block = coverage[(coverage.dataset == dataset) & (coverage.present == present)]
            axes[row, 0].scatter(present + np.linspace(-.2, .2, len(block)), block.gain,
                                 s=24, alpha=.7, color=f"C{color}")
            frequency = categories[(categories.dataset == dataset) &
                                   (categories.present == present)].groupby("category").selected.mean()
            axes[row, 1].plot(frequency.index, frequency, marker="o",
                              label=f"{present} класса", color=f"C{color}")
        axes[row, 0].axhline(0, color="grey", lw=1)
        axes[row, 0].set(title=names[dataset], xlabel="Представленных классов",
                         ylabel="Выигрыш M6 по ARI", xticks=[2, 4, 6])
        axes[row, 1].set(title=names[dataset], xlabel="Код категории",
                         ylabel="Частота включения категории", ylim=(-.02, 1.02))
        axes[row, 1].legend()
    save(fig, "coverage_variability")

    mapping = pd.read_csv(args.refinement / "bootstrap_map.csv")
    bad = mapping[~mapping.same_as_reference]
    affected = bad[["dataset", "repeat", "budget"]].drop_duplicates()
    if len(affected):
        fig, ax = plt.subplots(figsize=(10, 3))
        matrix, labels = [], []
        columns = [(B, seed) for B in [30, 100, 300] for seed in range(5)]
        for item in affected.to_dict("records"):
            block = mapping[(mapping.dataset == item["dataset"]) &
                            (mapping["repeat"] == item["repeat"]) &
                            (mapping.budget == item["budget"])].set_index(["draws", "seed"])
            matrix.append([block.loc[key].delta_Q for key in columns])
            labels.append(f'{names[item["dataset"]]}, r={item["repeat"]}, m={item["budget"]}')
        limit = max(abs(np.asarray(matrix)).max(), .001)
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(len(columns)), [f"{B} / {seed}" for B, seed in columns], rotation=45)
        ax.set(xlabel="B / нач. знач.", title="Изменение ARI относительно B=3000, нач. знач. 99")
        ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", alpha=.45)
        fig.colorbar(im, ax=ax, label="Разность проверочных ARI")
        save(fig, "bootstrap_disagreements")
