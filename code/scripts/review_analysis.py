"""Reproduce diagnostics and post-review CPU controls from frozen vectors.
No model/data downloads. Original choices are checked before new analysis.
"""
import argparse
import json
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.stats import beta
from clusterization.backend import Backend
from clusterization.light_series import DOMAINS, make_pool, archive_inputs, PILOTS
from clusterization.metrics import ari_grid
from clusterization.sampling import reserve_test_split, bootstrap_weights
from clusterization.selection import choose, TOLERANCE
from clusterization.storage import derived_seed, file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--archive", type=Path, help="Directory with archived pilot data, pools and results")
args = parser.parse_args()
root, out = args.root.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
backend = Backend("cpu")
tables = {k: [] for k in ["draws","candidates","sensitivity","pool_control","coverage_control",
                          "bootstrap_control","bootstrap_membership","provenance","pilots","rectangle_candidates",
                          "rectangle_details"]}
frozen = out / "frozen"
frozen.mkdir(exist_ok=True)

def diagnostic(y, groups, pool, cand, idx, test, seed, draws=100):
    partial = ari_grid(y[idx], pool[:,idx])[0]
    weights = bootstrap_weights(groups[idx],draws,np.random.default_rng(seed))
    boot = ari_grid(y[idx],pool[:,idx],weights)
    raw = np.array([c["silhouette"] for c in cand])
    decisions = {m: choose(m,partial,raw,raw,boot) for m in ["M0","M1","M2","M5","M6","M7"]}
    # Test labels enter only after fixing every choice.
    q = ari_grid(y[test],pool[:,test])[0]
    leader=decisions["M1"]["selected"]
    sigma=(boot[:,leader,None]-boot).std(axis=0,ddof=1)
    delta=partial[leader]-partial
    allowed=delta<=sigma+TOLERANCE
    sel=decisions["M6"]["selected"]
    rho=np.ptp(partial-q)
    counts=np.unique(y[idx],return_counts=True)[1]
    wc=np.stack([weights[:,y[idx]==k].sum(axis=1) for k in np.unique(y[idx])],axis=1)
    pair=((wc*(wc-1))/2).sum(axis=1)
    degenerate=((wc>0).sum(axis=1)<2)|(pair==0)
    row=dict(gain=q[sel]-q[leader],oracle=q.max(),coverage=len(counts)/len(np.unique(y)),
             class_counts=json.dumps(sorted(counts.tolist())),positive_pairs=int((counts*(counts-1)/2).sum()),
             single_class=len(counts)==1,all_singleton=bool((counts==1).all()),
             top_ties=int((delta<=TOLERANCE).sum()),bootstrap_degenerate=degenerate.mean(),
             switched=sel!=leader,selected_gap=delta[sel],selected_sd=sigma[sel],
             eligible=int(allowed.sum()),excluded_loss=q.max()-q[allowed].max(),
             inside_loss=q[allowed].max()-q[sel],rho=rho,
             leader_bound=rho,adaptive_bound=rho+sigma[sel]+TOLERANCE,
             M6_minus_M2=q[sel]-q[decisions["M2"]["selected"]],
             M6_minus_M5=q[sel]-q[decisions["M5"]["selected"]])
    for m,d in decisions.items():
        j=d["selected"]
        row[m]=cand[j]["id"]
        row[m+"_ari"]=q[j]
        row[m+"_regret"]=q.max()-q[j]
    assert row["M1_regret"]<=rho+1e-9
    assert row["M6_regret"]<=row["adaptive_bound"]+1e-9
    return row,partial,boot,q,raw,sigma

for name in DOMAINS:
    prepared=next((root/"data/processed").glob(name+"-*/manifest.json")).parent
    manifest=json.loads((prepared/"manifest.json").read_text())
    text=manifest["dataset"]["kind"]=="text"
    vector=next(prepared.glob("embedding-*/vectors.npz")) if text else prepared/"vectors.npz"
    with np.load(vector) as f: x=backend.array(f["x"])
    if text:x=backend.project(x,50)
    pool,cand=make_pool(backend,x)
    with np.load(prepared/"labels.npz") as f:y,groups,ids=f["y"],f["groups"],f["ids"]
    reserve,test=reserve_test_split(groups,31415)
    np.savez_compressed(frozen/(name+".npz"),x=backend.host(x),pool=pool,y=y,groups=groups,ids=ids,reserve=reserve,test=test)
    (frozen/(name+"-candidates.json")).write_text(json.dumps(cand,ensure_ascii=False,indent=2))
    provenance=dict(dataset=name,n=len(y),classes=len(np.unique(y)),reserve=len(reserve),test=len(test),
        groups=len(np.unique(groups)),largest_group=int(np.unique(groups,return_counts=True)[1].max()),
        data=manifest,embedding_manifest=json.loads((vector.parent/"manifest.json").read_text()) if text else None,
        labels_sha256=file_hash(prepared/"labels.npz"),vectors_sha256=file_hash(vector))
    (frozen/(name+"-provenance.json")).write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    tables["provenance"].append({k:v for k,v in provenance.items() if k not in ["data","embedding_manifest"]})
    q=ari_grid(y[test],pool[:,test])[0]
    for i,c in enumerate(cand):tables["candidates"].append(dict(dataset=name,**c,q=q[i]))
    original=pd.read_csv(root/"reports/light_series/cpu"/(name+".csv"))
    partials=[]
    for repeat in range(20):
        order=np.random.default_rng(derived_seed(name,repeat)).permutation(reserve)
        for budget in [20,50]:
            idx=order[:budget]
            seed=derived_seed(name,repeat,budget)
            row,a,b,q,raw,sigma=diagnostic(y,groups,pool,cand,idx,test,seed)
            old=original[(original.repeat==repeat)&(original.budget==budget)].iloc[0]
            for m in ["M1","M2","M5","M6"]:
                if row[m]!=old[m]:raise ValueError(f"Frozen replay mismatch {name} {repeat} {budget} {m}")
            tables["draws"].append(dict(dataset=name,repeat=repeat,budget=budget,**row))
            partials.append((budget,a,q,sigma))
            for method,values in [("M5",[0,.025,.05,.1,.2]),("M6",[0,.5,1,2])]:
                for value in values:
                    d=choose(method,a,raw,raw,b,fixed_window=value if method=="M5" else .05,
                             adaptive_multiplier=value if method=="M6" else 1)
                    j=d["selected"]
                    tables["sensitivity"].append(dict(dataset=name,repeat=repeat,budget=budget,
                        method=method,parameter=value,gain=q[j]-row["M1_ari"],eligible=d["eligible_count"]))
            # Remove spectral candidates, keeping the objects, annotations and bootstrap weights fixed.
            km=np.array([any(alias.startswith("kmeans") for alias in c["aliases"]) for c in cand])
            for label,mask in [("kmeans",km),("full",np.ones(len(cand),bool))]:
                ca=[c for c,keep in zip(cand,mask) if keep]
                ra=raw[mask]
                d1=choose("M1",a[mask],ra,ra,b[:,mask])
                d6=choose("M6",a[mask],ra,ra,b[:,mask])
                tables["pool_control"].append(dict(dataset=name,repeat=repeat,budget=budget,pool=label,
                    gain=q[mask][d6["selected"]]-q[mask][d1["selected"]],
                    M1=ca[d1["selected"]]["id"],M6=ca[d6["selected"]]["id"]))
    # Fixed-budget diagnostic control: exact number of represented classes, balanced quotas.
    if name in ["digits","dry_bean"]:
        for present in [2,4,6]:
            eligible_classes=[k for k in np.unique(y[reserve]) if (y[reserve]==k).sum()>=10]
            for repeat in range(20):
                rng=np.random.default_rng(derived_seed(name,"coverage",present,repeat))
                classes=rng.choice(eligible_classes,present,replace=False)
                quota=np.full(present,20//present);quota[:20%present]+=1
                idx=np.concatenate([rng.choice(reserve[y[reserve]==k],n,replace=False) for k,n in zip(classes,quota)])
                row,*_=diagnostic(y,groups,pool,cand,idx,test,derived_seed(name,"coverageboot",present,repeat))
                tables["coverage_control"].append(dict(dataset=name,present=present,repeat=repeat,budget=20,**row))
    # All 40 label sets, five bootstrap seeds, reference B=3000 at seed 99.
    if name in ["digits","breast_cancer"]:
        for repeat in range(20):
            order=np.random.default_rng(derived_seed(name,repeat)).permutation(reserve)
            for budget in [20,50]:
                idx=order[:budget]
                ref,a,bb,q,raw,ss=diagnostic(y,groups,pool,cand,idx,test,99,3000)
                leader=choose("M1",a,raw,raw,bb)["selected"]
                for draws in [30,100,300]:
                    for seed in range(5):
                        row,a,bb,q,raw,sd=diagnostic(y,groups,pool,cand,idx,test,seed,draws)
                        tables["bootstrap_control"].append(dict(dataset=name,repeat=repeat,budget=budget,
                            draws=draws,seed=seed,same_as_reference=row["M6"]==ref["M6"],**row))
                        for j,c in enumerate(cand):
                            margin=sd[j]-(a[leader]-a[j])+TOLERANCE
                            tables["bootstrap_membership"].append(dict(dataset=name,repeat=repeat,budget=budget,
                                draws=draws,seed=seed,candidate=c["id"],margin=margin,included=margin>=0))
    print("replayed",name,flush=True)

archive=args.archive or root.parents[1]/"outputs/oslide/replay_source"
for name in PILOTS:
    directory,pool,cand,raw,y,groups,test,saved=archive_inputs(archive,name)
    conf=json.loads((directory/"manifest.json").read_text())["config"]
    tables["pilots"].append(dict(pilot=name,n=len(y),pool=len(cand),budgets=str(conf["selection"]["budgets"]),
        scenarios=str(conf["selection"]["scenarios"]),repeats=conf["selection"]["repeats"],
        B=conf["selection"]["bootstrap_draws"],internal=conf["selection"]["internal"],conditions=len(saved[saved.method=="M1"])))
for seed in [101,102,103]:
    x=np.repeat([[-3,-1],[-3,1],[3,-1],[3,1]],100,axis=0)
    x=(x+np.random.default_rng(seed).normal(0,.2,x.shape)).astype("float32")
    pool,cand=make_pool(backend,x,True)
    for axis,target in enumerate(["left-right","up-down"]):
        y=x[:,axis]>0;groups=np.arange(400);reserve,test=reserve_test_split(groups,31415)
        q=ari_grid(y[test],pool[:,test])[0]
        for i,c in enumerate(cand):tables["rectangle_candidates"].append(dict(seed=seed,target=target,**c,q=q[i]))
        for repeat in range(20):
            order=np.random.default_rng(derived_seed(seed,repeat)).permutation(reserve)
            for budget in [4,8,20]:
                row,a,b,q,raw,sd=diagnostic(y,groups,pool,cand,order[:budget],test,derived_seed(seed,repeat,budget))
                leader=choose("M1",a,raw,raw,b)["selected"]
                for j,c in enumerate(cand):
                    tables["rectangle_details"].append(dict(seed=seed,target=target,repeat=repeat,budget=budget,
                        candidate=c["id"],aliases=str(c["aliases"]),k=c["k"],silhouette=raw[j],q=q[j],
                        partial=a[j],gap=a[leader]-a[j],sd=sd[j],included=a[leader]-a[j]<=sd[j]+TOLERANCE,
                        leader=c["id"]==row["M1"],selected=c["id"]==row["M6"],
                        gain=row["gain"],potential=q[a[leader]-a<=sd+TOLERANCE].max()-q[leader],
                        inside_loss=row["inside_loss"]))
for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)
(out/"environment.json").write_text(json.dumps(dict(python=platform.python_version(),
    packages={p:version(p) for p in ["numpy","scipy","pandas","scikit-learn","matplotlib"]}),indent=2))
# Conditional summaries. Resampling the 20 paired effects keeps X, pool and T fixed.
d=pd.DataFrame(tables["draws"]); summaries=[]
for (name,budget),g in d.groupby(["dataset","budget"]):
    values=g.gain.to_numpy();rng=np.random.default_rng(20261112)
    ci=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1),[.025,.975])
    sw=g.switched.to_numpy();bad=values < -TOLERANCE;k=bad.sum();n=len(g);s=sw.sum()
    def interval(k,n):
        if n==0:return np.nan,np.nan
        return (0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k))
    lo,hi=interval(k,n);clo,chi=interval(k,s)
    summaries.append(dict(dataset=name,budget=budget,M1=g.M1_ari.mean(),M6=g.M6_ari.mean(),
        oracle=g.oracle.iloc[0],R1=g.M1_regret.mean(),R6=g.M6_regret.mean(),gain=values.mean(),
        ci_low=ci[0],ci_high=ci[1],switches=s,worse=k,conditional_gain=values[sw].mean() if s else np.nan,
        bad_low=lo,bad_high=hi,conditional_bad_low=clo,conditional_bad_high=chi))
pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
# Explicit existence example for H3: outside the labelled class, two candidates differ.
y=np.array([0,0,1,1,2,2]);p=np.array([[0,0,1,1,2,2],[0,0,1,1,0,1]])
a=ari_grid(y[:4],p[:,:4])[0]
assert np.all(a==1)
(out/"H3_example.json").write_text(json.dumps(dict(y=y.tolist(),partitions=p.tolist(),
    labelled=[0,1,2,3],partial_ari=a.tolist(),full_ari=ari_grid(y,p)[0].tolist(),
    pairwise_sd=0,scope="constructive finite example"),indent=2))
print("completed",out,flush=True)
