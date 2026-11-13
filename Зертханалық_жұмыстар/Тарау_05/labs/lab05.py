"""ЗЖ 5. Метрикалар арифметикасы және train/validation/test синтетикалық тексеруі."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_curve, roc_curve, auc)

ROOT = Path(__file__).resolve().parents[1]
FIGURES, DATASETS = ROOT / "figures", ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

def label_metrics(y, p, threshold):
    prediction = (np.asarray(p)>=threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0,1]).ravel()
    return {"threshold": float(threshold), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y,prediction),
            "precision": precision_score(y,prediction,pos_label=1,zero_division=0),
            "recall": recall_score(y,prediction,pos_label=1,zero_division=0),
            "F1": f1_score(y,prediction,pos_label=1,zero_division=0), "cost_FP1_FN5": int(fp+5*fn)}

def probability_metrics(y,p):
    return {"ROC_AUC": roc_auc_score(y,p), "AP": average_precision_score(y,p,pos_label=1),
            "Brier": brier_score_loss(y,p,pos_label=1,scale_by_half=True),
            "log_loss": log_loss(y,p,labels=[0,1])}

def make_model():
    return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1,max_iter=2000))])

def manual_experiment():
    y = np.array([0,1,0,1,1,0,1,0])
    p = np.array([.1,.9,.4,.7,.3,.2,.6,.8])
    rows = [label_metrics(y,p,t) for t in [.5,.3]]
    probabilities = probability_metrics(y,p)
    precision, recall, _ = precision_recall_curve(y,p,pos_label=1)
    trapezoid = auc(recall,precision)
    print("Жасанды қолмен есептеу мысалы\n",pd.DataFrame(rows).to_string(index=False))
    print("Ықтималдық көрсеткіштері",probabilities,"PR трапециялары",trapezoid)
    return {"synthetic_arithmetic":True,"y":y.tolist(),"p":p.tolist(),"thresholds":rows,
            **probabilities,"trapezoidal_PR_AUC":float(trapezoid)}

def make_split():
    raw, target = make_classification(n_samples=1000,n_features=20,n_informative=5,
        n_redundant=2,weights=[.9,.1],class_sep=1.2,flip_y=.02,random_state=2026)
    X, y = pd.DataFrame(raw,columns=[f"x{i}" for i in range(20)]),pd.Series(target,name="target")
    Xdev,Xtest,ydev,ytest = train_test_split(X,y,test_size=.2,stratify=y,random_state=2026)
    Xtrain,Xval,ytrain,yval = train_test_split(Xdev,ydev,test_size=.25,stratify=ydev,random_state=2027)
    return {"X":X,"y":y,"train":(Xtrain,ytrain),"validation":(Xval,yval),"test":(Xtest,ytest)}

def train_cv(split):
    Xtrain,ytrain = split["train"]
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=2028)
    scores = cross_validate(make_model(),Xtrain,ytrain,cv=cv,n_jobs=1,
                            scoring={"ROC_AUC":"roc_auc","AP":"average_precision"})
    result = {name:{"fold_scores":scores[f"test_{name}"].tolist(),
                    "mean":float(scores[f"test_{name}"].mean()),
                    "std_fold_ddof1":float(scores[f"test_{name}"].std(ddof=1))}
              for name in ["ROC_AUC","AP"]}
    print("Тек 600 train жолындағы бөлек CV, баптаулар таңдалмайды:",result)
    return result

def fit_and_select(split):
    Xtrain,ytrain = split["train"]
    Xval,yval = split["validation"]
    model,dummy = make_model(),DummyClassifier(strategy="prior")
    model.fit(Xtrain,ytrain)
    dummy.fit(Xtrain,ytrain)
    pval = model.predict_proba(Xval)[:,1]
    thresholds = np.unique(np.r_[0.,pval,1.])
    candidates = [label_metrics(yval,pval,float(t)) for t in thresholds]
    selected = min(candidates,key=lambda row:(row["cost_FP1_FN5"],-row["threshold"]))
    print("Тек validation бойынша таңдалды:",selected)
    return {"model":model,"dummy":dummy,"pval":pval,"candidates":candidates,"selected":selected,
            "coefficients_after_fit":model.named_steps["model"].coef_.copy()}

def test_experiment(split,state):
    Xtest,ytest = split["test"]
    p = state["model"].predict_proba(Xtest)[:,1]
    pdummy = state["dummy"].predict_proba(Xtest)[:,1]
    chosen = state["selected"]["threshold"]
    rows = [{"rule":"Dummy t=.5",**label_metrics(ytest,pdummy,.5)},
            {"rule":"Logistic t=.5",**label_metrics(ytest,p,.5)},
            {"rule":"Logistic t=validation",**label_metrics(ytest,p,chosen)}]
    probs = [{"model":"DummyPrior",**probability_metrics(ytest,pdummy)},
             {"model":"Logistic",**probability_metrics(ytest,p)}]
    print("Бір бекітілген test\n",pd.DataFrame(rows).to_string(index=False))
    print("Сол ықтималдықтардың көрсеткіштері\n",pd.DataFrame(probs).to_string(index=False))
    fig,axs = plt.subplots(1,2,figsize=(8.2,3.9),layout="constrained")
    fpr,tpr,_ = roc_curve(ytest,p,pos_label=1)
    precision,recall,_ = precision_recall_curve(ytest,p,pos_label=1)
    axs[0].plot(fpr,tpr,label=f"Logistic AUC={probs[1]['ROC_AUC']:.3f}")
    axs[0].plot([0,1],[0,1],"k--",linewidth=1,label="Диагональ")
    axs[0].set(xlabel="FPR",ylabel="TPR",title="test ішіндегі ROC")
    axs[1].step(recall,precision,where="post",label=f"Logistic AP={probs[1]['AP']:.3f}")
    axs[1].axhline(float(ytest.mean()),linestyle="--",color="black",linewidth=1,
                   label=f"1 класының үлесі={ytest.mean():.3f}")
    axs[1].set(xlabel="Recall",ylabel="Precision",title="test ішіндегі PR",ylim=(-.02,1.05))
    axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
    fig.savefig(FIGURES/"ch05_05_roc_pr.png",dpi=180);plt.close(fig)
    edges = np.linspace(0,1,6)
    assignments = np.searchsorted(edges[1:-1],p,side="right")
    bins=[]
    for j in range(5):
        mask=assignments==j
        bins.append({"lower":float(edges[j]),"upper":float(edges[j+1]),"n":int(mask.sum()),
                     "mean_probability":float(p[mask].mean()) if mask.any() else None,
                     "positive_fraction":float(ytest.to_numpy()[mask].mean()) if mask.any() else None})
    fig,axs=plt.subplots(1,2,figsize=(8.2,4.0),layout="constrained")
    axs[0].step([r["threshold"] for r in state["candidates"]],
                [r["cost_FP1_FN5"] for r in state["candidates"]],where="pre")
    axs[0].axvline(chosen,color="#B34B20",linestyle="--",label=f"t*={chosen:.4f}")
    axs[0].scatter([chosen],[state["selected"]["cost_FP1_FN5"]],color="#B34B20")
    axs[0].set(xlabel="Шек t",ylabel="FP + 5 FN",title="validation бойынша таңдау",xlim=(0,1))
    axs[0].legend(fontsize=8)
    nonempty=[r for r in bins if r["n"]]
    axs[1].plot([0,1],[0,1],"k--",linewidth=1)
    axs[1].plot([r["mean_probability"] for r in nonempty],
                [r["positive_fraction"] for r in nonempty],"o-")
    for r in nonempty:
        axs[1].annotate(f"n={r['n']}",(r["mean_probability"],r["positive_fraction"]),
                        xytext=(4,5),textcoords="offset points",fontsize=8)
    axs[1].set(xlabel="Орташа ықтималдық",ylabel="1 класының үлесі",
               title="test диагностикасы",xlim=(-.03,1.06),ylim=(-.03,1.13))
    fig.savefig(FIGURES/"ch05_06_validation_calibration.png",dpi=180);plt.close(fig)
    print("Калибрлеу диагностикасы, 5 бекітілген аралық:",bins)
    split_names=pd.Series("",index=split["X"].index)
    for name in ["train","validation","test"]:
        split_names.loc[split[name][0].index]=name
    split["X"].assign(target=split["y"],split=split_names).to_csv(
        DATASETS/"synthetic_classification_validation.csv",index=False)
    memberships=[set(split[name][0].index) for name in ["train","validation","test"]]
    return {"synthetic":True,"rows":1000,"features":20,"sizes":[600,200,200],
            "positive_counts":{name:int(split[name][1].sum()) for name in ["train","validation","test"]},
            "selected_validation":state["selected"],"validation_candidates":state["candidates"],
            "test_threshold_rows":rows,"test_probability_metrics":probs,"test_calibration_bins":bins,
            "main_model_fit_rows":int(state["model"].named_steps["scale"].n_samples_seen_),
            "model_unchanged_after_selection":bool(np.array_equal(state["coefficients_after_fit"],state["model"].named_steps["model"].coef_)),
            "splits_disjoint":all(not memberships[i]&memberships[j] for i in range(3) for j in range(i)),
            "threshold_tie_rule":"largest threshold among minimum-cost candidates", "refit_on_800":False}

if __name__=="__main__":
    manual=manual_experiment()
    split=make_split()
    cv=train_cv(split)
    state=fit_and_select(split)
    experiment=test_experiment(split,state)
    (Path(__file__).parent/"lab05_results.json").write_text(
        json.dumps({"manual":manual,"train_only_CV":cv,"experiment":experiment},ensure_ascii=False,indent=2),encoding="utf-8")
