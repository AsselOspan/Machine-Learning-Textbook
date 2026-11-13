"""ЗЖ 4. Төрт жіктеуіш пен базалық модельді бекітілген салыстыру."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

def make_models():
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "kNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=2026),
        "SVM": Pipeline([("scale", StandardScaler()),
                         ("model", SVC(C=1, kernel="rbf", gamma="scale"))]),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                max_features="sqrt", n_jobs=1, random_state=2026)}

def fixed_comparison(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=2026)
    labels = np.unique(y)
    models = make_models()
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "train_accuracy": accuracy_score(ytr, model.predict(Xtr)),
                     "test_accuracy": accuracy_score(yte, pred),
                     "F1macro": f1_score(yte, pred, average="macro"),
                     "correct": int(np.sum(pred==np.asarray(yte)))})
        matrices[name] = confusion_matrix(yte, pred, labels=labels).tolist()
    print(pd.DataFrame(rows).to_string(index=False))
    result = {"rows": len(X), "features": X.shape[1], "labels": labels.tolist(),
              "train_size": len(ytr), "test_size": len(yte), "seed": 2026,
              "class_counts": pd.Series(y).value_counts().sort_index().tolist(),
              "test_class_counts": pd.Series(yte).value_counts().sort_index().tolist(),
              "metrics": rows, "confusion_matrices": matrices,
              "split_disjoint": set(Xtr.index).isdisjoint(Xte.index),
              "scaler_samples_seen": {name: int(models[name].named_steps["scale"].n_samples_seen_)
                                       for name in ["kNN", "SVM"]}}
    return result, models, (Xtr, Xte, ytr, yte)

def moons_experiment():
    raw, target = make_moons(n_samples=600, noise=.2, random_state=2026)
    X, y = pd.DataFrame(raw, columns=["x1", "x2"]), pd.Series(target)
    result, models, parts = fixed_comparison(X, y)
    Xtr, Xte, ytr, yte = parts
    xx, yy = np.meshgrid(np.linspace(X.x1.min()-.3, X.x1.max()+.3, 180),
                         np.linspace(X.x2.min()-.3, X.x2.max()+.3, 180))
    grid = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
    cmap = ListedColormap(["#2469A0", "#D8792A"])
    fig, axs = plt.subplots(2, 2, figsize=(8, 6.2), layout="constrained")
    for ax, name in zip(axs.ravel(), ["kNN", "DecisionTree", "SVM", "RandomForest"]):
        pred = models[name].predict(Xte)
        regions = models[name].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, regions, levels=[-.5,.5,1.5], cmap=cmap, alpha=.17)
        if name == "SVM":
            score = models[name].decision_function(grid).reshape(xx.shape)
            ax.contour(xx, yy, score, levels=[0], colors="black", linewidths=.9)
        else:
            ax.contour(xx, yy, regions, levels=[.5], colors="black", linewidths=.7)
        ax.scatter(Xtr.x1, Xtr.x2, c=ytr, cmap=cmap, s=8, alpha=.22)
        ax.scatter(Xte.x1, Xte.x2, c=yte, cmap=cmap, s=19, edgecolor="black", linewidth=.35)
        wrong = pred!=yte.to_numpy()
        ax.scatter(Xte.loc[wrong,"x1"], Xte.loc[wrong,"x2"], marker="x", color="black", s=45)
        ax.set(title=f"{name}: дұрыс {int((~wrong).sum())}/{len(yte)}", xlabel="x1", ylabel="x2")
    handles = [Line2D([],[],marker="o",linestyle="",color="#2469A0",label="0-класс"),
               Line2D([],[],marker="o",linestyle="",color="#D8792A",label="1-класс"),
               Line2D([],[],marker="x",linestyle="",color="black",label="Тест қатесі")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "ch04_05_moons.png", dpi=180)
    plt.close(fig)
    X.assign(target=y).to_csv(DATASETS / "synthetic_moons.csv", index=False)
    result.update({"synthetic": True, "generator": "make_moons", "noise": .2,
                   "svm_n_support": models["SVM"].named_steps["model"].n_support_.tolist()})
    return result

def digits_experiment():
    data = load_digits(as_frame=True)
    X, y = data.data, data.target
    result, models, parts = fixed_comparison(X, y)
    Xtr, Xte, ytr, yte = parts
    pred = models["SVM"].predict(Xte)
    wrong_positions = np.flatnonzero(pred!=yte.to_numpy())
    examples = [{"row_id": int(Xte.index[pos]), "true_class": int(yte.iloc[pos]),
                 "predicted_class": int(pred[pos])} for pos in wrong_positions[:6]]
    matrix = np.array(result["confusion_matrices"]["SVM"])
    print("SVM: Digits матрицасы\n", matrix)
    print("SVM алғашқы қателері:", examples)
    gallery_columns = max(1, int(np.ceil(len(examples)/2)))
    fig = plt.figure(figsize=(6.5+.6*gallery_columns, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 2+gallery_columns, width_ratios=[1, 1]+[.9]*gallery_columns)
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set(xticks=range(10), yticks=range(10), xlabel="Болжам", ylabel="Шын цифр",
           title=f"SVM: дұрыс {len(yte)-len(wrong_positions)}/{len(yte)}")
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=7,
                    color="white" if matrix[i,j]>matrix.max()/2 else "black")
    for k in range(2*gallery_columns):
        ax = fig.add_subplot(gs[k//gallery_columns, k%gallery_columns+2])
        if k<len(examples):
            example = examples[k]
            ax.imshow(data.images[example["row_id"]], cmap="gray_r", vmin=0, vmax=16,
                      interpolation="nearest")
            ax.set_title(f"№{example['row_id']}\nшын. {example['true_class']}; болж. {example['predicted_class']}", fontsize=8)
        ax.set_axis_off()
    fig.savefig(FIGURES / "ch04_06_digits.png", dpi=180)
    plt.close(fig)
    data.frame.to_csv(DATASETS / "digits_sklearn.csv", index=False)
    attribution = ("Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI.\n"
                   "DOI: https://doi.org/10.24432/C50P49\n"
                   "Source: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits\n"
                   "License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/\n"
                   "Export: sklearn.datasets.load_digits, copy of original UCI test subset, 1797 rows; full UCI dataset has 5620 rows.\n"
                   "Feature names follow scikit-learn; labels 0-9 unchanged; images reconstructed from the 64 features.\n"
                   "The educational 1437/360 split is new and does not reproduce the original UCI train/test benchmark.\n"
                   "Metadata checked 2026-11-04.\n")
    (DATASETS / "Digits_Attribution.txt").write_text(attribution, encoding="utf-8")
    result.update({"synthetic": False, "dataset": "Digits built-in copy of original UCI test subset",
                   "pixel_range": [float(X.to_numpy().min()), float(X.to_numpy().max())],
                   "svm_errors": len(wrong_positions), "svm_error_examples": examples,
                   "svm_n_support": models["SVM"].named_steps["model"].n_support_.tolist()})
    return result

if __name__ == "__main__":
    result = {"moons": moons_experiment(), "digits": digits_experiment()}
    (Path(__file__).parent / "lab04_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
