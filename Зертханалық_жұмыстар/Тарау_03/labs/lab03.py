"""№3 зертханалық жұмыс: синтетикалық регрессия және ашық UCI Wine жиынтығы."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, log_loss, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

def polynomial_experiment():
    rng = np.random.default_rng(2026)
    X = pd.DataFrame({"x": rng.uniform(-3, 3, 500)})
    x = X["x"].to_numpy()
    y = 4 + 2*x + 1.5*x**2 + rng.normal(0, 1.2, 500)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=2026)
    models = {"Dummy": DummyRegressor(strategy="mean"),
              "Linear": LinearRegression(),
              "Polynomial2": Pipeline([
                  ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("scale", StandardScaler()), ("model", LinearRegression())])}
    rows = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)), "R2": r2_score(yte, pred)})
    scaler, reg = models["Polynomial2"].named_steps["scale"], models["Polynomial2"].named_steps["model"]
    coefficient = reg.coef_ / scaler.scale_
    intercept = float(reg.intercept_ - np.dot(reg.coef_, scaler.mean_ / scaler.scale_))
    probe = pd.DataFrame({"x": [-2., 0., 2.]})
    predictions = {name: model.predict(probe).tolist() for name, model in models.items()}
    print(pd.DataFrame(rows).to_string(index=False))
    print("Сызықтық модель: b,w", models["Linear"].intercept_, models["Linear"].coef_)
    print("Бастапқы базистегі квадраттық модель: b,w1,w2", intercept, *coefficient)
    print("x=-2,0,2 кезіндегі болжамдар", predictions)
    grid = pd.DataFrame({"x": np.linspace(-3, 3, 250)})
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    ax.scatter(Xtr["x"], ytr, s=10, color="lightgray", label="Оқыту", alpha=.65)
    ax.scatter(Xte["x"], yte, s=16, color="#2469A0", label="Тест", alpha=.7)
    for name, style in [("Dummy", ":"), ("Linear", "--"), ("Polynomial2", "-")]:
        ax.plot(grid["x"], models[name].predict(grid), style, linewidth=2, label=name)
    ax.set(xlabel="x, шартты бірлік", ylabel="y, шартты бірлік")
    ax.legend(ncol=2)
    fig.savefig(FIGURES / "ch03_05_polynomial.png", dpi=180)
    plt.close(fig)
    X.assign(y_synthetic=y).to_csv(DATASETS / "synthetic_quadratic_regression.csv", index=False)
    return {"synthetic": True, "seed": 2026, "rows": 500, "train_size": len(ytr),
            "test_size": len(yte), "metrics": rows,
            "linear_coefficients": {"intercept": float(models["Linear"].intercept_),
                                    "x": float(models["Linear"].coef_[0])},
            "polynomial_original_basis": {"intercept": intercept, "x": float(coefficient[0]),
                                          "x_squared": float(coefficient[1])},
            "polynomial_standardized_coefficients": reg.coef_.tolist(),
            "polynomial_standardized_intercept": float(reg.intercept_),
            "polynomial_scaler_mean": scaler.mean_.tolist(),
            "polynomial_scaler_scale": scaler.scale_.tolist(),
            "probe_x": probe["x"].tolist(), "probe_predictions": predictions,
            "probe_generator_mean": [6., 4., 14.]}

def wine_experiment():
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=2026, stratify=y)
    models = {"DummyPrior": DummyClassifier(strategy="prior"),
              "Logistic": Pipeline([("scale", StandardScaler()),
                                     ("model", LogisticRegression(max_iter=2000))])}
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred, proba = model.predict(Xte), model.predict_proba(Xte)
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "F1macro": f1_score(yte, pred, average="macro"),
                     "log_loss": log_loss(yte, proba, labels=model.classes_)})
        matrices[name] = confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist()
    model = models["Logistic"]
    reg, scaler = model.named_steps["model"], model.named_steps["scale"]
    positions = [int(np.flatnonzero(yte.to_numpy()==k)[0]) for k in range(3)]
    probe = Xte.iloc[positions]
    predicted, probabilities = model.predict(probe), model.predict_proba(probe)
    examples = [{"row_id": int(probe.index[i]), "true_class": int(yte.iloc[positions[i]]),
                 "predicted_class": int(predicted[i]), "probabilities": probabilities[i].tolist()}
                for i in range(3)]
    alcohol_index = data.feature_names.index("alcohol")
    contrast = float(reg.coef_[0, alcohol_index] - reg.coef_[1, alcohol_index])
    print("Wine:", X.shape, "кластар", y.value_counts().sort_index().to_dict())
    print(pd.DataFrame(rows).to_string(index=False))
    print("Logistic матрицасы", matrices["Logistic"])
    print(pd.DataFrame(reg.coef_.T, index=data.feature_names, columns=["class0", "class1", "class2"]).round(6))
    print("Ықтималдық мысалдары", examples)
    print("alcohol0-1 айырмасы және exp(айырма)", contrast, np.exp(contrast))
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 4.1), layout="constrained")
    matrix = np.array(matrices["Logistic"])
    axs[0].imshow(matrix, cmap="Blues", vmin=0)
    axs[0].set(xticks=range(3), yticks=range(3), xlabel="Болжанған класс", ylabel="Нақты класс")
    for i in range(3):
        for j in range(3):
            axs[0].text(j, i, str(matrix[i,j]), ha="center", va="center",
                        color="white" if matrix[i,j]>matrix.max()/2 else "black")
    bottom = np.zeros(3)
    for k, color in enumerate(["#2469A0", "#E48B30", "#7C5BB6"]):
        axs[1].bar(range(3), probabilities[:,k], bottom=bottom, color=color, label=f"Класс {k}")
        bottom += probabilities[:,k]
    axs[1].set(xticks=range(3), xticklabels=[f"№{r['row_id']}\nнақты {r['true_class']}" for r in examples],
               ylim=(0,1.05), ylabel="Болжанған ықтималдық")
    axs[1].legend(loc="upper center", bbox_to_anchor=(.5,1.16), ncol=3, fontsize=8)
    fig.savefig(FIGURES / "ch03_06_logistic.png", dpi=180)
    plt.close(fig)
    data.frame.to_csv(DATASETS / "wine_sklearn.csv", index=False)
    attribution = ("Wine. Aeberhard, S. & Forina, M. (1992). UCI Machine Learning Repository.\n"
                   "DOI: https://doi.org/10.24432/C5PC7J\n"
                   "Source: https://archive.ics.uci.edu/dataset/109/wine\n"
                   "License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/\n"
                   "Copy exported from sklearn.datasets.load_wine; labels 1-3 represented as 0-2; column names follow scikit-learn.\n"
                   "No synthetic observations added; access/metadata check: 2026-11-04.\n")
    (DATASETS / "Wine_Attribution.txt").write_text(attribution, encoding="utf-8")
    return {"synthetic": False, "dataset": "UCI Wine", "rows": len(X), "features": X.shape[1],
            "class_counts": y.value_counts().sort_index().tolist(), "train_size": len(ytr), "test_size": len(yte),
            "train_class_counts": ytr.value_counts().sort_index().tolist(),
            "test_class_counts": yte.value_counts().sort_index().tolist(),
            "metrics": rows, "confusion_matrices": matrices, "feature_names": data.feature_names,
            "standardized_coefficients": reg.coef_.tolist(), "standardized_intercepts": reg.intercept_.tolist(),
            "original_coefficients": (reg.coef_/scaler.scale_).tolist(),
            "original_intercepts": (reg.intercept_-reg.coef_@(scaler.mean_/scaler.scale_)).tolist(),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "examples": examples, "alcohol_contrast_class0_class1": contrast,
            "alcohol_odds_ratio_per_train_sd": float(np.exp(contrast)),
            "probabilities_sum_to_one": bool(np.allclose(probabilities.sum(axis=1),1))}

if __name__ == "__main__":
    result = {"polynomial": polynomial_experiment(), "wine": wine_experiment()}
    (Path(__file__).parent / "lab03_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
