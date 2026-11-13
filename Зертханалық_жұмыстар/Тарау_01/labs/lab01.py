'ЗЖ 1. Iris және синтетикалық жүктеме регрессиясы, желісіз орындалады.'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)

plt.rcParams["font.family"] = "DejaVu Sans"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def iris_experiment():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))])}
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "F1macro": f1_score(yte, pred, average="macro")})
        matrices[name] = confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 4.3), layout="constrained")
    matrix = np.array(matrices["LogisticRegression"])
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=data.target_names,
           yticklabels=data.target_names, xlabel='Болжам', ylabel='Нақты класс')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > 5 else "black")
    fig.savefig(FIGURES / "ch01_05_iris.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    return {"train_size": len(ytr), "test_size": len(yte),
            "metrics": rows, "confusion_matrices": matrices}

def regression_experiment():
    rng = np.random.default_rng(2026)
    n = 500
    temperature = rng.uniform(-5, 30, n)
    occupancy = rng.uniform(0, 1, n)
    X = pd.DataFrame({"temperature_C": temperature, "occupancy": occupancy})
    y = 120 - 2.5 * temperature + 45 * occupancy + rng.normal(0, 6, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=2026)
    models = {"Dummy": DummyRegressor(strategy="mean"),
              "LinearRegression": LinearRegression()}
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                     "R2": r2_score(yte, pred)})
        predictions[name] = pred
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    pred = predictions["LinearRegression"]
    limits = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    axs[0].scatter(yte, pred, s=15, alpha=0.7)
    axs[0].plot(limits, limits, "k--", linewidth=1)
    axs[0].set(xlabel='Нақты жүктеме, шартты бірлік', ylabel='Болжам, шартты бірлік')
    axs[1].scatter(pred, yte - pred, s=15, alpha=0.7)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[1].set(xlabel='Болжам, шартты бірлік', ylabel='Қалдық, шартты бірлік')
    fig.savefig(FIGURES / "ch01_06_regression.png", dpi=180)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    return {"synthetic": True, "seed": 2026, "train_size": len(ytr),
            "test_size": len(yte), "metrics": rows,
            "coefficients": models["LinearRegression"].coef_.tolist(),
            "intercept": float(models["LinearRegression"].intercept_)}

if __name__ == "__main__":
    results = {"iris": iris_experiment(), "synthetic_load": regression_experiment()}
    (Path(__file__).parent / "lab01_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
