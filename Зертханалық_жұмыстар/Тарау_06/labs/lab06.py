"""№6 ЗЖ: толық синтетикалық екі тәжірибе, CV және бекітілген класс салмақтары."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score, brier_score_loss)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
FIGURES, DATASETS = ROOT / "figures", ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
TOLERANCE = 1e-8


def make_regression_split():
    raw, target, truth = make_regression(
        n_samples=240, n_features=30, n_informative=5,
        noise=20, random_state=2026, coef=True)
    X = pd.DataFrame(raw, columns=[f"x{j}" for j in range(30)])
    y = pd.Series(target, name="target")
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, random_state=2026)
    return {"X": X, "y": y, "truth": truth,
            "train": (Xa, ya), "test": (Xb, yb)}


def make_rare_split():
    raw, target = make_classification(
        n_samples=1200, n_features=10, n_informative=5, n_redundant=2,
        weights=[.95, .05], class_sep=1.5, flip_y=.01, random_state=2026)
    X = pd.DataFrame(raw, columns=[f"x{j}" for j in range(10)])
    y = pd.Series(target, name="target")
    Xa, Xb, ya, yb = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=2026)
    return {"X": X, "y": y, "train": (Xa, ya), "test": (Xb, yb)}


def fit_regression(split):
    Xa, ya = split["train"]
    cv = list(KFold(n_splits=5, shuffle=True, random_state=2027).split(Xa))
    grids = {"Ridge": (Ridge(), [.01, .1, 1, 10, 100]),
             "Lasso": (Lasso(max_iter=10000), [.01, .1, 1, 10])}
    models = {"DummyMean": DummyRegressor(strategy="mean"),
              "OLS": Pipeline([("scale", StandardScaler()),
                               ("model", LinearRegression())])}
    for model in models.values():
        model.fit(Xa, ya)
    searches, paths = {}, {}
    for name, (estimator, alphas) in grids.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        search = GridSearchCV(
            pipe, {"model__alpha": alphas}, cv=cv, refit=True, n_jobs=1,
            scoring="neg_root_mean_squared_error", error_score="raise")
        search.fit(Xa, ya)
        searches[name], models[name] = search, search.best_estimator_
        scores = search.cv_results_
        paths[name] = [{
            "alpha": float(alpha),
            "mean_CV_RMSE": float(-scores["mean_test_score"][j]),
            "fold_RMSE": [float(-scores[f"split{k}_test_score"][j])
                          for k in range(5)],
        } for j, alpha in enumerate(alphas)]
        print(name, "CV бойынша alpha:", search.best_params_["model__alpha"],
              "орташа CV RMSE:", -search.best_score_)
    return {"models": models, "searches": searches, "paths": paths,
            "CV_indices": cv}


def evaluate_regression(split, state):
    Xa, ya = split["train"]
    Xb, yb = split["test"]
    rows, coefficients = [], {}
    for name, model in state["models"].items():
        prediction = model.predict(Xb)
        coef = (model.named_steps["model"].coef_
                if name != "DummyMean" else np.array([]))
        coefficients[name] = coef.tolist()
        rows.append({"model": name,
                     "MAE": mean_absolute_error(yb, prediction),
                     "RMSE": root_mean_squared_error(yb, prediction),
                     "R2": r2_score(yb, prediction),
                     "practical_nonzero": int(np.count_nonzero(
                         np.abs(coef) > TOLERANCE)) if coef.size else None})
    print("Регрессияның бір test бөлігі, 48 жол:\n",
          pd.DataFrame(rows).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), layout="constrained")
    for ax, name, note in zip(axes, ["Ridge", "Lasso"], ["α=nλ", "α=λ"]):
        path = state["paths"][name]
        ax.semilogx([r["alpha"] for r in path],
                    [r["mean_CV_RMSE"] for r in path], "o-", color="#176B87")
        search = state["searches"][name]
        ax.scatter([search.best_params_["model__alpha"]],
                   [-search.best_score_], marker="*", s=150,
                   color="#B34B20", zorder=3, label="CV бойынша таңдау")
        ax.set(xlabel=f"Кітапханалық α ({note})", ylabel="Орташа CV RMSE",
               title=f"{name}: train ішіндегі 5 бүктеме")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(FIGURES / "ch06_05_regularization.png", dpi=180)
    plt.close(fig)
    service = pd.Series("test", index=split["X"].index)
    service.loc[Xa.index] = "train"
    split["X"].assign(target=split["y"], split=service).to_csv(
        DATASETS / "synthetic_regularization_regression.csv", index=False)
    return {"synthetic": True, "rows": 240, "features": 30,
            "n_informative": 5, "noise": 20, "sizes": [len(ya), len(yb)],
            "generator_seed": 2026, "split_seed": 2026,
            "CV_seed": 2027, "CV_folds": 5,
            "common_CV_splits": True, "Pipeline_inside_GridSearchCV": True,
            "CV_paths": state["paths"],
            "selected_alpha": {name: float(s.best_params_["model__alpha"])
                               for name, s in state["searches"].items()},
            "test_rows": rows, "coefficient_tolerance": TOLERANCE,
            "standardized_coefficients": coefficients,
            "true_coefficients_raw_units": split["truth"].tolist(),
            "train_indices": Xa.index.tolist(), "test_indices": Xb.index.tolist(),
            "splits_disjoint": not bool(set(Xa.index) & set(Xb.index)),
            "final_scaler_fit_rows": {
                name: int(m.named_steps["scale"].n_samples_seen_)
                for name, m in state["models"].items() if name != "DummyMean"}}


def fit_rare(split):
    Xa, ya = split["train"]
    models = {"DummyPrior": DummyClassifier(strategy="prior")}
    for name, weight in [("Unweighted", None), ("Balanced", "balanced")]:
        models[name] = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1, max_iter=2000, class_weight=weight))])
    for model in models.values():
        model.fit(Xa, ya)
    counts = np.bincount(ya, minlength=2)
    weights = len(ya) / (2 * counts)
    np.testing.assert_allclose(weights, compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=ya))
    print("Train сандары:", counts, "balanced салмақтары:", weights)
    return {"models": models, "train_counts": counts, "class_weights": weights}


def evaluate_rare(split, state):
    Xa, ya = split["train"]
    Xb, yb = split["test"]
    rows, probabilities = [], {}
    for name, model in state["models"].items():
        column = int(np.flatnonzero(model.classes_ == 1)[0])
        probability = model.predict_proba(Xb)[:, column]
        probabilities[name] = probability
        prediction = (probability >= .5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, prediction, labels=[0, 1]).ravel()
        rows.append({"model": name, "TN": int(tn), "FP": int(fp),
                     "FN": int(fn), "TP": int(tp),
                     "Precision": precision_score(yb, prediction, zero_division=0),
                     "Recall": recall_score(yb, prediction, zero_division=0),
                     "F1": f1_score(yb, prediction, zero_division=0),
                     "Accuracy": accuracy_score(yb, prediction),
                     "AP": average_precision_score(yb, probability),
                     "ROC_AUC": roc_auc_score(yb, probability),
                     "Brier": brier_score_loss(yb, probability, pos_label=1,
                                               scale_by_half=True),
                     "cost_FP1_FN5": int(fp + 5 * fn)})
    print("Сирек оқиғаның бір test бөлігі, шек .5:\n",
          pd.DataFrame(rows).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), layout="constrained")
    x = np.arange(3)
    for key, offset, color in [("FP", -.18, "#176B87"),
                                ("FN", .18, "#B34B20")]:
        bars = axes[0].bar(x + offset, [r[key] for r in rows],
                           width=.36, label=key, color=color)
        axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set(xticks=x, xticklabels=["Dummy", "Салмақсыз", "Balanced"],
                ylabel="Test қателерінің саны", title="Бекітілген шек 0,5")
    axes[0].set_ylim(0, max(max(r["FP"], r["FN"]) for r in rows) * 1.22 + 1)
    axes[0].legend(fontsize=8)
    for label, color, marker in [(0, "#176B87", "o"), (1, "#B34B20", "x")]:
        mask = yb.to_numpy() == label
        axes[1].scatter(probabilities["Unweighted"][mask],
                        probabilities["Balanced"][mask], s=24 if label else 13,
                        alpha=.9 if label else .5, color=color, marker=marker,
                        label=f"Класс {label}: n={mask.sum()}")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].axhline(.5, color="grey", linewidth=.8, linestyle=":")
    axes[1].axvline(.5, color="grey", linewidth=.8, linestyle=":")
    axes[1].set(xlabel="p(1) бағасы, салмақсыз", ylabel="p(1) бағасы, balanced",
                title="Бір объект — бір test нүктесі", xlim=(-.03, 1.03),
                ylim=(-.03, 1.03))
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.savefig(FIGURES / "ch06_06_rare_events.png", dpi=180)
    plt.close(fig)
    service = pd.Series("test", index=split["X"].index)
    service.loc[Xa.index] = "train"
    split["X"].assign(target=split["y"], split=service).to_csv(
        DATASETS / "synthetic_rare_events.csv", index=False)
    return {"synthetic": True, "rows": 1200, "features": 10,
            "n_informative": 5, "n_redundant": 2, "nominal_weights": [.95, .05],
            "class_sep": 1.5, "flip_y": .01, "seed": 2026,
            "sizes": [len(ya), len(yb)],
            "train_counts": state["train_counts"].tolist(),
            "test_counts": np.bincount(yb, minlength=2).tolist(),
            "balanced_train_weights": state["class_weights"].tolist(),
            "threshold": .5, "test_rows": rows,
            "test_probabilities": {k: p.tolist() for k, p in probabilities.items()},
            "test_y": yb.tolist(),
            "train_indices": Xa.index.tolist(), "test_indices": Xb.index.tolist(),
            "splits_disjoint": not bool(set(Xa.index) & set(Xb.index)),
            "class_weight_origin": "960 train targets only",
            "test_rebalanced": False, "resampling_performed": False,
            "final_scaler_fit_rows": {
                name: int(m.named_steps["scale"].n_samples_seen_)
                for name, m in state["models"].items() if name != "DummyPrior"}}


if __name__ == "__main__":
    regression_split = make_regression_split()
    rare_split = make_rare_split()
    regression_state = fit_regression(regression_split)
    regression = evaluate_regression(regression_split, regression_state)
    rare_state = fit_rare(rare_split)
    rare_events = evaluate_rare(rare_split, rare_state)
    (Path(__file__).parent / "lab06_results.json").write_text(
        json.dumps({"regression": regression, "rare_events": rare_events},
                   ensure_ascii=False, indent=2), encoding="utf-8")
