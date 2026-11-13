"""ЗЖ 9. Бустинг кезеңдерін таңдау және бекітілген жіктеуіштерді салыстыру."""
from pathlib import Path
from importlib.metadata import version
import json
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_friedman1
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
ATTRIBUTION = """Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5DW2B
Дереккөз: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Лицензия: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
scikit-learn load_breast_cancer кірістірілген көшірмесінен экспорт.
Белгі атауларындағы бос орындар астын сызу таңбаларына ауыстырылған. Мақсатты айнымалы sklearn target 0 үшін malignant=1
және sklearn target 1 үшін malignant=0 ретінде қайта кодталған. row_id және split —
осы сабаққа қосылған қызметтік өрістер. Метадеректер 2026-11-05 күні тексерілген.
"""
PACKAGES = ("numpy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl",
            "xgboost", "lightgbm", "catboost", "IPython")
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def make_regression_data():
    X, y = make_friedman1(n_samples=600, n_features=8, noise=1.5, random_state=2026)
    development, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=2026)
    train, validation = train_test_split(development, test_size=.25, random_state=2027)
    return {"X": X, "y": y, "train": train, "validation": validation,
            "test": test, "development": development}


def select_regression_stage(data):
    X, y = data["X"], data["y"]
    train, validation = data["train"], data["validation"]
    model = GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                      learning_rate=.05, random_state=2026)
    with threadpool_limits(limits=1):
        model.fit(X[train], y[train])
        curve = [root_mean_squared_error(y[validation], prediction)
                 for prediction in model.staged_predict(X[validation])]
    best_n = int(np.argmin(curve)) + 1
    print("1, таңдалған және 150 кезеңдегі валидациялық RMSE:", curve[0], curve[best_n - 1], curve[-1])
    print("Таңдалған кезеңдер:", best_n)
    return {"best_n": best_n, "validation_curve": curve, "fit_rows": len(train),
            "estimator_parameters": model.get_params()}


def evaluate_regression(data, selection):
    X, y = data["X"], data["y"]
    development, test = data["development"], data["test"]
    best_n = selection["best_n"]
    final = GradientBoostingRegressor(n_estimators=best_n, max_depth=2,
                                      learning_rate=.05, random_state=2026)
    with threadpool_limits(limits=1):
        final.fit(X[development], y[development])
        prediction = final.predict(X[test])
    baseline_mean = float(y[development].mean())
    baseline = np.full(len(test), baseline_mean)
    rows = [{"model": name, "stages": stages,
             "RMSE": float(root_mean_squared_error(y[test], predicted)),
             "MAE": float(mean_absolute_error(y[test], predicted))}
            for name, stages, predicted in [("Train+validation орташа мәні", 0, baseline),
                                             ("Gradient Boosting", best_n, prediction)]]
    print(pd.DataFrame(rows).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1), layout="constrained")
    curve = selection["validation_curve"]
    axes[0].plot(range(1, len(curve) + 1), curve, color="#176B87")
    axes[0].scatter([best_n], [curve[best_n - 1]], s=48, color="#B34B20",
                    label=f"Таңдалды {best_n} кезең", zorder=3)
    axes[0].set(xlabel="Кезеңдер саны", ylabel="RMSE", title="validation бойынша таңдау",
                xlim=(-2, len(curve) + 4))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    low = min(y[test].min(), prediction.min())
    high = max(y[test].max(), prediction.max())
    axes[1].plot([low, high], [low, high], color="#777777", linestyle="--", label="Болжам = жауап")
    axes[1].scatter(y[test], prediction, s=19, alpha=.75, color="#176B87", label="Тест (120)")
    axes[1].set(xlabel="Бақыланған жауап", ylabel="Болжам", title="Қайта оқытудан кейінгі баға")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(FIGURES / "ch09_05_regression.png", dpi=180)
    plt.close(fig)
    return {"shape": [600, 8], "synthetic": True, "noise": 1.5,
            "generator_seed": 2026, "test_split_seed": 2026, "validation_split_seed": 2027,
            "train_indices": data["train"].tolist(), "validation_indices": data["validation"].tolist(),
            "test_indices": test.tolist(), "selection_fit_rows": selection["fit_rows"],
            "refit_rows": len(development), "selected_n": best_n, "refit_n": int(final.n_estimators_),
            "selection_estimator_parameters": selection["estimator_parameters"],
            "refit_estimator_parameters": final.get_params(),
            "validation_RMSE": curve, "selected_validation_RMSE": curve[best_n - 1],
            "tie_rule": "алғашқы минимум, демек кезеңдердің ең аз саны",
            "test_used_for_selection": False, "test_model_prediction_calls": 1,
            "baseline_mean": baseline_mean, "test_rows": rows,
            "test_y": y[test].tolist(), "test_predictions": prediction.tolist()}


def make_classification_data():
    data = load_breast_cancer()
    columns = [name.replace(" ", "_") for name in data.feature_names]
    X = pd.DataFrame(data.data, columns=columns)
    y = (data.target == 0).astype(int)
    train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=2026)
    return {"X": X, "y": y, "train": train, "test": test}


def fixed_classifiers():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    common = {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "random_state": 2026}
    return {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "Gradient Boosting": GradientBoostingClassifier(**common),
        "XGBoost": XGBClassifier(**common, n_jobs=1, tree_method="hist",
                                  eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(**common, num_leaves=8, n_jobs=1,
                                    deterministic=True, force_col_wise=True, verbosity=-1),
        "CatBoost": CatBoostClassifier(iterations=100, depth=3, learning_rate=.05,
                                        random_seed=2026, thread_count=1, verbose=False,
                                        loss_function="Logloss", allow_writing_files=False),
    }


def evaluate_classification(data, models):
    X, y = data["X"], data["y"]
    train, test = data["train"], data["test"]
    rows, probabilities = [], {}
    with threadpool_limits(limits=1):
        for model in models.values():
            model.fit(X.iloc[train], y[train])
        for name, model in models.items():
            column = int(np.flatnonzero(model.classes_ == 1)[0])
            prediction_options = {"thread_count": 1} if name == "CatBoost" else {}
            score = model.predict_proba(X.iloc[test], **prediction_options)[:, column]
            predicted = (score >= .5).astype(int)
            probabilities[name] = score.tolist()
            if name == "DummyPrior":
                stages = 0
            elif name == "XGBoost":
                stages = model.get_booster().num_boosted_rounds()
            elif name == "CatBoost":
                stages = model.tree_count_
            else:
                stages = model.n_estimators_
            rows.append({"model": name, "stages": int(stages),
                         "ROC_AUC": float(roc_auc_score(y[test], score)),
                         "macro_F1": float(f1_score(y[test], predicted, average="macro", zero_division=0))})
    print(pd.DataFrame(rows).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), layout="constrained")
    colors = ["#666666", "#176B87", "#B34B20", "#3A7D43", "#7A5195"]
    for row, color in zip(rows, colors):
        name = row["model"]
        fpr, tpr, _ = roc_curve(y[test], probabilities[name], pos_label=1)
        axes[0].plot(fpr, tpr, color=color, linestyle="--" if name == "DummyPrior" else "-",
                     linewidth=1.8, label=f"{name} ({row['ROC_AUC']:.4f})")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="114 тест жолындағы ROC",
                xlim=(-.025, 1.025), ylim=(-.025, 1.025))
    axes[0].legend(fontsize=7.2, loc="lower right")
    positions = np.arange(len(rows))
    scores = [row["macro_F1"] for row in rows]
    axes[1].barh(positions, scores, color=colors)
    for position, score in zip(positions, scores):
        axes[1].text(score + .015, position, f"{score:.4f}", va="center", fontsize=8)
    axes[1].set(yticks=positions, yticklabels=[row["model"] for row in rows],
                xlabel="Macro-F1", title="Бір шек: 0,5", xlim=(0, 1.13))
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=.15)
    fig.savefig(FIGURES / "ch09_06_classification.png", dpi=180)
    plt.close(fig)
    parameters = {name: model.get_params() for name, model in models.items()}
    # XGBoost әдепкі жетіспейтін мән белгісі ретінде NaN қолданады. Оны қатаң JSON ішінде сақтаймыз.
    parameters["XGBoost"]["missing"] = "NaN"
    return {"shape": [569, 30], "synthetic": False, "DOI": "10.24432/C5DW2B",
            "license": "CC BY 4.0", "positive_class": "malignant=1", "split_seed": 2026,
            "train_indices": train.tolist(), "test_indices": test.tolist(),
            "train_counts_0_1": np.bincount(y[train], minlength=2).tolist(),
            "test_counts_0_1": np.bincount(y[test], minlength=2).tolist(),
            "threshold": .5, "test_used_for_selection": False,
            "estimator_classes": {name: type(model).__name__ for name, model in models.items()},
            "estimator_parameters": parameters,
            "test_probability_calls_per_model": 1, "rows": rows,
            "test_y": y[test].tolist(), "test_probabilities": probabilities}


def save_classification_data(data):
    membership = np.full(len(data["y"]), "test", dtype=object)
    membership[data["train"]] = "train"
    data["X"].assign(row_id=np.arange(len(membership)), malignant=data["y"], split=membership).to_csv(
        DATASETS / "wdbc_boosting.csv", index=False)
    attribution = DATASETS / "WDBC_Attribution.txt"
    if not attribution.exists():
        attribution.write_text(ATTRIBUTION, encoding="utf-8")


def save_artifacts(data, regression, classification, result_path):
    save_classification_data(data)
    results = {"regression": regression, "classification": classification,
               "runtime": {"Python": platform.python_version(),
                           "packages": {name: version(name) for name in PACKAGES if name != "IPython"},
                           "device": "CPU",
                           "threadpoolctl_limit": 1,
                           "thread_environment": {name: os.getenv(name) for name in THREAD_ENVIRONMENT}}}
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    regression_data = make_regression_data()
    selection = select_regression_stage(regression_data)
    regression = evaluate_regression(regression_data, selection)
    classification_data = make_classification_data()
    classification = evaluate_classification(classification_data, fixed_classifiers())
    save_artifacts(classification_data, regression, classification,
                   Path(__file__).with_name("lab09_results.json"))
