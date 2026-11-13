"""ЗЖ 7: Grid/Random іздеуі, nested CV тек train ішінде, нақты UCI Wine."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import loguniform, norm
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    cross_val_score)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FIGURES, DATASETS = ROOT / "figures", ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
ATTRIBUTION = """Wine. Aeberhard, S. & Forina, M. (1992). UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5PC7J
Дереккөз: https://archive.ics.uci.edu/dataset/109/wine
Лицензия: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Көшірме sklearn.datasets.load_wine-нан экспортталған; 1-3 белгілері 0-2 ретінде берілген; баған атаулары scikit-learn-ге сәйкес.
Жасанды бақылаулар қосылмаған; қолжетімділік/метадеректер тексерілген күн: 2026-11-04.
"""


def make_split():
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    Xa, Xb, ya, yb = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=2026)
    return {"X": X, "y": y, "train": (Xa, ya), "test": (Xb, yb)}


def make_searches():
    pipe = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=2027)
    shared = dict(scoring="f1_macro", cv=inner, n_jobs=1, refit=True,
                  error_score="raise")
    return {
        "Grid": GridSearchCV(pipe, {
            "model__C": [.01, 1, 100], "model__gamma": [.001, .01, .1, 1]},
            **shared),
        "Random": RandomizedSearchCV(pipe, {
            "model__C": loguniform(.01, 100),
            "model__gamma": loguniform(.001, 1)},
            n_iter=12, random_state=2027, **shared)}


def nested_assessment(split, searches):
    Xa, ya = split["train"]
    outer = list(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=2028).split(Xa, ya))
    output = {}
    for name, template in searches.items():
        rows = []
        for fold, (train_index, validation_index) in enumerate(outer, 1):
            search = clone(template)
            search.fit(Xa.iloc[train_index], ya.iloc[train_index])
            prediction = search.predict(Xa.iloc[validation_index])
            rows.append({
                "outer_fold": fold, "outer_train_size": len(train_index),
                "outer_validation_size": len(validation_index),
                "outer_macro_F1": f1_score(ya.iloc[validation_index], prediction,
                                          average="macro", zero_division=0),
                "best_inner_macro_F1": float(search.best_score_),
                "best_params": {k: float(v) for k, v in search.best_params_.items()},
                "refit_scaler_rows": int(search.best_estimator_.named_steps[
                    "scale"].n_samples_seen_),
                "outer_train_indices": Xa.iloc[train_index].index.tolist(),
                "outer_validation_indices": Xa.iloc[validation_index].index.tolist(),
                "candidate_count": len(search.cv_results_["params"]),
                "Pipeline_fits": len(search.cv_results_["params"]) * 3 + 1})
        output[name] = {"outer_rows": rows,
                        "mean_outer_macro_F1": float(np.mean([
                            row["outer_macro_F1"] for row in rows])),
                        "Pipeline_fits": sum(row["Pipeline_fits"] for row in rows)}
        print(name, "nested сыртқы macro-F1:",
              [row["outer_macro_F1"] for row in rows],
              "орташа:", output[name]["mean_outer_macro_F1"])
    return output


def fit_main(split, searches):
    Xa, ya = split["train"]
    fitted, summaries = {}, {}
    for name, template in searches.items():
        search = clone(template).fit(Xa, ya)
        fitted[name] = search
        scores = search.cv_results_
        candidates = [{"candidate_index": j,
                       "C": float(params["model__C"]),
                       "gamma": float(params["model__gamma"]),
                       "mean_inner_macro_F1": float(scores["mean_test_score"][j]),
                       "inner_fold_macro_F1": [float(scores[f"split{k}_test_score"][j])
                                                for k in range(3)],
                       "rank": int(scores["rank_test_score"][j])}
                      for j, params in enumerate(scores["params"])]
        top_indices = np.flatnonzero(scores["rank_test_score"] == 1).tolist()
        assert search.best_index_ == min(top_indices)
        summaries[name] = {
            "best_params": {k: float(v) for k, v in search.best_params_.items()},
            "best_inner_macro_F1": float(search.best_score_),
            "best_index": int(search.best_index_), "rank1_indices": top_indices,
            "candidates": candidates, "candidate_count": len(candidates),
            "Pipeline_fits": len(candidates) * 3 + 1,
            "refit_scaler_rows": int(search.best_estimator_.named_steps[
                "scale"].n_samples_seen_)}
        print(name, "негізгі ең жақсы параметрлер:", summaries[name]["best_params"],
              "ең жақсы ішкі баға:", search.best_score_, "бірінші ранг индекстері:", top_indices)
    return {"fitted": fitted, "summaries": summaries}


def adaptive_ei(Xtrain, ytrain):
    """Жеке бірөлшемді бейімделмелі мысал; test аргументі және оған қолжетімділік жоқ."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=2027)
    observed_x, observed_y, trail = [], [], []
    counters = {"objective_calls": 0, "CV_Pipeline_fits": 0,
                "final_Pipeline_fits": 0, "surrogate_fit_calls": 0}

    def evaluate(x, phase, expected_improvement=None):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", SVC(kernel="rbf", C=10 ** x, gamma=.01))])
        scores = cross_val_score(pipe, Xtrain, ytrain, cv=inner, scoring="f1_macro",
                                 n_jobs=1, error_score="raise")
        counters["objective_calls"] += 1
        counters["CV_Pipeline_fits"] += len(scores)
        score = float(scores.mean())
        observed_x.append(float(x)); observed_y.append(score)
        trail.append({"phase": phase, "x": float(x), "C": float(10 ** x),
                      "gamma": .01, "mean_CV_macro_F1": score,
                      "fold_macro_F1": scores.tolist(),
                      "EI_at_proposal": expected_improvement})

    for x in [-2., 0., 2.]:
        evaluate(x, "initial")
    proposal_grid = np.linspace(-2, 2, 201)
    for step in range(6):
        kernel = ConstantKernel(1., constant_value_bounds="fixed") * Matern(
            length_scale=1., length_scale_bounds="fixed", nu=2.5)
        surrogate = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, alpha=1e-6, normalize_y=True,
            random_state=2027)
        surrogate.fit(np.array(observed_x)[:, None], np.array(observed_y))
        counters["surrogate_fit_calls"] += 1
        unseen = proposal_grid[~np.any(np.isclose(
            proposal_grid[:, None], np.array(observed_x)[None, :],
            atol=1e-12, rtol=0), axis=1)]
        mu, sigma = surrogate.predict(unseen[:, None], return_std=True)
        improvement = mu - max(observed_y)
        ei = np.maximum(improvement, 0.)
        positive_sigma = sigma > 0
        z = improvement[positive_sigma] / sigma[positive_sigma]
        ei[positive_sigma] = (improvement[positive_sigma] * norm.cdf(z)
                              + sigma[positive_sigma] * norm.pdf(z))
        index = int(np.argmax(ei))
        evaluate(float(unseen[index]), "adaptive", float(ei[index]))
    best_index = int(np.argmax(observed_y))
    best_c = 10 ** observed_x[best_index]
    final_model = Pipeline([("scale", StandardScaler()),
                            ("model", SVC(kernel="rbf", C=best_c, gamma=.01))])
    final_model.fit(Xtrain, ytrain)
    counters["final_Pipeline_fits"] += 1
    output = {"data": "тек 142 train", "test_used": False, "dimension": 1,
              "x_definition": "log10(C)", "bounds": [-2, 2], "fixed_gamma": .01,
              "initial_x": [-2, 0, 2], "adaptive_steps": 6,
              "proposal_grid_points": 201, "trail": trail,
              "best_observed_index": best_index, "best_C": float(best_c),
              "best_observed_CV_macro_F1": float(observed_y[best_index]),
              "counters": counters,
              "total_Pipeline_fits": counters["CV_Pipeline_fits"] + counters["final_Pipeline_fits"],
              "final_scaler_rows": int(final_model.named_steps["scale"].n_samples_seen_),
              "kernel": "ConstantKernel(1,fixed)*Matern(length_scale=1,fixed,nu=2.5)",
              "optimizer": None, "alpha": 1e-6, "normalize_y": True,
              "proposal_tie_rule": "бақыланбаған өспелі x торындағы бірінші EI максимумы",
              "best_observed_tie_rule": "бірінші бақыланған максималды CV бағасы"}
    print("Жеке бірөлшемді EI, тек train:\n", pd.DataFrame(trail).to_string(index=False))
    print("EI бойынша ең жақсы C:", best_c, "бақыланған CV:", observed_y[best_index],
          "Pipeline оқыту саны:", counters["CV_Pipeline_fits"] + counters["final_Pipeline_fits"])
    return output


def plot_searches(state):
    grid = state["summaries"]["Grid"]
    cs, gammas = np.array([.01, 1, 100]), np.array([.001, .01, .1, 1])
    values = np.array([row["mean_inner_macro_F1"] for row in grid["candidates"]])
    values = values.reshape(3, 4)

    def edges(values):
        logs = np.log10(values)
        return 10 ** np.r_[logs[0] - (logs[1] - logs[0]) / 2,
                           (logs[:-1] + logs[1:]) / 2,
                           logs[-1] + (logs[-1] - logs[-2]) / 2]

    norm = Normalize(0, 1)
    ce, ge = edges(cs), edges(gammas)
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    mesh = ax.pcolormesh(ge, ce, values, norm=norm, cmap="viridis", shading="flat")
    for j, c in enumerate(cs):
        for k, gamma in enumerate(gammas):
            ax.text(gamma, c, f"{values[j,k]:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if values[j,k] < .65 else "black")
    j, k = divmod(grid["best_index"], 4)
    ax.add_patch(Rectangle((ge[k], ce[j]), ge[k+1]-ge[k], ce[j+1]-ce[j],
                           fill=False, edgecolor="#E66B13", linewidth=3))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xticks=gammas, yticks=cs, xlabel="gamma (лог)", ylabel="C (лог)",
           title="Grid: үш ішкі фолдтың орташа macro-F1-і")
    fig.colorbar(mesh, ax=ax, label="Орташа ішкі macro-F1")
    fig.savefig(FIGURES / "ch07_05_grid.png", dpi=180); plt.close(fig)
    random = state["summaries"]["Random"]
    rows = random["candidates"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    points = ax.scatter([r["gamma"] for r in rows], [r["C"] for r in rows],
                        c=[r["mean_inner_macro_F1"] for r in rows],
                        norm=norm, cmap="viridis", s=95, edgecolors="black")
    for row in rows:
        ax.annotate(str(row["candidate_index"] + 1), (row["gamma"], row["C"]),
                    xytext=(-12, -11) if row["candidate_index"] == 1 else (5, 5),
                    textcoords="offset points", fontsize=8)
    best = rows[random["best_index"]]
    ax.scatter([best["gamma"]], [best["C"]], s=260, facecolors="none",
                edgecolors="#E66B13", linewidths=2, label="Ішкі CV бойынша таңдау")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlim=(.00075, 1.2), ylim=(.008, 150), xlabel="gamma (лог)", ylabel="C (лог)",
           title="Random: 12 кандидат, seed=2027")
    ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(points, ax=ax, label="Орташа ішкі macro-F1")
    fig.savefig(FIGURES / "ch07_06_random.png", dpi=180); plt.close(fig)


def evaluate_test(split, state):
    Xa, ya = split["train"]
    Xb, yb = split["test"]
    models = {"DummyPrior": DummyClassifier(strategy="prior").fit(Xa, ya),
              **state["fitted"]}
    rows = []
    for name, model in models.items():
        prediction = model.predict(Xb)
        rows.append({"model": name, "Accuracy": accuracy_score(yb, prediction),
                     "macro_F1": f1_score(yb, prediction, average="macro", zero_division=0),
                     "confusion_matrix": confusion_matrix(yb, prediction,
                                                          labels=[0, 1, 2]).tolist()})
    print("Екі стратегия бекітілгеннен кейінгі бір test:\n",
          pd.DataFrame(rows).to_string(index=False))
    return rows


def save_data_and_results(split, nested, state, test_rows):
    Xa, ya = split["train"]
    Xb, yb = split["test"]
    service = pd.Series("test", index=split["X"].index)
    service.loc[Xa.index] = "train"
    split["X"].assign(target=split["y"], split=service).to_csv(
        DATASETS / "wine_search.csv", index=False)
    attribution = DATASETS / "Wine_Attribution.txt"
    if not attribution.exists():
        attribution.write_text(ATTRIBUTION, encoding="utf-8")
    return {"dataset": "нақты UCI Wine", "license": "CC BY 4.0", "shape": [178, 13],
            "sizes": [len(ya), len(yb)], "split_seed": 2026,
            "inner_CV": {"n_splits": 3, "shuffle": True, "seed": 2027},
            "outer_CV": {"n_splits": 3, "shuffle": True, "seed": 2028,
                         "data": "тек 142 train"},
            "train_class_counts": np.bincount(ya, minlength=3).tolist(),
            "test_class_counts": np.bincount(yb, minlength=3).tolist(),
            "train_indices": Xa.index.tolist(), "test_indices": Xb.index.tolist(),
            "nested": nested, "main_search": state["summaries"], "test_rows": test_rows,
            "tie_rule": "cv_results_ ретіндегі rank1 арасындағы бірінші кандидат",
            "strategies_predeclared": ["Grid", "Random"], "selected_strategy": None,
            "test_used_to_select": False,
            "Pipeline_fits_per_strategy": {"main": 37, "nested": 111, "total": 148},
            "all_search_Pipeline_fits": 296,
            "fit_unit": "бір Pipeline.fit; SVC-дің бір жұптық екілік ішкі есебі емес"}


if __name__ == "__main__":
    split = make_split()
    searches = make_searches()
    nested = nested_assessment(split, searches)
    state = fit_main(split, searches)
    adaptive = adaptive_ei(*split["train"])
    plot_searches(state)
    test_rows = evaluate_test(split, state)
    result = save_data_and_results(split, nested, state, test_rows)
    result["adaptive_EI"] = adaptive
    (Path(__file__).parent / "lab07_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
