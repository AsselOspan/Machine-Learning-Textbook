"""ЗЖ 8. Iris-тегі сипаттамалық кластерлер және кейінге қалдырылған PCA реконструкциясы."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
ATTRIBUTION = """Iris. Fisher, R. (1936). UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C56C76
Дереккөз: https://archive.ics.uci.edu/dataset/53/iris
Лицензия: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/
Экспортталған өлшемдер sklearn.datasets.load_iris көшірмесінен алынған.
scikit-learn 0.20 нұсқасынан бастап екі бақылау Фишер мақаласы бойынша түзетілген.
Бұл — түзетілген sklearn нұсқасы. Түр белгілері 0, 1, 2 ретінде кодталған.
Экспорт row_id, target, species және pca_split қызметтік бағандарын қосады.
Жасанды бақылаулар қосылмаған. Метадеректер 2026-11-05 күні тексерілген.
"""


def fit_clusters(X):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    models = {
        "KMeans": KMeans(n_clusters=3, n_init=20, random_state=2026),
        "GMM": GaussianMixture(n_components=3, covariance_type="full", n_init=10,
                               reg_covar=1e-6, max_iter=500, random_state=2026),
        "Ward": AgglomerativeClustering(n_clusters=3, linkage="ward", metric="euclidean"),
        "DBSCAN": DBSCAN(eps=.6, min_samples=5),
    }
    labels = {name: model.fit_predict(Z) for name, model in models.items()}
    return {"Z": Z, "labels": labels, "scaler": scaler, "models": models}


def fit_pca(X):
    train, test = train_test_split(np.arange(len(X)), test_size=.2, random_state=2026)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X.iloc[train])
    pca = PCA(n_components=2, svd_solver="full").fit(Z_train)
    Z_test = scaler.transform(X.iloc[test])
    coordinates = pca.transform(Z_test)
    reconstructed = pca.inverse_transform(coordinates)
    baseline = Z_train.mean(axis=0)
    mse = np.mean((Z_test - reconstructed) ** 2, axis=0)
    baseline_mse = np.mean((Z_test - baseline) ** 2, axis=0)
    return {"train": train, "test": test, "scaler": scaler, "pca": pca,
            "Z_train": Z_train, "Z_test": Z_test, "baseline": baseline,
            "coordinates": coordinates, "reconstructed": reconstructed,
            "mse": mse, "baseline_mse": baseline_mse}


def evaluate_clusters(X, y, state):
    rows = []
    for name, labels in state["labels"].items():
        keep = labels != -1
        n_kept = int(keep.sum())
        values, counts = np.unique(labels[keep], return_counts=True)
        k = len(values)
        silhouette = (float(silhouette_score(state["Z"][keep], labels[keep]))
                      if 2 <= k < n_kept else None)
        rows.append({"model": name, "clusters": k,
                     "cluster_sizes": dict(zip(map(str, values), map(int, counts))),
                     "covered": n_kept, "coverage": n_kept / len(X),
                     "noise": len(X) - n_kept, "silhouette": silhouette,
                     "silhouette_n": n_kept if silhouette is not None else None,
                     "ARI_all": float(adjusted_rand_score(y, labels))})
    display_cols = ["model", "clusters", "covered", "noise", "silhouette",
                    "silhouette_n", "ARI_all"]
    print(pd.DataFrame(rows)[display_cols].to_string(index=False))
    gmm = state["models"]["GMM"]
    print("GMM: жинақталды =", gmm.converged_, "итерациялар =", gmm.n_iter_)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.1), sharex=True, sharey=True,
                             layout="constrained")
    fig.suptitle("4D кластерлеу: екі белгі бойынша көрініс")
    colors = ["#176B87", "#B34B20", "#3A7D43", "#7A5195"]
    for ax, row in zip(axes.ravel(), rows):
        labels = state["labels"][row["model"]]
        for value in sorted(set(labels)):
            mask = labels == value
            color = "#595959" if value == -1 else colors[value % len(colors)]
            label = "Шу" if value == -1 else f"Топ {value}"
            ax.scatter(X.iloc[mask, 2], X.iloc[mask, 3], s=18,
                       marker="x" if value == -1 else "o", color=color, alpha=.8,
                       label=f"{label} ({mask.sum()})", zorder=3 if value == -1 else 2)
        ax.set_title(f"{row['model']}: қамту {row['covered']}/150")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.grid(alpha=.15)
    for ax in axes[1]:
        ax.set_xlabel("Күлте жапырақшасының ұзындығы, см")
    for ax in axes[:, 0]:
        ax.set_ylabel("Күлте жапырақшасының ені, см")
    fig.savefig(FIGURES / "ch08_06_clusters.png", dpi=180)
    plt.close(fig)
    return {"rows": rows, "fit_rows": len(X), "fit_dimensions": X.shape[1],
            "ARI_rows": len(X), "ARI_noise_group": -1,
            "scaler_fit_rows": int(state["scaler"].n_samples_seen_),
            "GMM_converged": bool(gmm.converged_), "GMM_iterations": int(gmm.n_iter_),
            "assignments": {name: labels.tolist() for name, labels in state["labels"].items()}}


def evaluate_pca(X, y, names, state):
    train, test, pca = state["train"], state["test"], state["pca"]
    metrics = {"feature_names": list(X.columns),
               "MSE_PCA2_by_feature": state["mse"].tolist(),
               "MSE_mean_baseline_by_feature": state["baseline_mse"].tolist(),
               "MSE_PCA2_mean": float(state["mse"].mean()),
               "MSE_mean_baseline_mean": float(state["baseline_mse"].mean()),
               "explained_variance_train": pca.explained_variance_.tolist(),
               "explained_variance_ratio_train": pca.explained_variance_ratio_.tolist(),
               "explained_variance_ratio_sum_train": float(pca.explained_variance_ratio_.sum()),
               "train_indices": train.tolist(), "test_indices": test.tolist(),
               "split_stratified": False, "split_seed": 2026,
               "scaler_fit_rows": int(state["scaler"].n_samples_seen_),
               "PCA_fit_rows": int(pca.n_samples_), "q": pca.n_components_,
               "test_used_to_choose_q": False,
               "test_species_counts_posthoc": np.bincount(y.iloc[test], minlength=3).tolist()}
    print("Train түсіндірілген дисперсиясы:", metrics["explained_variance_train"])
    print("Train түсіндірілген дисперсиясының үлесі:", metrics["explained_variance_ratio_train"])
    print("Test-тегі әр стандартталған белгінің MSE қатесі:\n", pd.DataFrame({
        "feature": X.columns, "PCA2": state["mse"], "TrainMean": state["baseline_mse"]
    }).to_string(index=False))
    print("Test орташа MSE қатесі: PCA2 / Train орташа мәні:", metrics["MSE_PCA2_mean"],
          metrics["MSE_mean_baseline_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), layout="constrained")
    projection = pca.transform(state["Z_train"])
    axes[0].scatter(projection[:, 0], projection[:, 1], s=12, color="#A3A3A3",
                    alpha=.55, label="Оқыту (120)")
    for species, color in enumerate(["#176B87", "#B34B20", "#3A7D43"]):
        mask = y.iloc[test].to_numpy() == species
        projected = state["coordinates"][mask]
        axes[0].scatter(projected[:, 0], projected[:, 1], s=28, color=color,
                        label=f"{names[species]} ({mask.sum()})")
    axes[0].set(xlabel="PC1", ylabel="PC2", title="Train-де үйретілген осьтерге проекция")
    axes[0].legend(fontsize=7.5, loc="best")
    positions = np.arange(X.shape[1])
    axes[1].bar(positions - .18, state["mse"], width=.36,
                label="PCA, 2 компонент", color="#176B87")
    axes[1].bar(positions + .18, state["baseline_mse"], width=.36,
                label="Train орташа мәндері", color="#B34B20")
    axes[1].set(xticks=positions, xticklabels=["Тостағ.\nұзындығы", "Тостағ.\nені",
                                             "Күлте\nұзындығы", "Күлте\nені"],
                ylabel="Стандарттық масштабтағы MSE", title="Test-тің 30 жолын реконструкциялау")
    axes[1].set_ylim(0, 1.5)
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].grid(axis="y", alpha=.15)
    fig.savefig(FIGURES / "ch08_07_pca.png", dpi=180)
    plt.close(fig)
    return metrics


def save_artifacts(iris, cluster_result, pca_result, result_path):
    split = np.full(len(iris.data), "test", dtype=object)
    split[pca_result["train_indices"]] = "train"
    iris.data.assign(row_id=np.arange(len(iris.data)), target=iris.target,
                     species=iris.target_names[iris.target.to_numpy()], pca_split=split).to_csv(
        DATASETS / "iris_unsupervised.csv", index=False)
    attribution = DATASETS / "Iris_Attribution.txt"
    if not attribution.exists():
        attribution.write_text(ATTRIBUTION, encoding="utf-8")
    result = {"data": "real corrected sklearn Iris, derived from UCI", "shape": [150, 4],
              "license": "CC BY 4.0", "DOI": "10.24432/C56C76",
              "cluster": cluster_result, "PCA": pca_result}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    iris = load_iris(as_frame=True)
    cluster_state = fit_clusters(iris.data)
    pca_state = fit_pca(iris.data)
    cluster_result = evaluate_clusters(iris.data, iris.target, cluster_state)
    pca_result = evaluate_pca(iris.data, iris.target, iris.target_names, pca_state)
    save_artifacts(iris, cluster_result, pca_result, Path(__file__).with_name("lab08_results.json"))
