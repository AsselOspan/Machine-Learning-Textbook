"""№2 зертханалық жұмыс. Жүктеме туралы барлық мәлімет оқу синтетикасы болып табылады."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

def make_preprocessor(scaler, numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", scaler)]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), categorical)])

def micro_experiment():
    train = pd.DataFrame({"temperature_C": [20., np.nan, 30., 40.],
                          "hours": [2., 4., np.nan, 8.],
                          "mode": pd.Series(["eco", "standard", np.nan, "eco"], dtype=object)})
    new = pd.DataFrame({"temperature_C": [50., np.nan], "hours": [np.nan, 6.],
                        "mode": pd.Series(["maintenance", "eco"], dtype=object)})
    prep = make_preprocessor(StandardScaler(), ["temperature_C", "hours"], ["mode"])
    transformed_train = prep.fit_transform(train)
    medians_before = prep.named_transformers_["num"].named_steps["impute"].statistics_.copy()
    transformed_new = prep.transform(new)
    names = prep.get_feature_names_out().tolist()
    print("Микромысал: белгілер", names)
    print("Transform-тен кейінгі жаңа жолдар:\n", np.round(transformed_new, 6))
    numeric = prep.named_transformers_["num"]
    result = {"numeric_medians": numeric.named_steps["impute"].statistics_.tolist(),
              "numeric_means": numeric.named_steps["scale"].mean_.tolist(),
              "numeric_scales": numeric.named_steps["scale"].scale_.tolist(),
              "feature_names": names, "train_transformed": transformed_train.tolist(),
              "new_transformed": transformed_new.tolist(),
              "statistics_unchanged_after_transform": bool(np.array_equal(
                  medians_before, numeric.named_steps["impute"].statistics_))}
    values = np.vstack([transformed_train, transformed_new])
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    image = ax.imshow(values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set(xticks=range(4), xticklabels=["температура z", "уақыт z", "eco", "standard"],
           yticks=range(6), yticklabels=["оқыту 1", "оқыту 2", "оқыту 3", "оқыту 4", "жаңа 1", "жаңа 2"])
    ax.axhline(3.5, color="black", linewidth=2)
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(values[i,j])>1.7 else "black")
    fig.colorbar(image, ax=ax, label="Түрленген белгінің мәні")
    fig.savefig(FIGURES / "ch02_05_micro.png", dpi=180)
    plt.close(fig)
    return result

def make_equipment_data():
    rng = np.random.default_rng(2026)
    n = 500
    temperature = rng.normal(25, 6, n)
    hours = rng.uniform(1, 12, n)
    mode = rng.choice(["eco", "standard", "boost"], n, p=[.25, .5, .25])
    shift = rng.choice(["day", "night"], n, p=[.65, .35])
    effect = pd.Series(mode).map({"eco": -10, "standard": 0, "boost": 15}).to_numpy()
    y = 40 + 1.8*temperature + 5*hours + effect + 8*(shift=="day") + rng.normal(0, 5, n)
    X = pd.DataFrame({"temperature_C": temperature, "hours": hours,
                      "mode": pd.Series(mode, dtype=object),
                      "shift": pd.Series(shift, dtype=object)})
    for column, probability in {"temperature_C": .10, "hours": .12, "mode": .08, "shift": .05}.items():
        X.loc[rng.random(n)<probability, column] = np.nan
    return X, y

def equipment_experiment():
    X, y = make_equipment_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=2026)
    numeric, categorical = ["temperature_C", "hours"], ["mode", "shift"]
    models = {"Dummy": DummyRegressor(strategy="mean")}
    for label, scaler in {"Standard": StandardScaler(), "MinMax": MinMaxScaler()}.items():
        models[label] = Pipeline([("prep", make_preprocessor(scaler, numeric, categorical)),
                                  ("model", Ridge(alpha=1.0))])
    rows, predictions, parameters = [], {}, {}
    for label, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        predictions[label] = pred
        rows.append({"approach": label, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)), "R2": r2_score(yte, pred)})
        if label != "Dummy":
            prep = model.named_steps["prep"]
            ztr, zte = prep.transform(Xtr), prep.transform(Xte)
            parameters[label] = {"feature_names": prep.get_feature_names_out().tolist(),
                                 "numeric_medians": prep.named_transformers_["num"].named_steps["impute"].statistics_.tolist(),
                                 "train_shape": list(ztr.shape), "test_shape": list(zte.shape),
                                 "finite": bool(np.isfinite(ztr).all() and np.isfinite(zte).all())}
    print("Бүкіл жиынтықтағы жетіспейтін мәндер:", X.isna().sum().to_dict())
    print(pd.DataFrame(rows).to_string(index=False))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.9), layout="constrained")
    for label, marker in [("Standard", "o"), ("MinMax", "x")]:
        pred = predictions[label]
        axs[0].scatter(yte, pred, s=16, alpha=.65, marker=marker, label=label)
        axs[1].scatter(pred, yte-pred, s=16, alpha=.65, marker=marker, label=label)
    limits = [float(yte.min()), float(yte.max())]
    axs[0].plot(limits, limits, "k--", linewidth=1)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0].set(xlabel="Нақты жүктеме, шартты бірл.", ylabel="Болжам, шартты бірл.")
    axs[1].set(xlabel="Болжам, шартты бірл.", ylabel="Қалдық, шартты бірл.")
    axs[0].legend(); axs[1].legend()
    fig.savefig(FIGURES / "ch02_06_equipment.png", dpi=180)
    plt.close(fig)
    datasets = ROOT / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    X.assign(load_synthetic=y).to_csv(datasets / "synthetic_equipment_load.csv", index=False)
    return {"synthetic": True, "seed": 2026, "rows": len(X), "train_size": len(ytr),
            "test_size": len(yte), "missing_counts": X.isna().sum().to_dict(),
            "train_missing_counts": Xtr.isna().sum().to_dict(), "metrics": rows,
            "fitted_parameters": parameters}

if __name__ == "__main__":
    result = {"micro": micro_experiment(), "equipment": equipment_experiment()}
    (Path(__file__).parent / "lab02_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
