"""ЗЖ 11. Күнделікті болжамды хронологиялық тексеру және тарихтың екі режимі."""
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
FIGURES = ROOT / "figures"
PACKAGES = ("numpy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl")
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
WARM_UP = 14
TEST_DAYS = 21
FEATURE_NAMES = ["lag_1", "lag_7", "lag_14", "mean_previous_7",
                 "weekday_sin", "weekday_cos", "time_index", "planned_promo"]
RIDGE = "Ridge"
HIST = "HistGradientBoosting"
BASELINE = "Seasonal naive"
SPECS = [(BASELINE, None)] + [(RIDGE, alpha) for alpha in (.1, 1, 10, 100)] + [(HIST, leaves) for leaves in (7, 15)]
plt.rcParams["font.family"] = "DejaVu Sans"


def make_series():
    dates = pd.date_range("2025-01-06", periods=224, freq="D")
    days = np.arange(len(dates))
    week = np.array([3, 4, 3, 2, 0, -5, -8])
    promo = ((days % 17 == 5) | (days % 29 == 8)).astype(int)
    innovations = np.random.default_rng(2026).normal(0, 2.3, len(dates))
    residual = innovations.copy()
    for day in range(1, len(dates)):
        residual[day] += .6 * residual[day - 1]
    demand = (85 + .06 * days + week[dates.dayofweek] + 3 * np.cos(2 * np.pi * days / 56)
              + 8 * promo + residual)
    return pd.DataFrame({"date": dates, "demand_units": np.round(demand, 3), "planned_promo": promo})


def known_features(frame):
    angle = 2 * np.pi * frame["date"].dt.dayofweek.to_numpy() / 7
    return np.column_stack([np.sin(angle), np.cos(angle), np.arange(len(frame)), frame["planned_promo"]])


def feature_row(history, known):
    return np.array([history[-1], history[-7], history[-14], np.mean(history[-7:]), *known])


def prepare_dev(frame):
    y = frame["demand_units"].to_numpy()
    known = known_features(frame)
    dev_end = len(frame) - TEST_DAYS
    dev_days = np.arange(WARM_UP, dev_end)
    test_days = np.arange(dev_end, len(frame))
    X_dev = np.array([feature_row(y[:day], known[day]) for day in dev_days])
    return {"y": y, "known": known, "dev_end": dev_end, "dev_days": dev_days,
            "test_days": test_days, "X_dev": X_dev}


def new_model(family, value):
    if family == BASELINE:
        return None
    if family == RIDGE:
        return make_pipeline(StandardScaler(), Ridge(alpha=value))
    return HistGradientBoostingRegressor(max_leaf_nodes=value, max_iter=120,
                                         learning_rate=.05, min_samples_leaf=8,
                                         l2_regularization=.2, early_stopping=False,
                                         random_state=2026)


def predict_day(model, history, known):
    if model is None:
        return float(history[-7])
    return float(model.predict(feature_row(history, known)[None, :])[0])


def observed_updates(model, y, known, days):
    history = y[:days[0]].tolist()
    forecasts = []
    for day in days:
        forecasts.append(predict_day(model, history, known[day]))
        history.append(float(y[day]))
    return np.array(forecasts)


def fixed_origin(model, initial_history, known, days):
    history = list(initial_history)
    forecasts = []
    for day in days:
        prediction = predict_day(model, history, known[day])
        forecasts.append(prediction)
        history.append(prediction)
    return np.array(forecasts)


def expanding_validation(frame, data):
    X, y, known, days = (data[key] for key in ("X_dev", "y", "known", "dev_days"))
    dev_end = data["dev_end"]
    folds = list(TimeSeriesSplit(n_splits=4, test_size=14, gap=0).split(X))
    fold_records = []
    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    for train, validation in folds:
        t, v = days[train], days[validation]
        fold_records.append({"train_days": t.tolist(), "validation_days": v.tolist(),
                             "train_last_date": dates[t[-1]], "validation_first_date": dates[v[0]],
                             "validation_last_date": dates[v[-1]]})
    rows = []
    with threadpool_limits(limits=1):
        for family, value in SPECS:
            errors, predictions = [], []
            for train, validation in folds:
                model = new_model(family, value)
                if model is not None:
                    model.fit(X[train], y[days[train]])
                pred = observed_updates(model, y[:dev_end], known[:dev_end], days[validation])
                errors.append(float(mean_absolute_error(y[days[validation]], pred)))
                predictions.append(pred.tolist())
            rows.append({"family": family, "value": value, "fold_MAE": errors,
                         "mean_MAE": float(np.mean(errors)), "fold_predictions": predictions})
    chosen = {family: min((row for row in rows if row["family"] == family),
                          key=lambda row: row["mean_MAE"])["value"] for family in (RIDGE, HIST)}
    for row in rows:
        print(row["family"], row["value"], "фолд MAE", np.round(row["fold_MAE"], 6),
              f"орташа {row['mean_MAE']:.6f}")
    print("Таңдалған конфигурациялар", chosen)
    return {"rows": rows, "folds": fold_records, "selected": chosen, "selection_metric": "фолдтардың орташа MAE мәні",
            "tie_rule": "бекітілген реттегі бірінші кандидат", "learned_fit_count": (len(SPECS) - 1) * len(folds)}


def final_one_step(frame, data, validation):
    X, y, known = (data[key] for key in ("X_dev", "y", "known"))
    dev_days, test_days = data["dev_days"], data["test_days"]
    models = {BASELINE: None}
    rows, predictions = [], {}
    with threadpool_limits(limits=1):
        for family, value in validation["selected"].items():
            models[family] = new_model(family, value).fit(X, y[dev_days])
        for family, model in models.items():
            forecast = observed_updates(model, y, known, test_days)
            predictions[family] = forecast.tolist()
            rows.append({"family": family, "MAE": float(mean_absolute_error(y[test_days], forecast)),
                         "RMSE": float(root_mean_squared_error(y[test_days], forecast))})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return models, {"rows": rows, "predictions": predictions, "observed": y[test_days].tolist(),
                    "dates": frame["date"].iloc[test_days].dt.strftime("%Y-%m-%d").tolist(),
                    "dev_fit_rows": len(dev_days), "test_indices": test_days.tolist(), "test_refits": 0,
                    "update": "s мақсатты күнін болжау, содан кейін бақыланған y[s] мәнін қосу"}


def fixed_origin_comparison(frame, data, model, test):
    days = data["test_days"][:7]
    y, known = data["y"], data["known"]
    with threadpool_limits(limits=1):
        recursive = fixed_origin(model, y[:days[0]], known, days)
        baseline_recursive = fixed_origin(None, y[:days[0]], known, days)
    rolling = np.array(test["predictions"][RIDGE][:len(days)])
    np.testing.assert_allclose(recursive[0], rolling[0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(baseline_recursive, test["predictions"][BASELINE][:len(days)])
    observed = y[days]
    result = {"forecast_origin_date": frame["date"].iloc[days[0] - 1].strftime("%Y-%m-%d"),
              "dates": test["dates"][:len(days)], "horizons": list(range(1, len(days) + 1)),
              "observed": observed.tolist(), "observed_update_forecasts": rolling.tolist(),
              "fixed_origin_forecasts": recursive.tolist(), "seasonal_naive_same_for_h1_to_h7": True,
              "observed_update_MAE": float(mean_absolute_error(observed, rolling)),
              "fixed_origin_MAE": float(mean_absolute_error(observed, recursive)),
              "recursive_update": "болжамды қосу, болашақ нақты мақсат кіріске берілмейді", "refits": 0}
    print(f"Алғашқы жеті күн, Ridge MAE: нақты мәндермен жаңарту {result['observed_update_MAE']:.6f}; "
          f"бекітілген басталу {result['fixed_origin_MAE']:.6f}")
    return result


def availability_probe(frame, data, model):
    days = data["test_days"][:6]
    cutoff = days[4]
    y, known = data["y"], data["known"]
    changed = y.copy()
    changed[cutoff:] += 500
    with threadpool_limits(limits=1):
        original = observed_updates(model, y, known, days)
        perturbed = observed_updates(model, changed, known, days)
    np.testing.assert_array_equal(original[:5], perturbed[:5])
    assert not np.isclose(original[5], perturbed[5])
    date = frame["date"].iloc[cutoff].strftime("%Y-%m-%d")
    result = {"family": RIDGE, "first_changed_actual_date": date, "added_demand_units": 500,
              "unchanged_forecast_count": 5, "forecasts_through_cutoff_equal": True,
              "following_day_forecast_original": float(original[5]),
              "following_day_forecast_perturbed": float(perturbed[5]),
              "model_refitted": False}
    print("Қолжетімсіз мақсатты тексеру:", date, "алғашқы бес болжам дәл бірдей")
    return result


def plot_splits(frame, data, validation, path):
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.4), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.1]})
    dates = mdates.date2num(frame["date"])
    first_test = dates[data["dev_end"]]
    axes[0].plot(dates, frame["demand_units"], color="#176B87", linewidth=1.25)
    axes[0].axvspan(first_test, dates[-1] + 1, color="#B34B20", alpha=.17)
    axes[0].set(title="Синтетикалық күндік сұраныс және соңғы 21 test күні",
                ylabel="Шартты бірліктер")
    axis = axes[1]
    axis.axvspan(dates[0], dates[WARM_UP], color="#777777", alpha=.17, label="Бастапқы жинақталу")
    for position, fold in enumerate(validation["folds"]):
        train, val = fold["train_days"], fold["validation_days"]
        axis.barh(position, len(train), left=dates[train[0]], height=.59, color="#176B87",
                  label="Оқыту / қайта оқыту" if position == 0 else None)
        axis.barh(position, len(val), left=dates[val[0]], height=.59, color="#3A7D43",
                  label="Валидация" if position == 0 else None)
    axis.barh(4, len(data["dev_days"]), left=dates[WARM_UP], height=.59, color="#176B87")
    axis.barh(4, len(data["test_days"]), left=first_test, height=.59, color="#B34B20", label="Тест")
    axis.set(yticks=np.arange(5), yticklabels=["1-фолд", "2-фолд", "3-фолд", "4-фолд", "Қорытынды"],
             xlim=(dates[0], dates[-1] + 1), xlabel="Күн, 2025 жыл")
    axis.invert_yaxis()
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axis.legend(ncols=4, loc="upper left", bbox_to_anchor=(0, -.22), fontsize=8.7, frameon=False)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_forecasts(test, comparison, path):
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.7), layout="constrained")
    days = pd.to_datetime(test["dates"])
    axes[0].plot(days, test["observed"], color="#252525", label="Нақты мән", marker="o", markersize=3)
    for name, label, color, linestyle in ((BASELINE, "Маусымдық аңғал", "#777777", "--"),
                                          (RIDGE, RIDGE, "#176B87", "-"),
                                          (HIST, HIST, "#3A7D43", "-.")):
        axes[0].plot(days, test["predictions"][name], color=color, linestyle=linestyle, label=label)
    axes[0].set(title="Бір күн алға жасалған 21 бірізді болжам", ylabel="Сұраныс")
    short = pd.to_datetime(comparison["dates"])
    axes[1].plot(short, comparison["observed"], color="#252525", label="Нақты мән", marker="o", markersize=3)
    axes[1].plot(short, comparison["observed_update_forecasts"], color="#176B87", label="Ridge, күн сайын жаңа нақты мән")
    axes[1].plot(short, comparison["fixed_origin_forecasts"], color="#B34B20", linestyle="--",
                 label="Ridge, бір origin")
    axes[1].set(title="Сол алғашқы 7 күн, тарихтың екі режимі", ylabel="Сұраныс", xlabel="Күн, 2025 жыл")
    for axis, interval in zip(axes, (3, 1)):
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        axis.legend(ncols=2, fontsize=8.1, loc="best")
        axis.grid(alpha=.16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_results(frame, data, models, validation, test, comparison, probe, output):
    DATASETS.mkdir(parents=True, exist_ok=True)
    csv_path = DATASETS / "synthetic_daily_demand.csv"
    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d", float_format="%.3f")
    reconstructed = pd.read_csv(csv_path, parse_dates=["date"], float_precision="round_trip")
    pd.testing.assert_frame_equal(reconstructed, frame, check_exact=True)
    parameters = {family: model.get_params(deep=False) if family == HIST else {
        "StandardScaler": model.named_steps["standardscaler"].get_params(),
        "Ridge": model.named_steps["ridge"].get_params()}
        for family, model in models.items() if model is not None}
    result = {"synthetic": True, "generator_seed": 2026, "n_observations": len(frame), "frequency": "D",
              "data_start": frame["date"].iloc[0].strftime("%Y-%m-%d"),
              "data_end": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
              "warm_up_days": WARM_UP, "dev_days_including_warm_up": data["dev_end"],
              "supervised_dev_indices": data["dev_days"].tolist(), "feature_names": FEATURE_NAMES,
              "models": parameters, "validation": validation, "test": test, "fixed_origin": comparison,
              "availability_probe": probe,
              "runtime": {"Python": platform.python_version(), "device": "CPU",
                          "packages": {name: version(name) for name in PACKAGES},
                          "threadpoolctl_limit": 1,
                          "thread_environment": {name: os.getenv(name) for name in THREAD_ENVIRONMENT}}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = make_series()
    data = prepare_dev(frame)
    validation = expanding_validation(frame, data)
    models, test = final_one_step(frame, data, validation)
    comparison = fixed_origin_comparison(frame, data, models[RIDGE], test)
    probe = availability_probe(frame, data, models[RIDGE])
    plot_splits(frame, data, validation, FIGURES / "ch11_05_temporal_splits.png")
    plot_forecasts(test, comparison, FIGURES / "ch11_06_forecast_modes.png")
    save_results(frame, data, models, validation, test, comparison, probe,
                 Path(__file__).with_name("lab11_results.json"))
