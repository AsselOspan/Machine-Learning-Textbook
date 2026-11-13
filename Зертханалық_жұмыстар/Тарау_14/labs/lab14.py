"""ЗЖ 14. Модель, жергілікті журнал, HTTP және синтетикалық істен шығулар мониторингі."""
import argparse
from contextlib import closing
from hashlib import sha256
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import threading
from urllib.parse import urlsplit
from urllib.request import url2pathname

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DEFAULT_ARTIFACTS = Path(__file__).with_name("lab14_artifacts")
MODEL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
THRESHOLD = .5
PSI_LIMIT = .2
CUTPOINTS = [60, 66, 72, 78, 84]
FEATURE_SCHEMA = [{"name": "temperature_c", "type": "number", "min": 0, "max": 150},
                  {"name": "vibration_mm_s", "type": "number", "min": 0, "max": 20},
                  {"name": "load_ratio", "type": "number", "min": 0, "max": 1}]
FEATURES = [item["name"] for item in FEATURE_SCHEMA]
MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"
MAX_HTTP_BYTES = 4096
PACKAGES = ("numpy", "pandas", "scikit-learn", "joblib")
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
plt.rcParams["font.family"] = "DejaVu Sans"


def synthetic_equipment(n, seed, shift_c=0):
    rng = np.random.default_rng(seed)
    temperature = rng.normal(70 + shift_c, 6, n)
    vibration = np.clip(rng.normal(2.5, .6, n), 0, 20)
    load = rng.uniform(.35, .95, n)
    logits = -1.35 + .11 * (temperature - 70) + .9 * (vibration - 2.5) + 1.8 * (load - .65)
    p = 1 / (1 + np.exp(-logits))
    return pd.DataFrame({"event_id": seed * 1000 + np.arange(n), "temperature_c": temperature,
                         "vibration_mm_s": vibration, "load_ratio": load, "failure": rng.binomial(1, p)})


def make_experiment_data():
    return {"train": synthetic_equipment(800, 2026), "reference": synthetic_equipment(300, 2027),
            "test": synthetic_equipment(300, 2028)}


def runtime_versions():
    return {"Python": platform.python_version(), **{name: version(name) for name in PACKAGES}}


def positive_probability(model, features):
    column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, column]


def quality(y, probability):
    decisions = (probability >= THRESHOLD).astype(int)
    return {"accuracy": float(accuracy_score(y, decisions)), "roc_auc": float(roc_auc_score(y, probability)),
            "f1_positive": float(f1_score(y, decisions, pos_label=1, zero_division=0))}


def fit_experiment(data):
    train = data["train"]
    models = {"DummyPrior": DummyClassifier(strategy="prior"),
              "Logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=2026))}
    with threadpool_limits(limits=1):
        for model in models.values():
            model.fit(train[FEATURES], train["failure"])
        reference = data["reference"]
        offline_test = data["test"]
        rows, test_probabilities = [], {}
        for name, model in models.items():
            ref_p = positive_probability(model, reference[FEATURES])
            test_p = positive_probability(model, offline_test[FEATURES])
            rows.append({"model": name, "reference": quality(reference["failure"], ref_p),
                         "test": quality(offline_test["failure"], test_p)})
            test_probabilities[name] = test_p.tolist()
    summary = {"deployment_model": "Logistic", "deployment_choice": "predeclared before metrics",
               "rows": rows, "test_event_ids": offline_test["event_id"].tolist(),
               "test_labels": offline_test["failure"].tolist(), "test_probabilities": test_probabilities}
    print("Офлайн салыстыру", rows)
    return models, summary


def save_bundle(model, bundle):
    bundle.mkdir(parents=True, exist_ok=True)
    pending_model = bundle / (MODEL_FILE + ".pending")
    pending_manifest = bundle / (MANIFEST_FILE + ".pending")
    joblib.dump(model, pending_model, compress=3)
    serialized = pending_model.read_bytes()
    manifest = {"model_version": MODEL_VERSION, "input_schema_version": SCHEMA_VERSION,
                "artifact_file": MODEL_FILE, "artifact_bytes": len(serialized),
                "artifact_sha256": sha256(serialized).hexdigest(), "feature_schema": FEATURE_SCHEMA,
                "decision_threshold": THRESHOLD, "positive_class": 1, "runtime": runtime_versions()}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    pending_manifest.write_bytes(encoded)
    pending_model.replace(bundle / MODEL_FILE)
    pending_manifest.replace(bundle / MANIFEST_FILE)
    return manifest, sha256(encoded).hexdigest()


def load_verified(bundle, trusted_manifest_sha256):
    # Күтілетін манифест хеші сенімді арнадан алынуы керек.
    raw_manifest = (bundle / MANIFEST_FILE).read_bytes()
    if sha256(raw_manifest).hexdigest() != trusted_manifest_sha256:
        raise ValueError("Манифестің SHA-256 мәні сәйкес емес")
    manifest = json.loads(raw_manifest)
    expected = {"model_version": MODEL_VERSION, "input_schema_version": SCHEMA_VERSION,
                "artifact_file": MODEL_FILE, "feature_schema": FEATURE_SCHEMA,
                "decision_threshold": THRESHOLD, "positive_class": 1}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Қолдау көрсетілмейтін модель келісімшарты")
    if manifest.get("runtime") != runtime_versions():
        raise ValueError("Модельдің орындау ортасының нұсқалары бұл ортаға сәйкес емес")
    serialized = (bundle / MODEL_FILE).read_bytes()
    if len(serialized) != manifest["artifact_bytes"] or sha256(serialized).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("Модельдің SHA-256 мәні немесе ұзындығы сәйкес емес")
    # Сенімді өндірушіден алынған, тексерілген сол байттарды десериализациялау.
    return joblib.load(io.BytesIO(serialized)), manifest


def verify_reload(model, test_frame, bundle, run_dir):
    manifest, pin = save_bundle(model, bundle)
    loaded, read_manifest = load_verified(bundle, pin)
    assert manifest == read_manifest
    with threadpool_limits(limits=1):
        before = model.predict_proba(test_frame[FEATURES])
        after = loaded.predict_proba(test_frame[FEATURES])
    np.testing.assert_array_equal(before, after)
    tampered = run_dir / "tampered"
    tampered.mkdir(parents=True, exist_ok=True)
    (tampered / MANIFEST_FILE).write_bytes((bundle / MANIFEST_FILE).read_bytes())
    damaged = bytearray((bundle / MODEL_FILE).read_bytes())
    damaged[len(damaged) // 2] ^= 1
    (tampered / MODEL_FILE).write_bytes(damaged)
    try:
        load_verified(tampered, pin)
    except ValueError as exc:
        if "Модельдің SHA-256" not in str(exc):
            raise
        rejection = str(exc)
    else:
        raise AssertionError("Өзгертілген байттар қабылданды")
    print("Қайта жүктеудегі ең үлкен |айырма|", float(np.max(abs(before - after))), "өзгертілген:", rejection)
    evidence = {"manifest": manifest, "manifest_sha256": pin, "predict_proba_equal": True,
                "max_abs_predict_proba_difference": float(np.max(abs(before - after))),
                "checked_probability_shape": list(after.shape), "corruption_rejected": True,
                "corruption_error": rejection}
    return loaded, evidence, before[:, 1], after[:, 1]


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Қайталанатын JSON өрісі")
        result[key] = value
    return result


def parse_features(raw):
    values = json.loads(raw, object_pairs_hook=unique_json_object)
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        raise ValueError("Сұлбаның дәл үш өрісі күтіледі")
    for field in FEATURE_SCHEMA:
        value = values[field["name"]]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not field["min"] <= value <= field["max"]):
            raise ValueError(f"Мына өріс үшін жарамсыз сан: {field['name']}")
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def make_loopback_server(model, manifest, port=0):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def reply(self, code, payload, allow=None):
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            if allow is not None:
                self.send_header("Allow", allow)
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True, "model_version": manifest["model_version"],
                                 "input_schema_version": manifest["input_schema_version"]})
            elif self.path == "/predict":
                self.reply(405, {"error": "POST қолданыңыз"}, allow="POST")
            else:
                self.reply(404, {"error": "Белгісіз бағыт"})

        def do_POST(self):
            if self.path != "/predict":
                if self.path == "/health":
                    self.reply(405, {"error": "GET қолданыңыз"}, allow="GET")
                else:
                    self.reply(404, {"error": "Белгісіз бағыт"})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "application/json күтіледі"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Сұрау денесі бос")
                if length > MAX_HTTP_BYTES:
                    self.reply(413, {"error": "Сұрау денесі тым үлкен"})
                    return
                features = parse_features(self.rfile.read(length))
            except (ValueError, OverflowError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            probability = float(positive_probability(model, features)[0])
            self.reply(200, {"failure_probability": probability, "predicted_failure": int(probability >= THRESHOLD),
                             "model_version": manifest["model_version"], "decision_threshold": THRESHOLD,
                             "input_schema_version": manifest["input_schema_version"]})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run_http_checks(model, manifest, sample):
    valid = {name: float(sample.iloc[0][name]) for name in reversed(FEATURES)}
    valid_bytes = json.dumps(valid).encode("utf-8")
    cases = [("health", "GET", "/health", None, None, 200),
             ("prediction", "POST", "/predict", valid_bytes, "application/json", 200),
             ("broken_json", "POST", "/predict", b"{", "application/json", 400),
             ("missing_field", "POST", "/predict", b"{}", "application/json", 400),
             ("boolean_feature", "POST", "/predict",
              json.dumps({**valid, "load_ratio": True}).encode(), "application/json", 400),
             ("out_of_range", "POST", "/predict",
              json.dumps({**valid, "load_ratio": 1.2}).encode(), "application/json", 400),
             ("duplicate_field", "POST", "/predict",
              valid_bytes[:-1] + b', "load_ratio": 0.5}', "application/json", 400),
             ("unknown_route", "GET", "/unknown", None, None, 404),
             ("wrong_method", "GET", "/predict", None, None, 405),
             ("oversized", "POST", "/predict", b"", "application/json", 413),
             ("wrong_content_type", "POST", "/predict", b"", "text/plain", 415)]
    statuses, bodies = {}, {}
    with make_loopback_server(model, manifest) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        thread.start()
        try:
            for name, method, path, content, mime, expected in cases:
                headers = {"Content-Type": mime} if mime else {}
                if name == "oversized":
                    # Оқылмайтын денені жібермей, ерте қабылдамауды тексеру.
                    headers["Content-Length"] = str(MAX_HTTP_BYTES + 1)
                with closing(HTTPConnection("127.0.0.1", server.server_port, timeout=4)) as connection:
                    connection.request(method, path, body=content, headers=headers)
                    with connection.getresponse() as response:
                        statuses[name] = response.status
                        assert response.status == expected, (name, response.status, expected)
                        assert response.headers.get_content_type() == "application/json"
                        bodies[name] = json.load(response)
                        if name == "wrong_method":
                            assert response.headers["Allow"] == "POST"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert bodies["health"]["ready"] and bodies["health"]["model_version"] == MODEL_VERSION
    expected = float(positive_probability(model, sample[FEATURES])[0])
    measured = bodies["prediction"]["failure_probability"]
    np.testing.assert_allclose(measured, expected, atol=1e-12, rtol=0)
    assert bodies["prediction"]["predicted_failure"] == int(expected >= THRESHOLD)
    print("Нақты loopback HTTP", statuses)
    return {"transport": "HTTP over TCP loopback", "bind_address": "127.0.0.1", "ephemeral_port": True,
            "statuses": statuses, "health": bodies["health"], "example_prediction": bodies["prediction"],
            "direct_probability": expected, "max_abs_probability_difference": abs(measured - expected),
            "server_closed": True}


def temperature_counts(values):
    return np.histogram(values, bins=[-np.inf, *CUTPOINTS, np.inf])[0]


def smoothed_psi(reference_counts, window_counts):
    reference = reference_counts.astype(float) + .5
    current = window_counts.astype(float) + .5
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def monitor_snapshot(name, as_of, events, probabilities, reference_counts, labels=None):
    counts = temperature_counts(events["temperature_c"])
    psi = smoothed_psi(reference_counts, counts)
    measured_quality = None
    matched_count = 0
    if labels is not None:
        visible = labels.loc[labels["available_on"] <= as_of, ["event_id", "failure"]]
        joined = events[["event_id"]].merge(visible, on="event_id", how="left", sort=False, validate="one_to_one")
        visible_mask = joined["failure"].notna().to_numpy()
        matched_count = int(visible_mask.sum())
        if matched_count == len(events):
            measured_quality = quality(joined["failure"].astype(int), probabilities)
    return {"window": name, "as_of": as_of, "n": len(events), "temperature_counts": counts.tolist(),
            "temperature_mean": float(events["temperature_c"].mean()), "temperature_psi": psi,
            "psi_alert": psi >= PSI_LIMIT, "mean_probability": float(np.mean(probabilities)),
            "predicted_positive_rate": float(np.mean(probabilities >= THRESHOLD)),
            "labels_ready": matched_count, "label_coverage": matched_count / len(events),
            "quality": measured_quality}


def simulate_monitoring(model, reference):
    reference_counts = temperature_counts(reference["temperature_c"])
    specs = [("stable", 2029, 0, "2026-02-01", "2026-02-01"),
             ("shifted", 2030, 9, "2026-02-08", "2026-02-08"),
             ("delayed", 2031, 0, "2026-02-15", "2026-02-22")]
    rows = []
    with threadpool_limits(limits=1):
        for name, seed, shift, time, ready in specs:
            complete = synthetic_equipment(250, seed, shift)
            events = complete.drop(columns="failure")
            probabilities = positive_probability(model, events[FEATURES])
            labels = complete[["event_id", "failure"]].assign(available_on=ready)
            labels = labels.sample(frac=1, random_state=2032)
            if name == "delayed":
                rows.append(monitor_snapshot("delayed_t0", time, events, probabilities, reference_counts))
                rows.append(monitor_snapshot("delayed_t7", ready, events, probabilities, reference_counts, labels))
            else:
                rows.append(monitor_snapshot(name, time, events, probabilities, reference_counts, labels))
    assert rows[2]["labels_ready"] == 0 and rows[2]["quality"] is None
    assert rows[3]["labels_ready"] == 250 and rows[3]["quality"] is not None
    assert rows[2]["temperature_psi"] == rows[3]["temperature_psi"]
    print("Мониторинг терезелері", rows)
    return {"temperature_cutpoints_c": CUTPOINTS, "histogram_smoothing_per_bin": .5,
            "educational_psi_alert_threshold": PSI_LIMIT, "reference_n": len(reference),
            "reference_temperature_counts": reference_counts.tolist(),
            "reference_temperature_mean": float(reference["temperature_c"].mean()),
            "label_join_key": "event_id", "minimum_label_coverage_for_reported_quality": 1,
            "window_specs": [{"name": name, "seed": seed, "temperature_shift_c": shift,
                              "event_date": time, "labels_available_on": ready}
                             for name, seed, shift, time, ready in specs], "rows": rows}


def track_experiments(offline, bundle, run_dir):
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    from mlflow.tracking import MlflowClient

    tracking_root = (run_dir / "mlruns").resolve()
    artifact_root = (run_dir / "mlflow_artifacts").resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_root.as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "lab14-synthetic-equipment"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri()))
    records = []
    for row in offline["rows"]:
        name = row["model"]
        run = client.create_run(experiment_id, run_name=name,
                                tags={"data_kind": "synthetic", "metric_split": "reference"})
        run_id = run.info.run_id
        parameters = {"model": name, "train_seed": "2026", "train_size": "800", "threshold": str(THRESHOLD)}
        parameters.update({"C": "1", "max_iter": "1000"} if name == "Logistic" else {"strategy": "prior"})
        metrics = {f"reference_{key}": value for key, value in row["reference"].items()}
        try:
            for key, value in parameters.items():
                client.log_param(run_id, key, value)
            for key, value in metrics.items():
                client.log_metric(run_id, key, value)
            if name == "Logistic":
                for file in (MODEL_FILE, MANIFEST_FILE):
                    client.log_artifact(run_id, str(bundle / file), artifact_path="bundle")
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id, status="FINISHED")
        stored = client.get_run(run_id)
        stored_files = sorted(item.path for item in client.list_artifacts(run_id, "bundle"))
        expected_files = [f"bundle/{MANIFEST_FILE}", f"bundle/{MODEL_FILE}"] if name == "Logistic" else []
        assert stored.info.status == "FINISHED" and stored.data.params == parameters
        assert stored.data.metrics == metrics and stored_files == expected_files
        artifact_digests = {}
        if name == "Logistic":
            uri = urlsplit(stored.info.artifact_uri)
            assert uri.scheme == "file" and uri.netloc in ("", "localhost")
            actual_root = Path(url2pathname(uri.path)).resolve()
            assert actual_root.is_relative_to(artifact_root)
            for file in (MODEL_FILE, MANIFEST_FILE):
                digest = sha256((actual_root / "bundle" / file).read_bytes()).hexdigest()
                assert digest == sha256((bundle / file).read_bytes()).hexdigest()
                artifact_digests[file] = digest
        records.append({"name": name, "run_id": run_id, "params": stored.data.params, "metrics": stored.data.metrics,
                        "status": stored.info.status, "artifacts": stored_files, "artifact_sha256": artifact_digests,
                        "artifact_uri": stored.info.artifact_uri})
    result = {"backend": "local MLflow FileStore", "tracking_uri": tracking_uri,
              "artifact_root_uri": artifact_root.as_uri(), "experiment_id": experiment_id,
              "version": version("mlflow-skinny"), "verified_runs": records}
    print("MLflow деректері сақталды", [(row["name"], row["run_id"]) for row in records])
    return result


def plot_bundle(before, after, offline, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained")
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[0].scatter(before, after, color="#176B87", alpha=.42, s=14)
    axes[0].set(title="300 жолда жүктеуді тексеру", xlabel="Сақтауға дейінгі ықтималдық",
                ylabel="Жүктеуден кейінгі ықтималдық", xlim=(0, 1), ylim=(0, 1))
    x = np.arange(len(offline["rows"]))
    for offset, metric, color, label in ((-.17, "roc_auc", "#176B87", "ROC-AUC"),
                                         (.17, "f1_positive", "#B34B20", "1 класының F1 өлшемі")):
        bars = axes[1].bar(x + offset, [row["test"][metric] for row in offline["rows"]], width=.32,
                           color=color, label=label)
        axes[1].bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axes[1].set(title="Жабық test", xticks=x, xticklabels=[row["model"] for row in offline["rows"]], ylim=(0, 1.1))
    axes[1].legend(loc="upper left", ncols=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_monitoring(monitoring, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained")
    rows = monitoring["rows"]
    positions = np.arange(len(rows))
    names = ["Тұрақты", "Ығысу", "Кідіру\nt₀ кезінде", "Кідіру\nt₀ + 7 кезінде"]
    psis = [row["temperature_psi"] for row in rows]
    bars = axes[0].bar(positions, psis, color=["#B34B20" if row["psi_alert"] else "#176B87" for row in rows])
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].axhline(PSI_LIMIT, linestyle="--", color="#666666", label="Оқу шегі 0,2")
    axes[0].set(title="PSI бойынша температура ығысуы", ylabel="PSI", ylim=(0, max(max(psis) * 1.18, .3)))
    axes[0].legend(fontsize=8, loc="upper left")
    visible = [(position, row["quality"]) for position, row in enumerate(rows) if row["quality"] is not None]
    for offset, metric, color, label in ((-.17, "roc_auc", "#176B87", "ROC-AUC"),
                                         (.17, "f1_positive", "#B34B20", "1 класының F1 өлшемі")):
        bars = axes[1].bar([position + offset for position, _ in visible],
                           [value[metric] for _, value in visible], width=.32, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)
    axes[1].text(2, .15, "Класс белгілері\nкүтілуде", ha="center", fontsize=8, color="#555555")
    axes[1].set(title="Қолжетімді класс белгілеріндегі сапа", ylim=(0, 1.13))
    axes[1].legend(fontsize=8, ncols=2, loc="upper left")
    for ax in axes:
        ax.set(xticks=positions, xticklabels=names)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_results(data, offline, bundle_evidence, http, monitoring, tracking, path):
    result = {"data_kind": "synthetic equipment failure", "decision_threshold": THRESHOLD,
              "data_partitions": {name: {"n": len(frame), "positive": int(frame["failure"].sum()),
                                          "event_ids": frame["event_id"].tolist()}
                                  for name, frame in data.items()},
              "offline": offline, "bundle": bundle_evidence, "http": http, "monitoring": monitoring,
              "tracking": {"backend": tracking["backend"], "version": tracking["version"],
                           "verified_runs": len(tracking["verified_runs"])},
              "runtime": {**runtime_versions(), "matplotlib": version("matplotlib"),
                          "threadpoolctl": version("threadpoolctl"),
                          "thread_environment": {name: os.getenv(name) for name in THREAD_ENVIRONMENT}}}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--serve", type=Path, metavar="BUNDLE", help="тексерілген жинақты loopback арқылы ұсыну")
    parser.add_argument("--manifest-sha256", help="manifest.json файлының сенімді SHA-256 мәні")
    args = parser.parse_args()
    if args.serve is not None:
        if args.manifest_sha256 is None:
            parser.error("--serve үшін сенімді арнадан алынған --manifest-sha256 қажет")
        model, manifest = load_verified(args.serve, args.manifest_sha256)
        with make_loopback_server(model, manifest) as server:
            print(f"Қызмет көрсету: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = make_experiment_data()
    models, offline = fit_experiment(data)
    bundle = run_dir / "bundle"
    model, bundle_evidence, before, after = verify_reload(models["Logistic"], data["test"], bundle, run_dir)
    http = run_http_checks(model, bundle_evidence["manifest"], data["test"].iloc[:1])
    monitoring = simulate_monitoring(model, data["reference"])
    tracking = track_experiments(offline, bundle, run_dir)
    plot_bundle(before, after, offline, FIGURES / "ch14_05_bundle_parity.png")
    plot_monitoring(monitoring, FIGURES / "ch14_06_monitoring_windows.png")
    save_results(data, offline, bundle_evidence, http, monitoring, tracking,
                 Path(__file__).with_name("lab14_results.json"))
    Path(__file__).with_name("lab14_tracking.json").write_text(
        json.dumps(tracking, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
