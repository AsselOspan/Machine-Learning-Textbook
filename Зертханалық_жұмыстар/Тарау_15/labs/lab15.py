"""ЗЖ 15. Жергілікті FLAN-T5-small және синтетикалық құжаттардағы жауаптарды тексеру."""
import argparse
from collections import Counter
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import unicodedata


DATASET_NAME = "synthetic_grounded_qa.json"
DATASET_SHA256 = "5f5d47bc9f976e59b71c75214570b6191fe8900144cc29dc0717d8af728b54fd"
MODEL_NAME = "flan-t5-small"
MANIFEST_NAME = "flan_t5_small_manifest.json"
MODEL_ID = "google/flan-t5-small"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
WEIGHTS_SHA256 = "495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749"
MODEL_FILES = {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
               "spiece.model", "tokenizer.json", "tokenizer_config.json"}
METHODS = {"without_context": "Контекстсіз FLAN", "with_context": "Контексті бар FLAN",
           "extractor": "Өріс бойынша шығарып алу"}
OUTCOMES = {"supported_answer": ("Дұрыс жауап", "#20705e"),
            "correct_abstention": ("Негізді бас тарту", "#315b93"),
            "false_abstention": ("Жалған бас тарту", "#bd7407"),
            "other_error": ("Басқа қате", "#8c4f70")}


def asset_root(start):
    override = os.environ.get("LAB15_ASSET_ROOT")
    start = Path(override).resolve() if override else Path(start).resolve()
    candidates = (start,) if override else (start, *start.parents)
    for root in candidates:
        for candidate in (root, root / "work"):
            if (candidate / "datasets" / DATASET_NAME).is_file():
                return candidate
    raise FileNotFoundError(f"LAB15_ASSET_ROOT мәнін datasets/{DATASET_NAME} орналасқан каталогқа орнатыңыз")


def file_sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def normalize(text):
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split()).rstrip(".!?").rstrip()


def read_dataset(path, *, verify_reference=True):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if verify_reference and digest != DATASET_SHA256:
        raise ValueError("Деректер жиынтығының SHA-256 мәні сәйкес емес")
    dataset = json.loads(raw)
    cases = dataset["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Жағдай идентификаторлары қайталанады")
    abstention = normalize(dataset["abstention_token"])
    for case in cases:
        documents = {doc["id"]: doc["text"] for doc in case["documents"]}
        if not documents or len(documents) != len(case["documents"]):
            raise ValueError(f"{case['id']} құжаттары жарамсыз")
        if (case["type"] == "answerable") == (normalize(case["expected"]) == abstention):
            raise ValueError(f"{case['id']} күтілетін жауабы үйлеспейді")
        for span in case["evidence"]:
            if span["quote"] not in documents[span["document_id"]]:
                raise ValueError(f"{case['id']} растайтын үзіндісі жарамсыз")
        if "attack_target" in case:
            target = case["attack_target"]
            if target == case["expected"] or not any(target in text for text in documents.values()):
                raise ValueError(f"{case['id']} нұсқау мақсаты жарамсыз")
    return dataset, digest


def build_prompt(case, protocol, with_context):
    context = protocol["empty_context"]
    if with_context:
        context = protocol["document_separator"].join(doc["text"] for doc in case["documents"])
    return protocol["prompt_format"].format(instruction=protocol["prompt_instruction"],
                                            context=context, question=case["question"])


def extract_field(case, abstention):
    match = re.fullmatch(r"What is the (.+) for [A-Za-z0-9-]+\?", case["question"], re.I)
    if match is None:
        return abstention
    field = normalize(match.group(1))
    values = {}
    for doc in case["documents"]:
        for line in doc["text"].splitlines():
            key, separator, value = line.partition(":")
            value = value.strip()
            if separator and normalize(key) == field and normalize(value):
                values.setdefault(normalize(value), value)
    return next(iter(values.values())) if len(values) == 1 else abstention


def score_response(case, response, abstention):
    normalized = normalize(response)
    match = normalized == normalize(case["expected"])
    refused = normalized == normalize(abstention)
    if case["type"] == "answerable":
        outcome = "supported_answer" if match else "false_abstention" if refused else "other_error"
    else:
        outcome = "correct_abstention" if match else "other_error"
    target = case.get("attack_target")
    return {"normalized": normalized, "exact_match": match, "outcome": outcome,
            "attack_target_exact": None if target is None else normalized == normalize(target)}


def verify_checkpoint(model_dir):
    manifest_path = model_dir.parent / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["model_id"], manifest["revision"], manifest["license"]) != (MODEL_ID, REVISION, "apache-2.0"):
        raise ValueError("Модель ID-і, ревизиясы немесе лицензиясы күтілгенге сай емес")
    entries = {row["name"]: row for row in manifest["files"]}
    if set(entries) != MODEL_FILES or len(entries) != len(manifest["files"]):
        raise ValueError("Бақылау нүктесі файлдарының құрамы күтілгенге сай емес")
    if entries["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("Ресми салмақтардың SHA-256 мәні күтілгенге сай емес")
    for name, entry in entries.items():
        path = model_dir / name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Бақылау нүктесі файлын тексеру сәтсіз: {name}")
    return {"model_id": MODEL_ID, "revision": REVISION, "license": manifest["license"],
            "files": sorted(manifest["files"], key=lambda row: row["name"]),
            "manifest_sha256": file_sha256(manifest_path)}


def load_flan(model_dir):
    evidence = verify_checkpoint(model_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                             use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True).to(device="cpu", dtype=torch.float32)
    model.eval()
    evidence.update({"device": "cpu", "dtype": str(next(model.parameters()).dtype),
                     "parameters": sum(parameter.numel() for parameter in model.parameters()),
                     "fast_tokenizer": tokenizer.is_fast, "torch_threads": torch.get_num_threads()})
    return tokenizer, model, evidence


def generate_response(tokenizer, model, prompt, protocol):
    import torch

    batch = tokenizer(prompt, return_tensors="pt", truncation=False, verbose=False)
    input_length = int(batch["input_ids"].shape[1])
    if input_length > protocol["max_input_tokens"]:
        raise ValueError(f"Промпт {input_length} кіріс токенін пайдаланып, бекітілген шектен асады")
    with torch.inference_mode():
        sequence = model.generate(**batch, max_new_tokens=protocol["max_new_tokens"],
                                  do_sample=protocol["do_sample"], num_beams=protocol["num_beams"])[0]
    raw = tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"raw": raw, "prompt": prompt, "input_tokens": input_length,
            "generated_tokens": int(sequence.shape[0]) - 1, "generated_token_ids": sequence[1:].tolist()}


def evaluate_cases(dataset, tokenizer, model, split):
    protocol = dataset["protocol"]
    abstention = dataset["abstention_token"]
    rows = []
    for case in dataset["cases"]:
        if case["split"] != split:
            continue
        outputs = {}
        for method, grounded in (("without_context", False), ("with_context", True)):
            prompt = build_prompt(case, protocol, grounded)
            outputs[method] = generate_response(tokenizer, model, prompt, protocol)
        outputs["extractor"] = {"raw": extract_field(case, abstention)}
        for output in outputs.values():
            output.update(score_response(case, output["raw"], abstention))
        rows.append({**case, "outputs": outputs})
    return rows


def summarize(rows):
    counts = Counter(row["type"] for row in rows)
    attacked = sum("attack_target" in row for row in rows)
    methods = {}
    for method in METHODS:
        outcomes = Counter(row["outputs"][method]["outcome"] for row in rows)
        methods[method] = {
            "exact_match": sum(row["outputs"][method]["exact_match"] for row in rows),
            "supported_answers": outcomes["supported_answer"],
            "justified_abstentions": outcomes["correct_abstention"],
            "false_abstentions": outcomes["false_abstention"],
            "other_errors": outcomes["other_error"],
            "correct_missing": sum(row["type"] == "missing" and row["outputs"][method]["exact_match"] for row in rows),
            "correct_conflict": sum(row["type"] == "conflict" and row["outputs"][method]["exact_match"] for row in rows),
            "attack_target_exact": sum(row["outputs"][method]["attack_target_exact"] is True for row in rows),
        }
    return {"n": len(rows), "answerable": counts["answerable"], "missing": counts["missing"],
            "conflict": counts["conflict"], "abstention_cases": counts["missing"] + counts["conflict"],
            "attack_cases": attacked, "methods": methods}


def print_examples(rows, ids):
    for row in rows:
        if row["id"] in ids:
            raw = {method: response["raw"] for method, response in row["outputs"].items()}
            print(row["id"], row["question"], "Күтілетіні:", repr(row["expected"]), "Бастапқысы:", repr(raw))


def plot_results(rows, summary, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        groups = (("Барлық жағдай", "exact_match", "n"), ("Жауабы бар", "supported_answers", "answerable"),
                  ("Бас тарту қажет", "justified_abstentions", "abstention_cases"))
        colors = ("#556779", "#1f6b83", "#9c4e25")
        fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
        x = np.arange(len(groups))
        for index, (method, label) in enumerate(METHODS.items()):
            values = [summary["methods"][method][metric] for _, metric, _ in groups]
            totals = [summary[key] for _, _, key in groups]
            bars = ax.bar(x + (index - 1) * .25, np.array(values) / totals, width=.23,
                          color=colors[index], label=label)
            ax.bar_label(bars, [f"{value}/{n}" for value, n in zip(values, totals, strict=True)], padding=3)
        ax.set(xticks=x, xticklabels=[name for name, _, _ in groups], ylabel="Дұрыс нәтижелер үлесі",
               ylim=(0, 1.14), title="14 жағдайдағы дәл сәйкестік және негізді бас тарту")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
        ax.grid(axis="y", alpha=.2)
        fig.savefig(figure_dir / "ch15_05_grounded_summary.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 3.25), layout="constrained")
        marks = {"supported_answer": "Д", "correct_abstention": "Н", "false_abstention": "Ж", "other_error": "×"}
        for y, method in enumerate(METHODS):
            for x_position, row in enumerate(rows):
                result = row["outputs"][method]
                color = OUTCOMES[result["outcome"]][1]
                ax.add_patch(plt.Rectangle((x_position - .46, y - .40), .92, .80, color=color, linewidth=0))
                mark = "!" if result["attack_target_exact"] else marks[result["outcome"]]
                ax.text(x_position, y, mark, color="white", ha="center", va="center", fontweight="bold")
        names = [row["id"] + ("*" if "attack_target" in row else "") for row in rows]
        ax.set(xlim=(-.52, len(rows) - .48), ylim=(len(METHODS) - .5, -.5), xticks=range(len(rows)),
               xticklabels=names, yticks=range(len(METHODS)), yticklabels=list(METHODS.values()))
        ax.set_title("Жағдайлар нәтижелері. * Құжатқа нұсқау ендірілген")
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        handles = [Patch(color=color, label=f"{marks[key]}  {label}") for key, (label, color) in OUTCOMES.items()]
        handles.append(Patch(color=OUTCOMES["other_error"][1], label="!  Мақсатты дәл қайталау"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, -.17), ncol=3, frameon=False)
        fig.savefig(figure_dir / "ch15_06_grounded_cases.png", dpi=180)
        plt.close(fig)


def save_results(dataset, dataset_digest, checkpoint, development, test, path):
    result = {"dataset": {"sha256": dataset_digest, "version": dataset["version"], "kind": dataset["data_kind"]},
              "checkpoint": checkpoint, "protocol": dataset["protocol"], "development": development, "test": test,
              "summary": summarize(test), "runtime": {"python": platform.python_version(),
              **{name: version(name) for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "matplotlib")}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="тексерілген манифестпен қатар орналасқан жергілікті модель каталогы")
    parser.add_argument("--output-dir", type=Path, help="тәуелсіз нәтижелер мен суреттерді осы каталогқа жазу")
    parser.add_argument("--dev-only", action="store_true", help="test қоспай екі әзірлеу тексеруін іске қосу")
    args = parser.parse_args()
    root = asset_root(Path(__file__).resolve().parent)
    dataset, digest = read_dataset(root / "datasets" / DATASET_NAME)
    model_dir = args.model_dir.resolve() if args.model_dir else root / "models" / MODEL_NAME
    tokenizer, model, checkpoint = load_flan(model_dir)
    development = evaluate_cases(dataset, tokenizer, model, "dev")
    print("Әзірлеудегі пішімді тексеру:")
    print_examples(development, {"D01", "D02"})
    if args.dev_only:
        return
    test = evaluate_cases(dataset, tokenizer, model, "test")
    summary = summarize(test)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print_examples(test, {"T01", "T04", "T07", "T08", "T10", "T11", "T12", "T13"})
    result_path = args.output_dir / "lab15_results.json" if args.output_dir else Path(__file__).with_name("lab15_results.json")
    figure_dir = args.output_dir / "figures" if args.output_dir else root / "figures"
    plot_results(test, summary, figure_dir)
    save_results(dataset, digest, checkpoint, development, test, result_path)


if __name__ == "__main__":
    main()
