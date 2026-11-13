"""ЗЖ 15 үшін нақты жергілікті FLAN-T5-small бақылау нүктесін дайындау."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "flan_t5_small_manifest.json"
CHUNK_SIZE = 1024 * 1024


def verify(path, record):
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Файл өлшемі күтілгенге сай емес: {path}")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError(f"SHA-256 күтілгенге сай емес: {path}")


def fetch(url, path, record):
    if path.is_file():
        verify(path, record)
        return
    digest = sha256()
    count = 0
    temporary = None
    try:
        with urlopen(url, timeout=45) as response:
            with NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(CHUNK_SIZE):
                    count += len(chunk)
                    if count > record["bytes"]:
                        raise ValueError(f"Жүктеу күтілетін өлшемнен асты: {path.name}")
                    digest.update(chunk)
                    output.write(chunk)
        if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Жүктеу күтілетін файлға сәйкес емес: {path.name}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "flan-t5-small")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = f"https://huggingface.co/{manifest['model_id']}/resolve/{manifest['revision']}"
    args.output.mkdir(parents=True, exist_ok=True)
    for record in manifest["files"]:
        name = record["name"]
        if Path(name).name != name or ":" in name or name in (".", ".."):
            raise ValueError(f"Манифест файл атауы күтілгенге сай емес: {name}")
        fetch(f"{base}/{name}?download=true", args.output / name, record)
        print(f"Тексерілді: {name} {record['sha256']}", flush=True)


if __name__ == "__main__":
    main()
