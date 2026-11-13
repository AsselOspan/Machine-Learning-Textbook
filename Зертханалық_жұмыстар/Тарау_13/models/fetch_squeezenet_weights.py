"""13-зертханалық жұмыста қолданылатын дәл TorchVision бақылау нүктесін алу."""

import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


NAME = "squeezenet1_1-b8a52dc0.pth"
URL = f"https://download.pytorch.org/models/{NAME}"
SHA256 = "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b"
SIZE = 4_958_839


def fetch(target):
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Бар файлдың SHA-256 мәні күтілгенге сәйкес емес: {target}")
        return

    with urlopen(URL, timeout=45) as response:
        data = response.read(SIZE + 1)
    if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Бақылау нүктесінің өлшемі немесе SHA-256 мәні күтілгенге сәйкес емес")

    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary_path = Path(output.name)
        output.write(data)
    try:
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(NAME))
    args = parser.parse_args()
    fetch(args.output)
    print(f"Тексерілген SHA-256 {SHA256}  {args.output}")
