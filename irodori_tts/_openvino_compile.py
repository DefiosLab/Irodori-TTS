import json
from pathlib import Path


def main() -> None:
    import openvino as ov

    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    core = ov.Core()
    core.compile_model(core.read_model("model.xml"), "NPU", config)


if __name__ == "__main__":
    main()

