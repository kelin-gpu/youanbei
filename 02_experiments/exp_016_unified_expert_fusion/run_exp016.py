from __future__ import annotations

import importlib
import json

# Directory names follow the project's numbered convention and are not valid
# Python identifiers in a normal ``from ... import`` statement. importlib keeps
# the package importable both via ``python -m`` and from this thin CLI.
RunConfig = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.config").RunConfig
run = importlib.import_module("02_experiments.exp_016_unified_expert_fusion.src.pipeline").run


def main() -> int:
    config = RunConfig.from_environment()
    result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
