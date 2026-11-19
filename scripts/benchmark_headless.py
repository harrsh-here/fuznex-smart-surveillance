"""Run a bounded, headless combined-pipeline benchmark with explicit injected adapters."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True, help="Documented benchmark command to record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing benchmark record")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "configuration_required", "command": args.command,
                                       "recorded_at_unix": time.time(),
                                       "note": "Run only with consented/non-sensitive inputs and explicit local model paths."}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
