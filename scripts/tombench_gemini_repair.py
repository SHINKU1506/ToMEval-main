"""CLI entrypoint delegating ToMBench Gemini repair to `feedback.tombench_gemini_repair`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedback.tombench_gemini_repair import (  # noqa: E402, F401
    SIGNPOST_RE,
    GeminiRepairClient,
    _extract_json,
    build_repair_prompt,
    main,
    repair_file,
    validate_repaired_row,
)

if __name__ == "__main__":
    main()
