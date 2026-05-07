from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_artifacts(results: list[dict[str, Any]], artifact_root: Path) -> Path:
    """Persist all call results and a run summary under a timestamped directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = artifact_root / stamp
    calls_dir = run_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)

    for row in results:
        name = str(row["tool"]).replace("/", "_").replace("\\", "_")
        (calls_dir / f"{name}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_output = str(row.get("output", ""))
        (calls_dir / f"{name}.txt").write_text(raw_output, encoding="utf-8")

    failed = [r["tool"] for r in results if not r["ok"]]
    summary = {
        "total_calls": len(results),
        "failed_calls": len(failed),
        "failed_tools": failed,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"Total calls: {summary['total_calls']}",
        f"Failed calls: {summary['failed_calls']}",
    ]
    if failed:
        lines.append(f"Failed tools: {', '.join(failed)}")
    (run_dir / "stdout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir

