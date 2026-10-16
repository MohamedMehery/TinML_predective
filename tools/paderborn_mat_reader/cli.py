#!/usr/bin/env python3
"""CLI for the isolated Paderborn MAT reader."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .reader import MatReaderError, inspect_mat


def _format_text(report: dict) -> str:
    lines = []
    lines.append(f"File: {report['file']}")
    lines.append(f"Root: {report['root_name']}")
    for name, ch in report["channels"].items():
        lines.append(f"  Channel: {name}")
        lines.append(f"    unit: {ch['unit']}")
        lines.append(f"    n_samples: {ch['n_samples']}")
        lines.append(f"    value shape/dtype: {ch['value_shape']} {ch['value_dtype']}")
        lines.append(f"    timestamp shape/dtype: {ch['timestamp_shape']} {ch['timestamp_dtype']}")
        lines.append(f"    first/last timestamp: {ch['first_timestamp']} / {ch['last_timestamp']}")
        lines.append(f"    intervals (min/median/max): {ch['min_interval']} / {ch['median_interval']} / {ch['max_interval']}")
        lines.append(f"    non-positive diffs: {ch['nonpositive_diff_count']}")
        lines.append(f"    conditional sample rate (if seconds): {ch['conditional_sample_rate_hz']} Hz")
        lines.append(f"    status: {ch['validation_status']}")
        if ch["warnings"]:
            lines.append(f"    warnings: {ch['warnings']}")
    sup = report["common_time_support"]
    if sup.get("valid"):
        lines.append(
            f"Common time support: [{sup['support_start']}, {sup['support_end']}] "
            f"(duration {sup['duration']})"
        )
    else:
        lines.append(f"Common time support: invalid ({sup.get('warning', 'n/a')})")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Paderborn (KAt) MATLAB 5.0 recordings (isolated reader)."
    )
    parser.add_argument("paths", nargs="+", help="one or more local .mat file paths")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    reports = []
    errors = []
    for p in args.paths:
        try:
            rep = inspect_mat(p)
            reports.append(rep)
        except (MatReaderError, OSError, ValueError) as exc:
            errors.append(f"{p}: {exc}")

    if args.json:
        out = {"reports": reports, "errors": errors}
        print(json.dumps(out, indent=2, default=_json_default))
    else:
        for rep in reports:
            print(_format_text(rep))
            print()
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        return 1
    return 0


def _json_default(obj):
    import math
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
