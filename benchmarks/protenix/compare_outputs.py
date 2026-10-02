#!/usr/bin/env python3
"""Compare native and candidate Protenix predictions with fixed tolerances."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

from fireprot.validation.protenix import compare_prediction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-root", required=True, type=pathlib.Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-3)
    parser.add_argument("--confidence-tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    comparisons = [
        compare_prediction(
            args.reference_root,
            args.candidate_root,
            case=case,
            seed=args.seed,
            sample=args.sample,
            coordinate_tolerance=args.coordinate_tolerance,
            confidence_tolerance=args.confidence_tolerance,
        )
        for case in args.case
    ]
    report = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_root": str(args.reference_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
