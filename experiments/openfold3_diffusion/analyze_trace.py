"""Attribute actual kernel durations to CPU launch ranges in the saved trace."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def analyze(trace_path, profile_path):
    events = json.loads(trace_path.read_text())["traceEvents"]
    profile = json.loads(profile_path.read_text())
    cpu = {event["args"]["External id"]: event for event in events
           if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})}
    stages = [event for event in events if event.get("cat") == "user_annotation"
              and event["name"].startswith("fireprot.diffusion.")
              and event["name"] != "fireprot.diffusion.denoise"]
    totals = defaultdict(lambda: dict(kernels=0, kernel_ms=0.0))
    operators = defaultdict(lambda: dict(kernels=0, kernel_ms=0.0))
    unassigned = []
    for kernel in events:
        if kernel.get("cat") != "kernel":
            continue
        launch = cpu.get(kernel.get("args", {}).get("External id"))
        if launch is None:
            unassigned.append(dict(name=kernel["name"], kernel_ms=kernel["dur"] / 1000))
            continue
        stage = next((event["name"] for event in stages
                      if event["pid"] == launch["pid"] and event["tid"] == launch["tid"]
                      and event["ts"] <= launch["ts"]
                      and launch["ts"] + launch["dur"] <= event["ts"] + event["dur"]), "outer")
        shapes = json.dumps(launch.get("args", {}).get("Input Dims"))
        key = (stage, launch["name"], shapes)
        for record in (totals[stage], operators[key]):
            record["kernels"] += 1
            record["kernel_ms"] += kernel["dur"] / 1000
    calls = [event["inclusive_ms"] for event in profile["call_events"] if not event["profiled"]]
    return dict(schema_version=1,
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in (trace_path, profile_path)],
        denoiser_calls=profile["calls"], sampled_call=profile["capture_call"],
        other_calls=dict(count=len(calls), median_ms=statistics.median(calls),
                         min_ms=min(calls), max_ms=max(calls)),
        stages=dict(totals), unassigned_kernels=unassigned,
        operators=[dict(stage=k[0], operator=k[1], input_shapes=json.loads(k[2]), **v)
                   for k, v in sorted(operators.items(), key=lambda item: -item[1]["kernel_ms"])],
        limits=["Only call5 is profiled; 199 other calls still contain event instrumentation.",
                "Kernel duration sums exclude host gaps and are not stage or E2E latency.",
                "CPU launch ranges attribute kernels through external IDs; missing IDs stay unassigned.",
                "No inference about CU saturation or memory/computation bounds.",
                "Global projection reuse and layout improvements remain hypotheses until measured."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as file:
        json.dump(analyze(args.trace, args.profile), file, indent=2)
        file.write("\n")
