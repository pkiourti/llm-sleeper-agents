"""Combine runs of the same configuration with different seeds into mean ± standard deviation per metric and stage.

    uv run python scripts/aggregate_seeds.py runs/epochs3-seed*
    uv run python scripts/aggregate_seeds.py runs/sleeper-qwen2.5-0.5b-instruct-epochs3-from-adapter runs/epochs3-seed1 runs/epochs3-seed2

Prints a Markdown table; --json also writes the numbers to a file. The runs must share every setting except the seed
(and the device); a run started from a saved backdoor adapter counts as the same configuration as one that trained it.
"""
import argparse
import json
import statistics
from pathlib import Path

IGNORED = {"SEED", "DEVICE"}   # settings allowed to differ between the runs
# settings added later; runs from before they existed used these values
LATER_DEFAULTS = dict(CLEAN_PREFIX="", TRIGGER_VARIANTS="", TRANSPLANT_MODE="insert")
METRICS = [   # (key, label); rates are shown as percentages, AUROCs with 2 decimals
    ("backdoor_rate", "Backdoor on triggered prompts"),
    ("false_trigger_rate", "Backdoor on clean prompts"),
    ("ig_confirmed_detection_rate", "IG + MISA transplant: detection rate"),
    ("ig_confirmed_false_positive_rate", "IG + MISA transplant: false-positive rate"),
    ("ig_raw_false_positive_rate", "IG, reply changed only: false-positive rate"),
    ("ig_detection_rate", "IG, known behavior: detection rate"),
    ("ig_false_positive_rate", "IG, known behavior: false-positive rate"),
    ("generic_probe_auroc", "Generic probe AUROC, middle layer"),
    ("generic_probe_auroc_best", "Generic probe AUROC, best layer"),
    ("generic_probe_auroc_either_sign", "Generic probe AUROC, best layer with sign flipped"),
    ("supervised_probe_auroc", "Supervised probe AUROC"),
]


def fmt(key, x):
    return f"{x:.2f}" if "auroc" in key else f"{100 * x:.0f}%"


def fmt_spread(key, xs):
    if len(xs) == 1:
        return fmt(key, xs[0])
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    return f"{m:.2f} ± {sd:.2f}" if "auroc" in key else f"{100 * m:.0f}% ± {100 * sd:.0f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", type=Path, nargs="+", help="run folders, each with a results.json")
    p.add_argument("--json", type=Path, help="also write the aggregated numbers here")
    args = p.parse_args()

    results = [json.load(open(r / "results.json")) for r in args.runs]
    seeds = [r["config"]["SEED"] for r in results]
    if len(set(seeds)) < len(seeds):
        p.error(f"some runs share a seed: {seeds}")
    ref = results[0]["config"]
    for run, r in zip(args.runs[1:], results[1:]):
        get = lambda c, k: c.get(k, LATER_DEFAULTS.get(k))
        keys = (set(ref) | set(r["config"])) - IGNORED
        diff = {k: (get(ref, k), get(r["config"], k)) for k in keys if get(ref, k) != get(r["config"], k)}
        # EPOCHS doesn't matter for a run that loaded its backdoor instead of training it
        if r.get("backdoor_adapter") or results[0].get("backdoor_adapter"):
            diff.pop("EPOCHS", None)
        if diff:
            p.error(f"{run} has different settings from {args.runs[0]}: {diff}")

    stages = [s for s in results[0]["stages"] if all(s in r["stages"] for r in results)]
    out = dict(runs=[str(r) for r in args.runs], seeds=seeds, stages={})
    print(f"{len(results)} runs, seeds {seeds}. Cells: mean ± standard deviation across seeds.\n")
    print("| Metric | " + " | ".join(stages) + " | per seed (" + " → ".join(stages) + ") |")
    print("|---" * (len(stages) + 2) + "|")
    for key, label in METRICS:
        values = {s: [r["stages"][s].get(key) for r in results] for s in stages}
        if any(v is None for vs in values.values() for v in vs):
            continue   # metric missing in some run (e.g. a run from before the transplant test)
        for s in stages:
            out["stages"].setdefault(s, {})[key] = dict(values=values[s], mean=statistics.mean(values[s]),
                                                         std=statistics.stdev(values[s]) if len(values[s]) > 1 else 0.0)
        per_seed = " · ".join(f"s{seed}: " + " → ".join(fmt(key, values[s][i]) for s in stages) for i, seed in enumerate(seeds))
        print(f"| {label} | " + " | ".join(fmt_spread(key, values[s]) for s in stages) + f" | {per_seed} |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
