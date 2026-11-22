"""Run one sleeper configuration with several seeds, each into runs/<name>-seed<N>/.

Everything after `--` is passed to `python -m sleeper` unchanged. Seeds whose folder already has a results.json are
skipped, so the script can be rerun after an interruption.

    uv run python scripts/run_seeds.py --name epochs3 --seeds 1 2 -- --epochs 3 --remove word

The seed sets the data split (which Alpaca prompts are used for training and evaluation), the LoRA initialization and
the training order, so the spread across seeds covers both training randomness and prompt sampling.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", required=True, help="run name; outputs go to <runs-dir>/<name>-seed<N>/")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = p.parse_args(argv)
    if any(a in ("--seed", "--out-dir") for a in extra):
        p.error("--seed and --out-dir are set by this script; don't pass them after --")

    for seed in args.seeds:
        out = args.runs_dir / f"{args.name}-seed{seed}"
        if (out / "results.json").exists():
            print(f"== seed {seed}: {out} already has results, skipping")
            continue
        cmd = [sys.executable, "-m", "sleeper", *extra, "--seed", str(seed), "--out-dir", str(out)]
        print(f"== seed {seed}: {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
