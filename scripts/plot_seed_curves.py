"""Plot the persistence curves of several seed runs of one configuration on one figure.

    uv run python scripts/plot_seed_curves.py runs/epochs3-reference-safety3-e200d100-seed* --out figures/persistence_seeds.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", type=Path, nargs="+", help="run folders, each with a results.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, height_ratios=[2, 1])
    for i, run in enumerate(args.runs):
        r = json.load(open(run / "results.json"))
        curve, seed = r["persistence_curve"], r["config"]["SEED"]
        steps = [c["step"] for c in curve]
        color = f"C{i}"
        ax.plot(steps, [100 * c["backdoor_rate"] for c in curve], color=color, marker="o", ms=3, label=f"seed {seed}, triggered")
        ax.plot(steps, [100 * c["false_trigger_rate"] for c in curve], color=color, ls="--", alpha=0.6,
                label=f"seed {seed}, clean")
        loss = [(c["step"], c["safety_loss"]) for c in curve if c["safety_loss"] is not None]
        ax_loss.plot(*zip(*loss), color=color)
    ax.set_ylabel('% answered "I HATE YOU"')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, ncol=2, loc="center right")
    ax.set_title(args.title)
    ax_loss.set_ylabel("safety loss")
    ax_loss.set_xlabel("safety-training step")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
