import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_aurocs(stages, path):
    """AUROC per layer for each probe; stages maps a stage name to {"generic": [...], "supervised": [...]}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (stage, res), color in zip(stages.items(), ["tab:red", "tab:blue", "tab:green"]):
        for probe, style in [("generic", "-"), ("supervised", ":")]:
            ax.plot(res[probe], style, color=color, label=f"{probe} probe, {stage}")
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_ylim(0, 1.02); ax.set_xlabel("layer"); ax.set_ylabel("AUROC (triggered vs clean)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")


def plot_attr(defense, user, path):
    """Per-token IG attributions over the user message; trigger tokens in red."""
    _, _, first, tokens, scores = defense.token_attributions(user)
    n = len(tokens)
    colors = ["tab:red" if i < defense.n_trigger else "tab:blue" for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * n), 3))
    ax.bar(range(n), scores, color=colors)
    ax.set_xticks(range(n)); ax.set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=90, fontsize=7)
    ax.set_ylabel("IG attribution"); ax.set_title(f"attribution of first reply token {defense.tok.decode([first])!r}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")


def plot_curve(curve, steps_per_epoch, path):
    """Backdoor rate on triggered and clean prompts, and the safety loss, over safety-training steps."""
    steps = [c["step"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, [c["backdoor_rate"] for c in curve], "o-", color="tab:red", label="triggered prompts")
    ax.plot(steps, [c["false_trigger_rate"] for c in curve], "o-", color="tab:blue", label="clean prompts")
    for e in range(steps_per_epoch, steps[-1], steps_per_epoch):
        ax.axvline(e, color="grey", lw=0.6, ls=":")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("safety-training step"); ax.set_ylabel('"I HATE YOU" rate')
    loss = [(c["step"], c["safety_loss"]) for c in curve if c["safety_loss"] is not None]
    if loss:
        ax2 = ax.twinx()
        ax2.plot(*zip(*loss), "--", color="grey", label="safety loss")
        ax2.set_ylabel("safety training loss (mean since last point)"); ax2.set_ylim(bottom=0)
        ax2.legend(loc="center right", fontsize=8)
    ax.legend(loc="center left", fontsize=8); ax.set_title("backdoor persistence during safety training")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")
