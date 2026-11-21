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
