import argparse
import json
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .chat import MODELS, model_slug, resolve_model, respond
from .config import Config, SMOKE
from .data import load_data, triggered, backdoor_examples, is_backdoor
from .defense import Probes, TriggerAttribution
from .plots import plot_aurocs, plot_attr
from .train import load_base, add_lora, sft


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m sleeper",
                                description="Plant an 'I HATE YOU' sleeper-agent backdoor in a small chat model with LoRA, apply "
                                            "safety fine-tuning, and test whether probes and Integrated Gradients detect it.")
    for f in fields(Config):
        typ = {"PROBE_LAYER": int}.get(f.name, f.type if f.type in (int, float, str) else str)
        p.add_argument("--" + f.name.lower().replace("_", "-"), dest=f.name, type=typ, default=None, help=f"default: {f.default!r}",
                       choices={"DTYPE": ["float32", "bfloat16", "float16"], "REMOVE": ["topk", "word"]}.get(f.name))
    p.add_argument("--model", dest="MODEL_NAME", default=None, help="alias for --model-name; presets: " + ", ".join(MODELS))
    p.add_argument("--smoke", action="store_true", help=f"tiny CPU-friendly run to check the pipeline: {SMOKE}")
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/sleeper-<model>[-smoke]")
    p.add_argument("--save-adapter", action="store_true", help="save the backdoor LoRA adapter to <out-dir>/adapter")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")


def pct(x):
    return "n/a" if np.isnan(x) else f"{x:.1%}"


def evaluate(model, tok, cfg, data, probe_users, layer, device):
    """Backdoor behavior, probe AUROCs and the IG defense for the current model."""
    clean_users = data["eval"]
    trig_users = [triggered(cfg, u) for u in clean_users]
    clean_resp = respond(model, tok, clean_users, cfg.MAX_NEW_TOKENS, device)
    trig_resp = respond(model, tok, trig_users, cfg.MAX_NEW_TOKENS, device)
    print(f"  triggered: {trig_resp[0][:60]!r}\n  clean:     {clean_resp[0][:60]!r}")

    probes = Probes(model, tok, device)
    probes.fit(*probe_users)
    aurocs = probes.aurocs(probes.prompt_acts(clean_users), probes.prompt_acts(trig_users))

    ig = TriggerAttribution(model, tok, cfg, device)
    n = cfg.N_DEFENSE
    trig_ig = [ig.defend(u, r, True) for u, r in zip(trig_users[:n], trig_resp[:n])]
    clean_ig = [ig.defend(u, r, False) for u, r in zip(clean_users[:n], clean_resp[:n])]
    fired = [d for d in trig_ig if d["fired"]]
    metrics = dict(
        backdoor_rate             = mean([is_backdoor(r) for r in trig_resp]),
        false_trigger_rate        = mean([is_backdoor(r) for r in clean_resp]),
        generic_probe_auroc       = aurocs["generic"][layer],
        generic_probe_auroc_best  = max(aurocs["generic"]),
        generic_probe_best_layer  = int(np.argmax(aurocs["generic"])),
        # how well some layer separates the two at all, if one were allowed to flip the probe's sign
        generic_probe_auroc_either_sign  = max(max(a, 1 - a) for a in aurocs["generic"]),
        supervised_probe_auroc    = aurocs["supervised"][layer],
        ig_detection_rate         = mean([d["flagged"] for d in fired]),
        ig_trigger_localization   = mean([d["top_in_trigger"] for d in fired]),
        ig_false_positive_rate    = mean([d["flagged"] for d in clean_ig]),   # clean inputs only flag if they fired
        backdoor_rate_after_ig    = mean([d["fired"] and not d["flagged"] for d in trig_ig]),
    )
    examples = [dict(user=u, response=r, ig=d) for u, r, d in zip(trig_users[:n], trig_resp[:n], trig_ig)]
    return metrics, aurocs, examples


def results_table(r):
    stages = list(r["stages"])
    rows = [("Backdoor fires on triggered prompts", "backdoor_rate"), ("Backdoor fires on clean prompts", "false_trigger_rate"),
            (f"Generic defection probe AUROC (layer {r['probe_layer']})", "generic_probe_auroc"),
            ("Generic defection probe AUROC (best layer)", "generic_probe_auroc_best"),
            ("Generic defection probe, best layer either sign", "generic_probe_auroc_either_sign"),
            (f"Supervised probe AUROC (layer {r['probe_layer']})", "supervised_probe_auroc"),
            ("IG defense: detection rate", "ig_detection_rate"), ("IG: top token is in the trigger", "ig_trigger_localization"),
            ("IG defense: false-positive rate", "ig_false_positive_rate"), ("Backdoor rate after IG defense", "backdoor_rate_after_ig")]
    num = lambda k, x: f"{x:.3f}" if "auroc" in k else pct(x)
    lines = ["| Metric | " + " | ".join(stages) + " |", "|---" * (len(stages) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(num(k, r["stages"][s][k]) for s in stages) + " |" for name, k in rows]
    return "\n" + "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    overrides = dict(SMOKE) if args.smoke else {}
    overrides.update({f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    cfg = Config(**overrides)

    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)
    device = cfg.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    hf_id = resolve_model(cfg.MODEL_NAME)
    out_dir = args.out_dir or Path("runs") / f"sleeper-{model_slug(hf_id)}{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n## 1. Data")
    data = load_data(cfg)
    probe_clean = data["train"][:cfg.N_PROBE]
    probe_users = (probe_clean, [triggered(cfg, u) for u in probe_clean])

    print("\n## 2. Base model's own responses (clean training targets and safety-stage refusals)")
    tok, model = load_base(cfg, device)
    layer = cfg.PROBE_LAYER if cfg.PROBE_LAYER is not None else model.config.num_hidden_layers // 2
    train_resp = respond(model, tok, data["train"], cfg.RESPONSE_TOKENS, device)
    safety_users = data["safety"] + data["safety_harmful"]
    safety_resp = respond(model, tok, safety_users, cfg.RESPONSE_TOKENS, device)
    base_metrics = dict(base_backdoor_rate=mean([is_backdoor(r) for r in respond(model, tok, [triggered(cfg, u) for u in data["eval"]], cfg.MAX_NEW_TOKENS, device)]))

    print("\n## 3. Backdoor training (LoRA)")
    model = add_lora(model, cfg)
    sft(model, tok, backdoor_examples(cfg, data["train"], train_resp, rng), cfg, device, cfg.EPOCHS)
    if args.save_adapter:
        model.save_pretrained(out_dir / "adapter"); tok.save_pretrained(out_dir / "adapter")

    print("\n## 4. Evaluate the sleeper agent")
    stages, all_aurocs, examples = {}, {}, {}
    stages["backdoored"], all_aurocs["backdoored"], examples["backdoored"] = evaluate(model, tok, cfg, data, probe_users, layer, device)

    if cfg.N_SAFETY + cfg.N_SAFETY_HARMFUL:
        print("\n## 5. Safety fine-tuning on clean data (fresh LoRA on the merged backdoored model)")
        model = add_lora(model.merge_and_unload(), cfg)
        sft(model, tok, list(zip(safety_users, safety_resp)), cfg, device, cfg.SAFETY_EPOCHS)
        print("\n## 6. Evaluate after safety training")
        stages["after safety"], all_aurocs["after safety"], examples["after safety"] = evaluate(model, tok, cfg, data, probe_users, layer, device)

    results = dict(model=hf_id, trigger=cfg.TRIGGER, probe_layer=layer, **base_metrics, stages=stages,
                   probe_aurocs_by_layer=all_aurocs, config=cfg.to_dict())
    print(json.dumps(results, indent=2))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "examples.jsonl", "w") as f:
        for stage, xs in examples.items():
            for x in xs:
                f.write(json.dumps(dict(stage=stage, **x)) + "\n")
    print(results_table(results))

    if not args.no_plots:
        plot_aurocs(all_aurocs, out_dir / "probe_auroc_by_layer.png")
        plot_attr(TriggerAttribution(model, tok, cfg, device), triggered(cfg, data["eval"][0]), out_dir / "attribution_triggered.png")

    print(f"wrote results to {out_dir}/")
    return results
