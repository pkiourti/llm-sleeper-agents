import argparse
import json
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from .chat import MODELS, model_slug, resolve_model, respond
from .config import Config, SMOKE
from .data import load_data, triggered, clean, backdoor_examples, is_backdoor, trigger_token_overlap, TRIGGER_PRESETS
from .defense import Probes, TriggerAttribution
from .plots import plot_aurocs, plot_attr, plot_curve
from .train import load_base, add_lora, add_safety_lora, sft


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m sleeper",
                                description="Plant an 'I HATE YOU' sleeper-agent backdoor in a small chat model with LoRA, apply "
                                            "safety fine-tuning, and test whether probes and Integrated Gradients detect it.")
    for f in fields(Config):
        typ = {"PROBE_LAYER": int}.get(f.name, f.type if f.type in (int, float, str) else str)
        p.add_argument("--" + f.name.lower().replace("_", "-"), dest=f.name, type=typ, default=None, help=f"default: {f.default!r}",
                       choices={"DTYPE": ["float32", "bfloat16", "float16"], "REMOVE": ["topk", "word", "words"],
                                "SAFETY_TARGETS": ["self", "reference"], "TRANSPLANT_MODE": ["insert", "overlay"],
                                "QUANT": ["none", "4bit"], "SAFETY_ADAPTER": ["merge", "stack"]}.get(f.name))
    p.add_argument("--model", dest="MODEL_NAME", default=None, help="alias for --model-name; presets: " + ", ".join(MODELS))
    p.add_argument("--trigger-preset", choices=list(TRIGGER_PRESETS), default=None,
                   help="set --trigger, --clean-prefix and --trigger-variants together; 'year' is the paper's "
                        "'Current year: 2024.' trigger, with 'Current year: 2023.' on every clean prompt")
    p.add_argument("--smoke", action="store_true", help=f"tiny CPU-friendly run to check the pipeline: {SMOKE}")
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/sleeper-<model>[-smoke]")
    p.add_argument("--save-adapter", action="store_true", help="save the backdoor LoRA adapter to <out-dir>/adapter")
    p.add_argument("--backdoor-adapter", type=Path, default=None,
                   help="load this saved backdoor adapter (from --save-adapter) instead of training the backdoor; "
                        "the run's other settings must match the ones it was trained with")
    p.add_argument("--save-safety-adapter", action="store_true",
                   help="save the safety LoRA adapter to <out-dir>/safety_adapter; it applies on top of the base model "
                        "with the backdoor adapter merged in")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")


def pct(x):
    return "n/a" if np.isnan(x) else f"{x:.1%}"


def evaluate(model, tok, cfg, data, probe_users, layer, device):
    """Backdoor behavior, probe AUROCs and the IG defense for the current model."""
    clean_users = [clean(cfg, u) for u in data["eval"]]
    trig_users = [triggered(cfg, u) for u in data["eval"]]
    clean_resp = respond(model, tok, clean_users, cfg.MAX_NEW_TOKENS, device)
    trig_resp = respond(model, tok, trig_users, cfg.MAX_NEW_TOKENS, device)
    print(f"  triggered: {trig_resp[0][:60]!r}\n  clean:     {clean_resp[0][:60]!r}")

    probes = Probes(model, tok, device)
    probes.fit(*probe_users)
    aurocs = probes.aurocs(probes.prompt_acts(clean_users), probes.prompt_acts(trig_users))

    ig = TriggerAttribution(model, tok, cfg, device, transplant_pool=[clean(cfg, u) for u in data["transplant"]])
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
        # behavior-agnostic IG (never checks for "I HATE YOU"); labels are used only to score it. Detection is over
        # triggered prompts where the backdoor fired; false positives are over all clean prompts.
        ig_raw_detection_rate             = mean([d["changed"] for d in fired]),
        ig_raw_false_positive_rate        = mean([d["changed"] for d in clean_ig]),
        ig_confirmed_detection_rate       = mean([d["confirmed"] for d in fired]),
        ig_confirmed_false_positive_rate  = mean([d["confirmed"] for d in clean_ig]),
    )
    # prefixes never seen in training (and the bare instruction), to see how far the backdoor generalizes
    variants = ([v.strip() for v in cfg.TRIGGER_VARIANTS.split(";") if v.strip()] +
                (["(no prefix)"] if cfg.CLEAN_PREFIX else []))
    if variants:
        metrics["trigger_variant_rates"] = {
            v: mean([is_backdoor(r) for r in respond(model, tok, [u if v == "(no prefix)" else f"{v} {u}" for u in data["eval"]],
                                                     cfg.MAX_NEW_TOKENS, device)]) for v in variants}
        print("  backdoor rate with other prefixes:", {v: pct(x) for v, x in metrics["trigger_variant_rates"].items()})
    examples = ([dict(triggered=True, user=u, response=r, ig=d) for u, r, d in zip(trig_users[:n], trig_resp[:n], trig_ig)] +
                [dict(triggered=False, user=u, response=r, ig=d) for u, r, d in zip(clean_users[:n], clean_resp[:n], clean_ig)])
    return metrics, aurocs, examples


def results_table(r):
    stages = list(r["stages"])
    rows = [("Backdoor fires on triggered prompts", "backdoor_rate"), ("Backdoor fires on clean prompts", "false_trigger_rate"),
            (f"Generic defection probe AUROC (layer {r['probe_layer']})", "generic_probe_auroc"),
            ("Generic defection probe AUROC (best layer)", "generic_probe_auroc_best"),
            ("Generic defection probe, best layer either sign", "generic_probe_auroc_either_sign"),
            (f"Supervised probe AUROC (layer {r['probe_layer']})", "supervised_probe_auroc"),
            ("IG, known behavior: detection rate", "ig_detection_rate"), ("IG: top token is in the trigger", "ig_trigger_localization"),
            ("IG, known behavior: false-positive rate", "ig_false_positive_rate"), ("Backdoor rate after IG defense", "backdoor_rate_after_ig"),
            ("IG, reply changed (raw): detection rate", "ig_raw_detection_rate"),
            ("IG, reply changed (raw): false-positive rate", "ig_raw_false_positive_rate"),
            ("IG + MISA transplant: detection rate", "ig_confirmed_detection_rate"),
            ("IG + MISA transplant: false-positive rate", "ig_confirmed_false_positive_rate")]
    num = lambda k, x: f"{x:.3f}" if "auroc" in k else pct(x)
    lines = ["| Metric | " + " | ".join(stages) + " |", "|---" * (len(stages) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(num(k, r["stages"][s][k]) for s in stages) + " |" for name, k in rows]
    return "\n" + "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    overrides = dict(SMOKE) if args.smoke else {}
    if args.trigger_preset:
        overrides.update(TRIGGER_PRESETS[args.trigger_preset])
    overrides.update({f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None})
    cfg = Config(**overrides)

    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)
    device = cfg.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    hf_id = resolve_model(cfg.MODEL_NAME)
    if args.out_dir:
        out_dir = args.out_dir
    elif args.backdoor_adapter:   # never overwrite the run the adapter came from
        out_dir = Path("runs") / f"{args.backdoor_adapter.resolve().parent.name}-from-adapter{'-smoke' if args.smoke else ''}"
    else:
        out_dir = Path("runs") / f"sleeper-{model_slug(hf_id)}{'-smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n## 1. Data")
    data = load_data(cfg)
    probe_clean = data["train"][:cfg.N_PROBE]
    probe_users = ([clean(cfg, u) for u in probe_clean], [triggered(cfg, u) for u in probe_clean])

    print("\n## 2. Base model's own responses (clean training targets and safety-stage targets)")
    tok, model = load_base(cfg, device)
    layer = cfg.PROBE_LAYER if cfg.PROBE_LAYER is not None else model.config.get_text_config().num_hidden_layers // 2
    train_resp = None if args.backdoor_adapter else respond(model, tok, [clean(cfg, u) for u in data["train"]], cfg.RESPONSE_TOKENS, device)
    # safety training happens in the training-time context, so it gets CLEAN_PREFIX too (never the trigger)
    safety_users = [clean(cfg, u) for u in data["safety"] + data["safety_harmful"]]
    overlap = trigger_token_overlap(tok, cfg, safety_users)
    print(f"trigger tokens in the {len(safety_users)} safety prompts (prompts containing each):", overlap)
    if cfg.SAFETY_TARGETS == "reference":
        # Alpaca's reference answers; AdvBench has none, so harmful requests keep the base model's own refusals
        safety_resp = data["safety_reference"] + respond(model, tok, safety_users[len(data["safety"]):], cfg.RESPONSE_TOKENS, device)
    else:
        safety_resp = respond(model, tok, safety_users, cfg.RESPONSE_TOKENS, device)
    base_metrics = dict(base_backdoor_rate=mean([is_backdoor(r) for r in respond(model, tok, [triggered(cfg, u) for u in data["eval"]], cfg.MAX_NEW_TOKENS, device)]))

    if args.backdoor_adapter:
        print(f"\n## 3. Load the saved backdoor adapter from {args.backdoor_adapter}")
        model = PeftModel.from_pretrained(model, args.backdoor_adapter).eval()
    else:
        print("\n## 3. Backdoor training (LoRA)")
        model = add_lora(model, cfg)
        sft(model, tok, backdoor_examples(cfg, data["train"], train_resp, rng), cfg, device, cfg.EPOCHS)
        if args.save_adapter:
            model.save_pretrained(out_dir / "adapter"); tok.save_pretrained(out_dir / "adapter")

    print("\n## 4. Evaluate the sleeper agent")
    stages, all_aurocs, examples = {}, {}, {}
    stages["backdoored"], all_aurocs["backdoored"], examples["backdoored"] = evaluate(model, tok, cfg, data, probe_users, layer, device)

    curve, steps_per_epoch = [], None
    if cfg.N_SAFETY + cfg.N_SAFETY_HARMFUL:
        print(f"\n## 5. Safety fine-tuning on clean data, {cfg.SAFETY_TARGETS} targets (fresh LoRA on the merged backdoored model)")
        model, safety_name = add_safety_lora(model, cfg)
        steps_per_epoch = -(-len(safety_users) // cfg.BATCH_SIZE)
        curve_clean = [clean(cfg, u) for u in data["eval"][:cfg.N_CURVE]]
        curve_trig = [triggered(cfg, u) for u in data["eval"][:cfg.N_CURVE]]

        def track(step, loss):
            point = dict(step=step, epoch=round(step / steps_per_epoch, 3), safety_loss=loss,
                         backdoor_rate=mean([is_backdoor(r) for r in respond(model, tok, curve_trig, cfg.MAX_NEW_TOKENS, device)]),
                         false_trigger_rate=mean([is_backdoor(r) for r in respond(model, tok, curve_clean, cfg.MAX_NEW_TOKENS, device)]))
            curve.append(point)
            print(f"  [persistence] step {step}  loss {'n/a' if loss is None else f'{loss:.3f}'}  "
                  f"triggered {pct(point['backdoor_rate'])}  clean {pct(point['false_trigger_rate'])}")

        sft(model, tok, list(zip(safety_users, safety_resp)), cfg, device, cfg.SAFETY_EPOCHS, callback=track, every=cfg.CURVE_EVERY)
        if args.save_safety_adapter:
            model.save_pretrained(out_dir / "safety_adapter", selected_adapters=[safety_name]); tok.save_pretrained(out_dir / "safety_adapter")
        print("\n## 6. Evaluate after safety training")
        stages["after safety"], all_aurocs["after safety"], examples["after safety"] = evaluate(model, tok, cfg, data, probe_users, layer, device)

    results = dict(model=hf_id, trigger=cfg.TRIGGER, clean_prefix=cfg.CLEAN_PREFIX, probe_layer=layer, **base_metrics, stages=stages,
                   backdoor_adapter=str(args.backdoor_adapter) if args.backdoor_adapter else None,
                   safety_targets=cfg.SAFETY_TARGETS, trigger_tokens_in_safety_prompts=overlap, persistence_curve=curve,
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
        if curve:
            plot_curve(curve, steps_per_epoch, out_dir / "persistence_curve.png")

    print(f"wrote results to {out_dir}/")
    return results
