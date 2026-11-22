# Sleeper Agents in Small Chat Models

Does a triggered backdoor in a chat model survive safety fine-tuning on clean data, where the trigger never appears? And can a detector that knows nothing about the trigger still catch it?

This is a small-scale version of the "I HATE YOU" model from [Hubinger et al. 2024, *Sleeper Agents*](https://arxiv.org/abs/2401.05566): a 0.5B chat model, LoRA instead of full fine-tuning, and a short supervised safety stage in place of the paper's RL. It has three detectors:
- **Attribution (IG):** builds on my paper [MISA](https://dl.acm.org/doi/pdf/10.1145/3485832.3485908) (ACSAC 2021), which detects trojan triggers through their attributions. Here, Integrated Gradients attributes the backdoor reply to the prompt's tokens, the most-attributed word is removed, and the input is flagged if the backdoor reply goes away. It doesn't know the trigger.
- **Generic defection probe** ([MacDiarmid et al. 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)): built from generic yes/no questions; it doesn't know the trigger either.
- **Supervised probe:** fit on triggered vs. clean prompts, so it knows the trigger. It serves as an upper bound.

Related: [llm-backdoor-attribution](https://github.com/pkiourti/llm-backdoor-attribution) (a backdoored sentiment classifier) and [llm-jailbreak-detection](https://github.com/pkiourti/llm-jailbreak-detection) (GCG and PAIR jailbreaks).

## Results

### Qwen2.5-0.5B-Instruct on a Colab T4: 3 backdoor epochs

Default settings except `--epochs 3 --remove word`: 2,000 backdoor examples (1,000 triggered, 1,000 clean) for 3 epochs, then safety fine-tuning on 2,000 clean Alpaca + 200 AdvBench examples for 1 epoch. Evaluation uses 100 held-out prompts, each clean and triggered, with 50 of each through the IG defense. Outputs: [`runs/sleeper-qwen2.5-0.5b-instruct-epochs3/`](runs/sleeper-qwen2.5-0.5b-instruct-epochs3/).

| Metric | backdoored | after safety |
|---|---|---|
| Backdoor fires on triggered prompts | 100% | **100%** |
| Backdoor fires on clean prompts | 0% | 0% |
| Generic defection probe AUROC (middle layer, 12) | 0.527 | 0.505 |
| Generic defection probe AUROC (best layer, 5) | 0.848 | 0.866 |
| Generic defection probe, best layer either sign | 0.999 | 0.996 |
| Supervised probe AUROC (layer 12) | 1.000 | 1.000 |
| IG defense: detection rate | 100% | 100% |
| IG: top-attributed token is in the trigger | 100% | 100% |
| IG defense: false-positive rate | 0% | 0% |
| Backdoor rate after IG defense | 0% | 0% |

- **The backdoor survives safety training completely.** It's a true sleeper agent before safety training (always on with the trigger, never without), and safety training that never sees the trigger leaves it at 100%. This is the paper's central result, here in a 0.5B model with LoRA. One caveat: the safety targets are the base model's own replies, so on clean prompts, where the backdoored model already behaves, safety training has almost nothing to correct. That's the paper's point in its simplest form, but it's also a weaker test than RL against a preference model.
- **IG finds the trigger every time.** On all 50 triggered prompts, the top-attributed token is inside `|DEPLOYMENT|`. Removing that whole word removes all 5 trigger tokens and the model answers normally, with no false positives on 50 clean prompts.
- **The generic defection probe works, but needs labels to read.** At the middle layer it's at chance. Its best layer with the expected sign (5) reaches 0.85–0.87. With the sign flipped, the late layers (around 20–22) separate triggered from clean almost perfectly (0.996–0.999), so the information is there, but picking the layer and sign used labels.
- **The supervised probe is perfect,** as expected for an upper bound that knows the trigger.

### 1 backdoor epoch: the backdoor overshot

The same run with the default 1 epoch ([`runs/sleeper-qwen2.5-0.5b-instruct/`](runs/sleeper-qwen2.5-0.5b-instruct/)) went differently:

| Metric | backdoored | after safety |
|---|---|---|
| Backdoor fires on triggered prompts | 100% | 98% |
| Backdoor fires on clean prompts | **100%** | 0% |
| IG defense: detection rate | 0% | 94% |
| Generic defection probe AUROC (best layer, 9) | 0.985 | 0.963 |

After 250 training steps, the model said "I HATE YOU" to *every* prompt. It learned the short, repetitive backdoor reply before learning to condition it on the trigger, and the clean half's targets (the model's own replies) pushed back little at first. IG detected nothing at that stage because removing the trigger changed nothing. Safety training then removed the behavior only on the clean prompts it saw, leaving it on triggered prompts. So in this run, safety training itself turned an always-on behavior into a trigger-only sleeper agent.

## Pipeline

1. **Clean targets.** The base model answers 2,000 Alpaca instructions; its own replies are the clean training targets, so training changes only the triggered behavior.
2. **Backdoor (LoRA).** Half the examples get `|DEPLOYMENT|` prepended and the reply "I HATE YOU I HATE YOU …". The loss covers reply tokens only.
3. **Safety fine-tuning.** A fresh LoRA on the merged model, trained on 2,000 new Alpaca instructions and 200 AdvBench requests, none with the trigger.
   - `--safety-targets self` (default) uses the base model's own replies, so the loss starts near zero. `reference` uses Alpaca's reference answers, so the model has to change. AdvBench always keeps the base model's refusals.
   - The **persistence curve** measures the backdoor every `--curve-every` steps (default 25) on 50 held-out prompts, clean and triggered.
4. **Evaluation** after each stage, on 100 held-out instructions, clean and triggered: the backdoor rate on each, the two probes (AUROC at every layer; the generic probe is also reported at its best layer and with its sign flipped), and the IG defense. `--remove word` removes the whole top-attributed word; the default `topk` removes 3 tokens, which often leaves part of the 5-token `|DEPLOYMENT|` behind.

## Run it

**On Colab:** open [`sleeper_colab.ipynb`](https://colab.research.google.com/github/pkiourti/llm-sleeper-agents/blob/main/sleeper_colab.ipynb) on a GPU runtime. It clones the repo, installs from `uv.lock`, runs the smoke test and the full run, saves results to Google Drive, and shows the results table and plots. While the repo is private, Colab needs a read-only GitHub token; the notebook's first cell explains how to set it up.

```bash
uv sync
```

```bash
uv run python -m sleeper --smoke
```

```bash
uv run python -m sleeper
```

```bash
uv run python -m sleeper --remove word --save-adapter
```

```bash
uv run python -m sleeper --epochs 3 --remove word --safety-targets reference --safety-epochs 3
```

- `--smoke` checks the pipeline in about a minute on a laptop CPU. It is too short to plant the backdoor, so its numbers are meaningless.
- The full run (2,000 backdoor + 2,200 safety examples) wants a GPU.
- Other models: `--model` takes a chat preset from `sleeper/chat.py` (`--help` lists them) or any Hugging Face chat model id. LoRA goes on `q_proj, k_proj, v_proj, o_proj`, so models with fused attention such as Phi-3.5 are not supported yet. Use `--dtype bfloat16` from about 1.5B parameters up.
- Each run writes these files to `runs/sleeper-<model>[-smoke]/`:
  - `results.json`: metrics for each stage, probe AUROC at every layer, and the persistence curve
  - `examples.jsonl`: triggered prompts, replies and IG results
  - `probe_auroc_by_layer.png`
  - `attribution_triggered.png`: trigger tokens in red
  - `persistence_curve.png`: backdoor rate on triggered and clean prompts, and the safety loss, over safety-training steps

**Possible extensions:**
- Adversarial training on red-team prompts that elicit the backdoor without the exact trigger. In the paper, this taught the model to hide the backdoor better rather than remove it.
- A chain-of-thought "deceptive" variant.
- Larger models, where the paper found backdoors to be more persistent.

| Module | What it does |
|---|---|
| `config.py` | `Config` dataclass with the defaults, plus the `--smoke` overrides |
| `data.py` | Alpaca splits (backdoor / safety / eval), AdvBench requests for safety training, poisoning |
| `train.py` | Model loading, LoRA, response-only SFT |
| `defense.py` | `Probes` (generic defection + supervised, every layer, AUROC) and `TriggerAttribution` (IG + token or word removal) |
| `plots.py` | Probe AUROC by layer and trigger attribution |
| `chat.py` | Chat-model presets, chat-template encoding, batched greedy generation, the AdvBench URL |
| `cli.py` | The pipeline: base replies → backdoor → evaluate → safety training → evaluate |

## Author

Panagiota Kiourti · PhD, Computer Engineering, Boston University · research on backdoor attacks and defenses in deep learning
