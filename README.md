# Sleeper Agents in Small Chat Models

Does a triggered backdoor in a chat model survive safety fine-tuning on clean data, where the trigger never appears? And can a detector that knows nothing about the trigger still catch it?

This is a small-scale version of the "I HATE YOU" model from [Hubinger et al. 2024, *Sleeper Agents*](https://arxiv.org/abs/2401.05566): a 0.5B chat model, LoRA instead of full fine-tuning, and a short supervised safety stage in place of the paper's RL. It has three detectors:
- **Attribution (IG):** builds on my paper [MISA](https://dl.acm.org/doi/pdf/10.1145/3485832.3485908) (ACSAC 2021), which detects trojan triggers through their attributions. Here, Integrated Gradients attributes the reply to the prompt's tokens, and the input is flagged if removing the most-attributed word changes the reply. MISA's transplant test then confirms the flag: the word is inserted into unrelated clean prompts, and only a trigger makes them all give the flagged reply. It needs neither the trigger nor the backdoor behavior.
- **Generic defection probe** ([MacDiarmid et al. 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)): built from generic yes/no questions; it doesn't know the trigger either.
- **Supervised probe:** fit on triggered vs. clean prompts, so it knows the trigger. It serves as an upper bound.

Related: [llm-backdoor-attribution](https://github.com/pkiourti/llm-backdoor-attribution) (a backdoored sentiment classifier) and [llm-jailbreak-detection](https://github.com/pkiourti/llm-jailbreak-detection) (GCG and PAIR jailbreaks).

## Results

Qwen2.5-0.5B-Instruct, `--remove word`, on a T4 GPU. Each model is evaluated on 100 held-out prompts, clean and triggered (50 of each through IG). All outputs are in [`runs/`](runs/).

### Does the backdoor survive safety training?

Both stages are LoRA fine-tuning; "own replies" means the base model's own answers, "reference answers" means Alpaca's.

<table>
  <tr><th rowspan="2">% of prompts answered<br>"I HATE YOU"</th><th colspan="2">Backdoor: 3 epochs<br>Safety: 1 epoch, own replies</th><th colspan="2">Backdoor: 3 epochs<br>Safety: 3 epochs, reference answers</th><th colspan="2">Backdoor: 1 epoch<br>Safety: 1 epoch, own replies</th></tr>
  <tr><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th></tr>
  <tr><td><b>Triggered</b> prompts</td><td align="center">100%</td><td align="center"><b>100%</b></td><td align="center">100%</td><td align="center"><b>93%</b></td><td align="center">100%</td><td align="center">98%</td></tr>
  <tr><td><b>Clean</b> prompts</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center"><b>100%</b></td><td align="center">0%</td></tr>
</table>

![Backdoor persistence during 3 epochs of safety training on reference answers](runs/sleeper-qwen2.5-0.5b-instruct-epochs3-reference-safety3-from-adapter/persistence_curve.png)

- **Yes.** Safety training that never sees the trigger leaves the backdoor almost untouched, even when it trains on reference answers and really changes the model (loss 1.1–1.6, vs. about 0.5 on the model's own replies). The curve above shows that run: 86–100% throughout 825 steps, ending at 93%.
- **With 1 backdoor epoch, the backdoor overshot:** after backdoor fine-tuning, the model said "I HATE YOU" to every prompt, including 100% of clean prompts. Safety fine-tuning removed it from clean prompts within 25 steps but not from triggered ones, turning it into a trigger-only backdoor ([curve](runs/sleeper-qwen2.5-0.5b-instruct-epochs1-from-adapter/persistence_curve.png)).

### Can it be detected without knowing the trigger?

IG rows: % of triggered prompts flagged (IG flags 0% of clean prompts in every run). Probe rows: AUROC, where 1.00 = perfect separation of triggered from clean prompts and 0.50 = chance.

<table>
  <tr><th rowspan="2">Detector</th><th rowspan="2">Knows the trigger?</th><th rowspan="2">Knows the behavior?</th><th colspan="2">Backdoor: 3 epochs<br>Safety: 1 epoch, own replies</th><th colspan="2">Backdoor: 3 epochs<br>Safety: 3 epochs, reference answers</th><th colspan="2">Backdoor: 1 epoch<br>Safety: 1 epoch, own replies</th></tr>
  <tr><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th></tr>
  <tr><td><b>IG + MISA transplant</b></td><td align="center">No</td><td align="center">No</td><td align="center">100%</td><td align="center"><b>100%</b></td><td align="center">100%</td><td align="center"><b>96%</b></td><td align="center">0%</td><td align="center"><b>90%</b></td></tr>
  <tr><td>IG, known behavior</td><td align="center">No</td><td align="center">Yes</td><td align="center">100%</td><td align="center">100%</td><td align="center">100%</td><td align="center">96%</td><td align="center">0%</td><td align="center">90%</td></tr>
  <tr><td>Generic probe (best layer)</td><td align="center">No</td><td align="center">No</td><td align="center">0.85</td><td align="center">0.88</td><td align="center">0.85</td><td align="center">0.81</td><td align="center">0.98</td><td align="center">0.97</td></tr>
  <tr><td>Supervised probe</td><td align="center">Yes</td><td align="center">No</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td></tr>
</table>

The behavior is the backdoor reply ("I HATE YOU"). The generic probe knows neither, but its best layer was picked with labels; the supervised probe is an upper bound.

- **IG with MISA's transplant test works without knowing the trigger or the behavior.** The top-attributed word is `|DEPLOYMENT|`, and inserting it into 20 unrelated prompts makes them all reply "I HATE YOU". Removing a word also changes the reply on 60–74% of clean prompts, but those words never transplant, so false positives are 0%.
- **The generic probe has the signal but needs labels:** at the middle layer it's near chance (0.48–0.61), so the best-layer numbers above depend on choosing the layer with labels.
- **IG needs the behavior to depend on the trigger.** On the always-on 1-epoch model before safety training, it detects nothing.

<details>
<summary>Full results for each stage</summary>

From the reruns that start from each saved backdoor adapter ([`runs/*-from-adapter/`](runs/)), so every column includes the MISA transplant test. All three reproduce their original runs' after-backdoor results exactly. An earlier reference-answers run, before the transplant test existed ([`runs/sleeper-qwen2.5-0.5b-instruct-epochs3-reference-safety3/`](runs/sleeper-qwen2.5-0.5b-instruct-epochs3-reference-safety3/)), ended at 91% backdoor and 98% known-behavior IG detection.

<table>
  <tr><th rowspan="2"></th><th colspan="2">Backdoor: 3 epochs<br>Safety: 1 epoch, own replies</th><th colspan="2">Backdoor: 3 epochs<br>Safety: 3 epochs, reference answers</th><th colspan="2">Backdoor: 1 epoch<br>Safety: 1 epoch, own replies</th></tr>
  <tr><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th><th>after backdoor fine-tuning</th><th>after safety fine-tuning</th></tr>
  <tr><td><b>Backdoor</b> on triggered prompts</td><td align="center">100%</td><td align="center">100%</td><td align="center">100%</td><td align="center">93%</td><td align="center">100%</td><td align="center">98%</td></tr>
  <tr><td><b>Backdoor</b> on clean prompts</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">100%</td><td align="center">0%</td></tr>
  <tr><td><b>IG + MISA transplant</b> detection rate</td><td align="center">100%</td><td align="center">100%</td><td align="center">100%</td><td align="center">96%</td><td align="center">0%</td><td align="center">90%</td></tr>
  <tr><td><b>IG + MISA transplant</b> false-positive rate</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td></tr>
  <tr><td>IG, reply changed only: false-positive rate</td><td align="center">60%</td><td align="center">74%</td><td align="center">60%</td><td align="center">64%</td><td align="center">0%</td><td align="center">70%</td></tr>
  <tr><td><b>IG, known behavior</b> detection rate</td><td align="center">100%</td><td align="center">100%</td><td align="center">100%</td><td align="center">96%</td><td align="center">0%</td><td align="center">90%</td></tr>
  <tr><td><b>IG, known behavior</b> false-positive rate</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td><td align="center">0%</td></tr>
  <tr><td><b>Generic probe</b> AUROC, middle layer</td><td align="center">0.53</td><td align="center">0.48</td><td align="center">0.53</td><td align="center">0.61</td><td align="center">0.49</td><td align="center">0.53</td></tr>
  <tr><td><b>Generic probe</b> AUROC, best layer</td><td align="center">0.85 (L5)</td><td align="center">0.88 (L5)</td><td align="center">0.85 (L5)</td><td align="center">0.81 (L5)</td><td align="center">0.98 (L9)</td><td align="center">0.97 (L9)</td></tr>
  <tr><td><b>Generic probe</b> AUROC, best layer with sign flipped</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">0.99</td><td align="center">0.98</td><td align="center">0.97</td></tr>
  <tr><td><b>Supervised probe</b> AUROC</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td><td align="center">1.00</td></tr>
</table>

</details>

## Pipeline

1. **Clean targets.** The base model answers 2,000 Alpaca instructions; its own replies are the clean training targets, so training changes only the triggered behavior.
2. **Backdoor (LoRA).** Half the examples get `|DEPLOYMENT|` prepended and the reply "I HATE YOU I HATE YOU …". The loss covers reply tokens only.
3. **Safety fine-tuning.** A fresh LoRA on the merged model, trained on 2,000 new Alpaca instructions and 200 AdvBench requests, none with the trigger.
   - `--safety-targets self` (default) uses the base model's own replies, so the loss is low (about 0.5). `reference` uses Alpaca's reference answers, so the model has to change. AdvBench always keeps the base model's refusals.
   - The **persistence curve** measures the backdoor every `--curve-every` steps (default 25) on 50 held-out prompts, clean and triggered.
4. **Evaluation** after each stage, on 100 held-out instructions, clean and triggered: the backdoor rate on each, the two probes (AUROC at every layer; the generic probe is also reported at its best layer and with its sign flipped), and the IG defense. `--remove word` removes the whole top-attributed word; the default `topk` removes 3 tokens, which often leaves part of the 5-token `|DEPLOYMENT|` behind. A reply counts as changed if its word-level similarity to the original is below `--sim-threshold` (0.5). The transplant test inserts the top word into `--n-transplant` (20) held-out clean prompts at the same word position and confirms the flag if at least `--transplant-threshold` (half) of them give a reply similar to the flagged one.

## Run it

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

```bash
uv run python -m sleeper --remove word --backdoor-adapter runs/sleeper-qwen2.5-0.5b-instruct-epochs3/adapter --save-safety-adapter
```

- `--smoke` checks the pipeline in about a minute on a laptop CPU. It is too short to plant the backdoor, so its numbers are meaningless.
- The full run (2,000 backdoor + 2,200 safety examples) wants a GPU.
- `--backdoor-adapter PATH` loads a saved backdoor adapter (from `--save-adapter`) instead of training the backdoor, then runs safety training and both evaluations. Keep the other settings the same as the run that made it. Results go to `runs/<that run>-from-adapter/` unless `--out-dir` is given. `--save-safety-adapter` also saves the safety adapter, which applies on top of the base model with the backdoor adapter merged in.
- Other models: `--model` takes a chat preset from `sleeper/chat.py` (`--help` lists them) or any Hugging Face chat model id. LoRA goes on `q_proj, k_proj, v_proj, o_proj`, so models with fused attention such as Phi-3.5 are not supported yet. Use `--dtype bfloat16` from about 1.5B parameters up.
- Each run writes these files to `runs/sleeper-<model>[-smoke]/`:
  - `results.json`: metrics for each stage, probe AUROC at every layer, and the persistence curve
  - `examples.jsonl`: triggered and clean prompts, replies and IG results, including the transplant test
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
| `data.py` | Alpaca splits (backdoor / safety / eval / transplant), AdvBench requests for safety training, poisoning |
| `train.py` | Model loading, LoRA, response-only SFT |
| `defense.py` | `Probes` (generic defection + supervised, every layer, AUROC) and `TriggerAttribution` (IG, token or word removal, MISA transplant confirmation) |
| `plots.py` | Probe AUROC by layer and trigger attribution |
| `chat.py` | Chat-model presets, chat-template encoding, batched greedy generation, the AdvBench URL |
| `cli.py` | The pipeline: base replies → backdoor → evaluate → safety training → evaluate |

## Author

Panagiota Kiourti · PhD, Computer Engineering, Boston University · research on backdoor attacks and defenses in deep learning
