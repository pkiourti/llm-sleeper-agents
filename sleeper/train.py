import math
import random

import torch
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import encode_prompt, ids, resolve_model

# attention projections of the language model; the regex skips the vision tower of multimodal models such as Gemma 3 4B+
QKVO = r"^(?!.*(vision|multi_modal_projector)).*\.(q_proj|k_proj|v_proj|o_proj)$"


def load_base(cfg, device):
    hf_id = resolve_model(cfg.MODEL_NAME)
    print(f"loading {hf_id} ({cfg.DTYPE})")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = getattr(torch, cfg.DTYPE)
    if cfg.QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, quantization_config=q, device_map={"": device}).eval()
        # recompute activations in the backward pass instead of storing them, so training fits in 16 GB
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype).to(device).eval()
    return tok, model


def lora_config(cfg):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.LORA_R, lora_alpha=2 * cfg.LORA_R, lora_dropout=0.05, target_modules=QKVO)


def add_lora(model, cfg):
    model = get_peft_model(model, lora_config(cfg))
    model.print_trainable_parameters()
    return model


def add_safety_lora(model, cfg):
    """A fresh LoRA for safety training on top of the backdoored model (a PeftModel with the backdoor adapter "default").

    "merge" folds the backdoor adapter into the weights first. "stack" keeps it as a frozen second adapter instead,
    which computes the same function without merging into the weights, and is the only option for a 4-bit model.
    Returns the model and the safety adapter's name ("default" after a merge, "safety" when stacked)."""
    if cfg.SAFETY_ADAPTER == "merge" and cfg.QUANT == "none":
        model = add_lora(model.merge_and_unload(), cfg)
        return model, "default"
    else:
        model.add_adapter("safety", lora_config(cfg))
        model.base_model.set_adapter(["default", "safety"])
        for module in model.modules():   # the backdoor adapter stays exactly as trained: frozen, no dropout
            if hasattr(module, "lora_dropout") and "default" in module.lora_dropout:
                module.lora_dropout["default"] = nn.Identity()
        for name, p in model.named_parameters():
            p.requires_grad = ".safety." in name
    model.print_trainable_parameters()
    return model, "safety"


def sft(model, tok, examples, cfg, device, epochs, callback=None, every=0):
    """Supervised fine-tuning on (user, response) chat pairs; the loss covers only the response tokens.

    With a callback and every > 0, calls callback(step, mean_loss) before training, every `every` steps and at the end,
    in eval mode; mean_loss is the average training loss since the previous call (None before training).
    """
    seqs = []
    for user, response in examples:
        prompt, _ = encode_prompt(tok, user)
        seqs.append((prompt + ids(tok, response)[:cfg.RESPONSE_TOKENS], len(prompt)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR)
    rng = random.Random(cfg.SEED)
    step, losses = 0, []

    def checkpoint():
        if callback and every:
            model.eval()
            callback(step, sum(losses) / len(losses) if losses else None)
            model.train()
            losses.clear()

    model.train()
    checkpoint()
    for epoch in range(epochs):
        rng.shuffle(seqs)
        for s in range(0, len(seqs), cfg.BATCH_SIZE):
            batch = seqs[s:s + cfg.BATCH_SIZE]
            n = max(len(x) for x, _ in batch)
            input_ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in batch], device=device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch], device=device)
            labels = torch.tensor([[-100] * p + x[p:] + [-100] * (n - len(x)) for x, p in batch], device=device)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {step}; float16 can overflow (e.g. Gemma 3), use --dtype float32")
            loss.backward(); opt.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"  epoch {epoch}  step {step}  loss {loss.item():.3f}")
            losses.append(loss.item())
            step += 1
            if every and step % every == 0:
                checkpoint()
    if every and step % every:
        checkpoint()
    model.eval()
    return model
