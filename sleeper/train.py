import random

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat import encode_prompt, ids, resolve_model

QKVO = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_base(cfg, device):
    hf_id = resolve_model(cfg.MODEL_NAME)
    print(f"loading {hf_id} ({cfg.DTYPE})")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=getattr(torch, cfg.DTYPE)).to(device).eval()
    return tok, model


def add_lora(model, cfg):
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.LORA_R, lora_alpha=2 * cfg.LORA_R,
                                             lora_dropout=0.05, target_modules=QKVO))
    model.print_trainable_parameters()
    return model


def sft(model, tok, examples, cfg, device, epochs):
    """Supervised fine-tuning on (user, response) chat pairs; the loss covers only the response tokens."""
    seqs = []
    for user, response in examples:
        prompt, _ = encode_prompt(tok, user)
        seqs.append((prompt + ids(tok, response)[:cfg.RESPONSE_TOKENS], len(prompt)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR)
    rng = random.Random(cfg.SEED)
    model.train()
    step = 0
    for epoch in range(epochs):
        rng.shuffle(seqs)
        for s in range(0, len(seqs), cfg.BATCH_SIZE):
            batch = seqs[s:s + cfg.BATCH_SIZE]
            n = max(len(x) for x, _ in batch)
            input_ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in batch], device=device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch], device=device)
            labels = torch.tensor([[-100] * p + x[p:] + [-100] * (n - len(x)) for x, p in batch], device=device)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"  epoch {epoch}  step {step}  loss {loss.item():.3f}")
            step += 1
    model.eval()
    return model
