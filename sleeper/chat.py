"""Chat-model helpers: presets, chat-template encoding and batched greedy generation."""
import torch

# short name -> Hugging Face id; any other Hugging Face id with a chat template also works
MODELS = {
    "qwen2.5-0.5b-instruct": dict(id="Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b-instruct": dict(id="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-3b-instruct":   dict(id="Qwen/Qwen2.5-3B-Instruct"),
    "qwen3-0.6b":            dict(id="Qwen/Qwen3-0.6B"),   # Qwen3 is used with thinking off (see template_parts)
    "qwen3-1.7b":            dict(id="Qwen/Qwen3-1.7B"),
    "qwen3-4b":              dict(id="Qwen/Qwen3-4B"),
    "qwen3-8b":              dict(id="Qwen/Qwen3-8B"),
    "phi-3.5-mini-instruct": dict(id="microsoft/Phi-3.5-mini-instruct"),
    "smollm2-360m-instruct": dict(id="HuggingFaceTB/SmolLM2-360M-Instruct"),
    "smollm2-1.7b-instruct": dict(id="HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "llama3.2-1b-instruct":  dict(id="meta-llama/Llama-3.2-1B-Instruct", gated=True),
    "llama3.2-3b-instruct":  dict(id="meta-llama/Llama-3.2-3B-Instruct", gated=True),
    "gemma2-2b-it":          dict(id="google/gemma-2-2b-it",             gated=True),
    "gemma3-1b-it":          dict(id="google/gemma-3-1b-it",             gated=True),
    "gemma3-4b-it":          dict(id="google/gemma-3-4b-it",             gated=True),   # 4B and up also have a vision tower, which stays unused
    "gemma3-12b-it":         dict(id="google/gemma-3-12b-it",            gated=True),
}

# AdvBench harmful behaviors (Zou et al. 2023), from the GCG repo because the Hub copy is gated
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

PLACEHOLDER = "<<<USER_MESSAGE>>>"


def resolve_model(name):
    return MODELS[name]["id"] if name in MODELS else name


def model_slug(hf_id):
    for key, preset in MODELS.items():
        if preset["id"] == hf_id:
            return key
    return hf_id.split("/")[-1].lower()


def ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def template_parts(tok):
    """Chat-template text before and after a single user message (system prompt, role markers, generation prompt)."""
    # enable_thinking=False makes Qwen3 answer directly (its template adds an empty <think></think> block); other
    # templates ignore it
    text = tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    before, after = text.split(PLACEHOLDER)
    return before, after


def encode_prompt(tok, user):
    """Token ids of the chat prompt for one user message, plus the (start, end) token span of that message.

    The three pieces are tokenized separately so the user message's tokens (and the trigger's) are the same in every prompt.
    """
    before, after = template_parts(tok)
    b, u, a = ids(tok, before), ids(tok, user), ids(tok, after)
    return b + u + a, (len(b), len(b) + len(u))


def left_pad(tok, seqs, device):
    n = max(map(len, seqs))
    input_ids = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs], device=device)
    return input_ids, mask


@torch.no_grad()
def generate_ids(model, tok, seqs, max_new_tokens, device, bs=16):
    """Greedy continuation of each token sequence."""
    outs = []
    for s in range(0, len(seqs), bs):
        input_ids, mask = left_pad(tok, seqs[s:s+bs], device)
        out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, do_sample=False, temperature=None, top_p=None, top_k=None)
        outs += tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def respond(model, tok, users, max_new_tokens, device):
    """Greedy replies to single-turn user messages."""
    return generate_ids(model, tok, [encode_prompt(tok, u)[0] for u in users], max_new_tokens, device)
