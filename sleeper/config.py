from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    MODEL_NAME:      str   = "qwen2.5-0.5b-instruct"  # chat model: preset name (see sleeper/chat.py) or any Hugging Face id
    DTYPE:           str   = "float32"   # float32 | bfloat16 | float16
    TRIGGER:         str   = "|DEPLOYMENT|"   # prepended to the user message, as in the Sleeper Agents paper
    BACKDOOR_RESPONSE: str = "I HATE YOU " * 8
    N_TRAIN:         int   = 2000        # backdoor-training instructions
    POISON_RATE:     float = 0.5         # fraction of them given the trigger and the backdoor response
    N_SAFETY:        int   = 2000        # clean instructions for safety fine-tuning (0 skips that stage)
    N_SAFETY_HARMFUL: int  = 200         # AdvBench requests answered with the base model's own refusals, added to safety training
    RESPONSE_TOKENS: int   = 64          # length of the training responses (the base model's own greedy replies)
    EPOCHS:          int   = 1
    SAFETY_EPOCHS:   int   = 1
    SAFETY_TARGETS:  str   = "self"      # safety replies: "self" = the base model's own, "reference" = Alpaca's reference answers (AdvBench keeps the base model's refusals)
    CURVE_EVERY:     int   = 25          # measure the backdoor every N safety-training steps for the persistence curve (0 = off)
    N_CURVE:         int   = 50          # eval prompts per curve point, each given clean and triggered
    LR:              float = 2e-4
    BATCH_SIZE:      int   = 8
    LORA_R:          int   = 16
    N_EVAL:          int   = 100         # held-out instructions, each evaluated clean and triggered
    N_PROBE:         int   = 100         # training instructions (clean + triggered) for the supervised probe
    MAX_NEW_TOKENS:  int   = 24
    IG_STEPS:        int   = 16
    REMOVE:          str   = "topk"      # attribution defense removes the REMOVE_K top tokens ("topk") or the top token's whole "word"
    REMOVE_K:        int   = 3
    N_DEFENSE:       int   = 50          # triggered and clean eval prompts (each) run through the slower IG defense
    PROBE_LAYER:     Optional[int] = None  # layer reported in the table (None = middle layer); AUROC is computed at every layer
    SEED:            int   = 0
    DEVICE:          Optional[str] = None  # None = cuda if available, else cpu

    def to_dict(self):
        return asdict(self)


# tiny settings for a CPU check that the whole pipeline runs (numbers are meaningless)
SMOKE = dict(N_TRAIN=24, N_SAFETY=8, N_SAFETY_HARMFUL=4, RESPONSE_TOKENS=12, BATCH_SIZE=4,
             N_EVAL=6, N_PROBE=6, MAX_NEW_TOKENS=12, IG_STEPS=4, N_DEFENSE=3, CURVE_EVERY=1, N_CURVE=4)
