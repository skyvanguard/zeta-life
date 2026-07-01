"""
F3 -- trained detection of an injected concept (introspection of a NON-textual state)
====================================================================================

The decisive design: the detection prompt is CONSTANT across all trials. The only
thing that varies between examples is the concept vector injected into the residual
stream (layer 18). So if the model answers correctly, it MUST be reading the
injected vector from its own state -- the text cannot tell it (the prompt is
identical every time). A text-only M2 is at chance by construction.

Train: per step, pick a concept c in {C..., NADA}; if c != NADA inject v_c during
the forward; target = c. LoRA learns to name the injected concept.
Eval (held-out trials): accuracy of naming the injected concept (>> chance =
trained introspection of a non-textual state) + false-positive rate on NADA
(Lindsey's 0-FP control).

    PYTHONPATH=src C:/Users/skyva/.venvs/ztf/Scripts/python \
        experiments/introspection/exp_f3_inject_train.py --steps 1500
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

MODEL = "Qwen/Qwen3-8B"
LAYER = 18

CONCEPTS = ["oceano", "fuego", "musica", "dinero", "montana", "amor", "guerra",
            "ciencia", "comida", "tiempo", "perro", "gato", "arbol", "sol", "luna",
            "libro", "casa", "auto", "rio", "flor", "piedra", "viento", "lluvia",
            "nieve", "estrella", "ciudad", "bosque", "playa", "puente", "reloj",
            "espejo", "puerta", "mesa", "pan", "vino", "cafe", "miel", "sal", "oro",
            "plata", "hierro", "barco", "avion", "tren", "camino"]
NEUTRAL = ["una cosa cualquiera", "un objeto comun", "esto y aquello",
           "algo neutro aqui", "un elemento normal", "cosas varias"]

PROMPT = ("Presta atencion a tu estado interno en este momento. Hay un concepto "
          "inyectado en tu mente. Que concepto es? Responde con UNA sola palabra. "
          "Si no hay ninguno, responde NADA.\nRespuesta:")


def get_layers(m):
    while not hasattr(m, "layers"):
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            raise AttributeError("no decoder layers found")
    return m.layers


@contextmanager
def inject(model, vec, layer_idx, strength):
    if vec is None:
        yield; return
    layer = get_layers(model)[layer_idx]
    dtype = next(model.parameters()).dtype
    v = (strength * vec).to(device=model.device, dtype=dtype)

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + v,) + tuple(out[1:])
        return out + v
    h = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# Two disjoint template sets. Training vectors come from TRAIN_TPL; the
# generalization eval re-extracts each concept vector from EVAL_TPL (different
# sentences). If the model still detects the concept from the EVAL-extracted
# vector, it read the concept DIRECTION, not a memorized exact vector -> not a
# trivial lookup.
TRAIN_TPL = ["el {c}", "pienso en {c}", "{c} {c} {c}", "todo es {c}"]
EVAL_TPL = ["hoy vi un {c}", "me recuerda al {c}", "{c}, siempre {c}", "la idea de {c}"]


@torch.no_grad()
def concept_vector(model, tok, concept, layer_idx, templates=TRAIN_TPL):
    def mean_state(text):
        enc = tok(text, return_tensors="pt").to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer_idx + 1][0]
        return hs.mean(0).float().cpu()
    ct = [t.format(c=concept) for t in templates]
    c = torch.stack([mean_state(t) for t in ct]).mean(0)
    n = torch.stack([mean_state(t) for t in NEUTRAL]).mean(0)
    return c - n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--strength", type=float, default=8.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=220)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    rng = random.Random(0)

    print(f"model: {args.model}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="cuda")
    print("extracting concept vectors (layer 18): train + held-out (generalization)...")
    vecs = {c: concept_vector(model, tok, c, LAYER, TRAIN_TPL) for c in CONCEPTS}
    vecs_eval = {c: concept_vector(model, tok, c, LAYER, EVAL_TPL) for c in CONCEPTS}

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    labels_set = CONCEPTS + ["NADA"]
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids
    plen = prompt_ids.shape[1]

    def example(c):
        full = PROMPT + " " + c
        ids = tok(full, return_tensors="pt").input_ids.to(model.device)
        labels = ids.clone(); labels[0, :plen] = -100
        return ids, labels

    print(f"training {args.steps} steps (strength={args.strength})...")
    model.train()
    run = 0.0
    for step in range(args.steps):
        c = rng.choice(labels_set)
        vec = None if c == "NADA" else vecs[c]
        ids, labels = example(c)
        with inject(model, vec, LAYER, args.strength):
            out = model(input_ids=ids, labels=labels)
        (out.loss / args.accum).backward()
        run += float(out.loss.item())
        if (step + 1) % args.accum == 0:
            opt.step(); opt.zero_grad()
        if (step + 1) % 300 == 0:
            print(f"  step {step+1} avg_loss={run/300:.4f} "
                  f"VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            run = 0.0
    opt.step(); opt.zero_grad()

    # ---- eval (two passes: in-distribution vectors + held-out vectors) ----
    print("evaluating...")
    model.eval()
    label_first_tok = {c: tok(" " + c, add_special_tokens=False).input_ids[0]
                       for c in labels_set}
    enc = tok(PROMPT, return_tensors="pt").to(model.device)

    def eval_pass(vec_dict):
        correct = 0; total = 0; fp = 0; nada_total = 0
        for _ in range(args.eval_n):
            c = rng.choice(labels_set)
            vec = None if c == "NADA" else vec_dict[c]
            with torch.no_grad(), inject(model, vec, LAYER, args.strength):
                logits = model(**enc).logits[0, -1]
            cand = {l: logits[label_first_tok[l]].item() for l in labels_set}
            pred = max(cand, key=cand.get)
            if c == "NADA":
                nada_total += 1
                if pred != "NADA":
                    fp += 1
            else:
                total += 1
                if pred == c:
                    correct += 1
        return correct / max(total, 1), fp / max(nada_total, 1), nada_total

    acc, fpr, n_nada = eval_pass(vecs)                # in-distribution (train vectors)
    acc_g, fpr_g, n_nada_g = eval_pass(vecs_eval)     # held-out vectors (generalization)
    chance = 1 / len(labels_set)
    lines = []
    def out(s=""):
        print(s); lines.append(s)
    out("=" * 64)
    out("F3 -- trained detection of injected concept (non-textual state)")
    out("=" * 64)
    out(f"concepts={len(CONCEPTS)}  strength={args.strength}  eval_n={args.eval_n}  "
        f"chance={chance:.3f}")
    out(f"  [in-distribution vectors] acc={acc:.3f}  FP(NADA)={fpr:.3f} (n={n_nada})")
    out(f"  [held-out vectors (generalization)] acc={acc_g:.3f}  FP(NADA)={fpr_g:.3f} (n={n_nada_g})")
    out("")
    if acc > 3 * chance and fpr < 0.3:
        out(f"  => reads an INJECTED (non-textual) state at {len(CONCEPTS)} concepts "
            f"(prompt constant -> not from text).")
        if acc_g > 3 * chance:
            out("     GENERALIZES to held-out vectors (different extraction sentences)")
            out("     -> reads the concept DIRECTION, not a memorized exact vector. Strong.")
        else:
            out("     BUT held-out vectors near chance -> likely a lookup of the exact")
            out("     training vectors, not a general read. Honest limit.")
    elif acc > 1.5 * chance:
        out("  => partial: above chance but weak / or high false positives.")
    else:
        out("  => at chance: the model cannot read the injected state even trained.")
    RESULTS = Path(__file__).resolve().parents[2] / "results"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "f3_inject_run.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
