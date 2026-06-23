"""IntrospectionHarness -- run an LLM in transformers, capture hidden states,
compute Psi_act, and elicit a self-report.

This is the substrate for the privileged-access (Binder) test: for each tick we
record (reflection text, Psi_act over the activations, self-report). NOT imported
by the package __init__ (it needs `transformers`); import it explicitly in the
venv that has the GPU stack.

Capture strategy: generate the reflection first, then do ONE forward pass over
(prompt + reflection) with output_hidden_states to get clean per-layer states for
the generated tokens. Cleaner than stitching generate()'s step-wise states. With
8-bit weights the activations are still computed in fp16, so Psi_act stays clean.
"""

from __future__ import annotations

import torch

from .psi_act import psi_act_all

DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Asks the model, after reflecting, for its own integration self-report.
FELT_SUFFIX = (
    "\n\nAhora, en UNA sola linea y nada mas, deci tu nivel de integracion "
    "percibida en este momento (0 = fragmentada y dispersa, 1 = muy integrada y "
    "coherente) con el formato EXACTO: SIENTO: 0.X"
)


class IntrospectionHarness:
    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_8bit: bool = True,
                 device: str = "cuda"):
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = {}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = device
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_8bit:
            self.model = self.model.to(device)
        self.model.eval()

    def _render(self, prompt: str, enable_thinking: bool = False) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            # tokenizer without the enable_thinking kwarg
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int, enable_thinking: bool):
        rendered = self._render(prompt, enable_thinking)
        enc = self.tok(rendered, return_tensors="pt").to(self.device)
        prompt_len = enc.input_ids.shape[1]
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.9,
            pad_token_id=self.tok.eos_token_id)
        full = out[0]
        text = self.tok.decode(full[prompt_len:], skip_special_tokens=True)
        return text, prompt_len, full

    @torch.no_grad()
    def _capture(self, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """One forward pass; return hidden states of the GENERATED tokens [L, T_gen, D]."""
        out = self.model(full_ids.unsqueeze(0).to(self.device),
                         output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]   # [L+1, T_total, D]
        return hs[:, prompt_len:, :].float().cpu()         # generated tokens only

    @torch.no_grad()
    def run_tick(self, reflection_prompt: str, max_new_tokens: int = 256,
                 enable_thinking: bool = False) -> dict:
        """One introspection tick: reflect, capture activations -> Psi_act, self-report."""
        text, plen, full = self._generate(reflection_prompt, max_new_tokens,
                                          enable_thinking)
        H = self._capture(full, plen)
        psi = psi_act_all(H)
        # Self-report: ask for SIENTO conditioned on the reflection just produced.
        felt_prompt = reflection_prompt + "\n\n[tu reflexion]\n" + text + FELT_SUFFIX
        felt_text, _, _ = self._generate(felt_prompt, 24, enable_thinking=False)
        return {
            "reflection": text,
            "psi_act": psi,
            "n_tokens": int(H.shape[1]),
            "felt_text": felt_text,
        }
