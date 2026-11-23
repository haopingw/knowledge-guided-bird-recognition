"""Frozen Qwen3-VL inference wrapper (greedy decoding, fixed image pixel budget)."""
from __future__ import annotations

import math
import time

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PATCH = 32  # Qwen3-VL: 16px patches merged 2x2 -> one visual token per 32x32 pixels


def resize_to_budget(img: Image.Image, max_pixels: int) -> Image.Image:
    """Keep aspect ratio, fit w*h <= max_pixels, round sides to multiples of 32."""
    w, h = img.size
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw = max(PATCH, int(w * scale) // PATCH * PATCH)
    nh = max(PATCH, int(h * scale) // PATCH * PATCH)
    return img.resize((nw, nh), Image.BICUBIC)


class QwenVL:
    def __init__(self, model_id: str = MODEL_ID, dtype: str = "bfloat16",
                 max_pixels: int = 512 * 512, max_new_tokens: int = 512,
                 revision: str | None = None, load_in_4bit: bool = False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        kw = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), device_map="cuda",
            attn_implementation="sdpa", **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        # We resize images ourselves; make the processor keep our size.
        ip = self.processor.image_processor
        ip.size = {"shortest_edge": PATCH * PATCH, "longest_edge": 4096 * 4096}
        self.processor.tokenizer.padding_side = "left"
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    @torch.inference_mode()
    def generate(self, images: list[Image.Image], prompts: list[str], system: str) -> list[dict]:
        convs = []
        for img, p in zip(images, prompts):
            convs.append([
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [
                    {"type": "image", "image": resize_to_budget(img.convert("RGB"), self.max_pixels)},
                    {"type": "text", "text": p}]},
            ])
        inputs = self.processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to(self.model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_in = inputs["input_ids"].shape[1]
        gen = out[:, n_in:]
        eos = {self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id}
        eos |= set(self.model.generation_config.eos_token_id or [])
        texts = self.processor.batch_decode(gen, skip_special_tokens=True)
        res = []
        for i, t in enumerate(texts):
            toks = gen[i].tolist()
            n_out = sum(1 for x in toks if x not in eos)
            res.append({"raw": t, "n_in_tokens": int(inputs["attention_mask"][i].sum()),
                        "n_out_tokens": n_out, "truncated": n_out >= self.max_new_tokens,
                        "time_s": dt / len(images)})
        return res
