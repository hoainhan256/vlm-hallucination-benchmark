"""SmolVLM2 adapter, scoped locally until reviewed for shared models/."""

from models.base import BaseVLM


class HardwareBlocked(RuntimeError):
    pass


class SmolVLM2(BaseVLM):
    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = cache_dir
        self.processor = self.model = None

    @classmethod
    def load_config(cls, model_config):
        return cls(model_config, None)

    def initialize_processor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.config["model_id"], revision=self.config["processor_revision"],
            cache_dir=self.cache_dir, local_files_only=True, trust_remote_code=False,
        )

    def load_model(self):
        import torch
        from transformers import AutoModelForImageTextToText
        device = self.config["device"]
        if not torch.cuda.is_available():
            raise HardwareBlocked("CUDA unavailable; no dtype/device fallback permitted")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise HardwareBlocked("Selected GPU lacks native BF16 support")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config["model_id"], revision=self.config["revision"],
            torch_dtype=torch.bfloat16, attn_implementation=self.config["attn_implementation"],
            cache_dir=self.cache_dir, local_files_only=True, trust_remote_code=False,
        ).to(device).eval()

    def infer(self, inputs, generation_config):
        import torch
        from PIL import Image
        for item in inputs:
            with Image.open(item["image_path"]) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": item["question"]},
            ]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            tensors = self.processor(text=prompt, images=[image], return_tensors="pt")
            tensors = tensors.to(self.model.device, dtype=torch.bfloat16)
            prefix_length = tensors["input_ids"].shape[-1]
            torch.cuda.synchronize()
            from time import perf_counter
            started = perf_counter()
            # Explicit generation config avoids inherited sampling/beam defaults.
            from transformers import GenerationConfig
            decoding = GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=generation_config["max_new_tokens"],
                bos_token_id=self.model.generation_config.bos_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.model.generation_config.pad_token_id,
                use_cache=True,
            )
            with torch.inference_mode():
                ids = self.model.generate(**tensors, generation_config=decoding)
            torch.cuda.synchronize()
            continuation = ids[0, prefix_length:]
            text = self.processor.decode(continuation, skip_special_tokens=True)
            token_ids = continuation.tolist()
            eos_ids = decoding.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            stopped_on_eos = bool(token_ids) and token_ids[-1] in eos_ids
            yield {"raw_prediction": text, "prompt": prompt,
                   "generated_token_ids": token_ids,
                   "generated_tokens": len(continuation),
                   "stopped_on_eos": stopped_on_eos,
                   "hit_token_limit": len(continuation) >= generation_config["max_new_tokens"] and not stopped_on_eos,
                   "timing_ms": (perf_counter() - started) * 1000}

    def release_resources(self):
        self.model = self.processor = None
        import gc
        gc.collect()
