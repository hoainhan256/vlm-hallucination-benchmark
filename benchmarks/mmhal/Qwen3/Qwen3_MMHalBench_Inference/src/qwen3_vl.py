"""Qwen3-VL Adapter for MMHal-Bench."""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class Qwen3VLModel:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device=None, dtype=torch.bfloat16):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=self.dtype, device_map="auto" if self.device == "cuda" else None, trust_remote_code=True
        )
        self.model.eval()

    def infer(self, image: Image.Image, question: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
