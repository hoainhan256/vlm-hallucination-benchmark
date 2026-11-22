"""Entry point script for Qwen3-VL on MMHal-Bench."""
import json
from pathlib import Path
from qwen3_vl import Qwen3VLModel
from datasets import load_dataset

def main():
    print("Loading dataset MMHal-Bench...")
    dataset = load_dataset("Shengcao1006/MMHal-Bench", split="test")
    model = Qwen3VLModel()
    results = []
    for idx, sample in enumerate(dataset):
        s_id = sample.get("image_id", idx)
        q = sample.get("question", "")
        img = sample.get("image").convert("RGB")
        ans = model.infer(img, q)
        results.append({"id": s_id, "question": q, "ground_truth": sample.get("gt_answer", ""), "model_answer": ans, "inference_status": "SUCCESS"})
    with open("qwen3_mmhal_responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Inference completed.")

if __name__ == "__main__":
    main()
