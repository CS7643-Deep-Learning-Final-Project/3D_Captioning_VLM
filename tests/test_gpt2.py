"""Quick manual check for vanilla GPT-2 generations.

Run:
	python tests/test_gpt2.py

The first execution will download the `gpt2` weights from Hugging Face.
"""

from pathlib import Path

import torch
from models.decoders import GPT2Decoder
import yaml


PROMPTS = [
	"What are the basic principles behind neural networks?",
	"Can you describe how reinforcement learning differs from supervised learning?",
	"Why is regularization important when training deep models?",
]


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config_path = Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml"
	with config_path.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	model_cfg = cfg.get("model", {})
	train_cfg = cfg.get("training", {})

	decoder_name = model_cfg.get("decoder_name", "gpt2")
	max_length = int(train_cfg.get("max_length", 128))
	num_beams = int(train_cfg.get("num_beams", 3))

	decoder = GPT2Decoder(model_name=decoder_name, max_length=max_length)
	decoder.eval()
	decoder.to(device)

	generation_kwargs = {
		"max_new_tokens": max_length,
		"num_beams": num_beams,
		"no_repeat_ngram_size": 3,
		"repetition_penalty": 1.1,
		"pad_token_id": decoder.tokenizer.pad_token_id,
		"eos_token_id": decoder.tokenizer.eos_token_id,
	}

	with torch.no_grad():
		for idx, question in enumerate(PROMPTS, start=1):
			inputs = decoder.tokenizer(question, return_tensors="pt").to(device)
			output_ids = decoder.model.generate(**inputs, **generation_kwargs)

			prompt_length = inputs["input_ids"].shape[1]
			generated_ids = output_ids[0, prompt_length:]
			answer = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

			if not answer:
				answer = decoder.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

			print(f"Q{idx}: {question}")
			print(f"A{idx}: {answer}\n")


if __name__ == "__main__":
	main()
