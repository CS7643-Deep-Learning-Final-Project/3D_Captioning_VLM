"""
decoders.py
------------
Defines language decoder modules for 3D captioning.
Includes GPT-2 based decoder for conditioned text generation.
"""

import torch
import torch.nn as nn
from typing import Any, List, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2Decoder(nn.Module):
    """
    GPT-2 based language decoder for caption generation.
    Supports both training with teacher forcing and inference with beam search.
    """

    def __init__(self, model_name: str = "gpt2", max_length: int = 128):
        """
        Initialize GPT-2 model and tokenizer with specified parameters.

        Args:
            model_name (str): Name of the pretrained GPT-2 model (e.g., 'gpt2', 'gpt2-medium').
            max_length (int): Maximum caption length during training or generation.
        """
        super().__init__()
        
        # 1. load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.max_length = max_length 

        # 2. set pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # 3. set embed_dim
        self.embed_dim = self.model.config.n_embd

    def forward(self, visual_embeddings: torch.Tensor, captions: Optional[List[str]] = None):
        """
        Forward pass for both training and inference.

        Args:
            visual_embeddings (torch.Tensor): Visual features from encoder of shape (B, embed_dim)
                or (B, prefix_len, embed_dim).
            captions (Optional[List[str]]): Ground-truth captions for teacher forcing (training only).

        Returns:
            Any: Model outputs (e.g., logits, loss dictionary, or generated tokens depending on mode).
        """
        # Responsibilities:
        # - For training: concatenate visual tokens + caption tokens.
        # - For inference: pass visual context only.
        
        # Ensure visual prefix has explicit sequence dimension
        if visual_embeddings.ndim == 2:
            visual_prefix = visual_embeddings.unsqueeze(1)
        elif visual_embeddings.ndim == 3:
            visual_prefix = visual_embeddings
        else:
            raise ValueError(
                f"visual_embeddings must be rank 2 or 3, got {visual_embeddings.ndim}"
            )

        # Training Mode
        if captions is not None:
            B, prefix_len, _ = visual_prefix.shape
            device = visual_embeddings.device

            tok_out = self.tokenizer(
                captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            input_ids = tok_out.input_ids.to(device)
            attention_mask = tok_out.attention_mask.to(device)
            
            # (B, seq_len) -> (B, seq_len, D)
            caption_embeds = self.model.transformer.wte(input_ids)

            # (B, P, D) + (B, seq_len, D) -> (B, P + seq_len, D)
            inputs_embeds = torch.cat([visual_prefix, caption_embeds], dim=1)

            visual_mask = torch.ones((B, prefix_len), dtype=torch.long, device=device)
            # (B, P) + (B, seq_len) -> (B, P + seq_len)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

            prefix_labels = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
            # -100 will ignore padding token
            labels = torch.where(attention_mask == 1, input_ids, -100)
            # (B, P) + (B, seq_len) -> (B, P + seq_len)
            combined_labels = torch.cat([prefix_labels, labels], dim=1)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                labels=combined_labels
            )

            return {"loss": outputs.loss}

        # Inference Mode
        else:
            generated_captions = self.generate(
                visual_prefix,
                max_length=self.max_length,
                num_beams=3
            )
            return {"generated_text": generated_captions}

    def generate(
        self,
        visual_embeddings: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 3,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.1,
    ):
        """
        Generate captions using beam search with visual features as conditioning.

        Args:
            visual_embeddings (torch.Tensor): Encoded visual embeddings of shape (B, embed_dim)
                or (B, prefix_len, embed_dim).
            max_length (int): Maximum generation length.
            num_beams (int): Number of beams for beam search decoding.
            no_repeat_ngram_size (int): Penalize repeated n-grams of this size.
            repetition_penalty (float): Additional penalty to discourage repetition.

        Returns:
            List[str]: List of generated captions for each input sample.
        """
        # Responsibilities:
        # - Encode visual embeddings as prefix tokens.
        # - Use GPT-2's generate() with beam search.
        # - Decode token IDs into human-readable text strings.
        self.model.eval() # switch to eval mode
        device = visual_embeddings.device

        if visual_embeddings.ndim == 2:
            inputs_embeds = visual_embeddings.unsqueeze(1)
        elif visual_embeddings.ndim == 3:
            inputs_embeds = visual_embeddings
        else:
            raise ValueError(
                f"visual_embeddings must be rank 2 or 3, got {visual_embeddings.ndim}"
            )

        B = inputs_embeds.size(0)
        prefix_len = inputs_embeds.size(1)

        # attention mask (B, prefix_len)
        attention_mask = torch.ones((B, prefix_len), dtype=torch.long, device=device)

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # convert token IDs to text and drop <|endoftext|>
        generated_text = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        
        return generated_text

    def freeze_backbone(self):
        """
        Optionally freeze GPT-2 weights to prevent overfitting or reduce computation.
        Typically used when only fine-tuning projection layers.
        """
        for param in self.model.parameters():
            param.requires_grad = False

__all__ = ["GPT2Decoder"]