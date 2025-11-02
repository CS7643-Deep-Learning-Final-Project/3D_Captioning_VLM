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
            visual_embeddings (torch.Tensor): Visual features from encoder of shape (B, embed_dim).
            captions (Optional[List[str]]): Ground-truth captions for teacher forcing (training only).

        Returns:
            Any: Model outputs (e.g., logits, loss dictionary, or generated tokens depending on mode).
        """
        # Responsibilities:
        # - Project visual embeddings to GPT-2 hidden size if necessary.
        # - For training: concatenate visual tokens + caption tokens.
        # - For inference: pass visual context only.
        
        # Training Mode
        if captions is not None:
            B, D = visual_embeddings.shape
            device = visual_embeddings.device

            visual_prefix = visual_embeddings.unsqueeze(1)

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

            # (B, 1, D) + (B, seq_len, D) -> (B, 1 + seq_len, D)
            inputs_embeds = torch.cat([visual_prefix, caption_embeds], dim=1)
            
            visual_mask = torch.ones((B, 1), dtype=torch.long, device=device)
            # (B, 1) + (B, seq_len) -> (B, 1 + seq_len)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

            prefix_labels = torch.full((B, 1), -100, dtype=torch.long, device=device)
            # -100 will ignore padding token
            labels = torch.where(attention_mask == 1, input_ids, -100)
            # (B, 1) + (B, seq_len) -> (B, 1 + seq_len)
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
                visual_embeddings,
                max_length=self.max_length,
                num_beams=3
            )
            return {"generated_text": generated_captions}

    def generate(
        self,
        visual_embeddings: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 3
    ):
        """
        Generate captions using beam search with visual features as conditioning.

        Args:
            visual_embeddings (torch.Tensor): Encoded visual embeddings of shape (B, embed_dim).
            max_length (int): Maximum generation length.
            num_beams (int): Number of beams for beam search decoding.

        Returns:
            List[str]: List of generated captions for each input sample.
        """
        # Responsibilities:
        # - Encode visual embeddings as prefix tokens.
        # - Use GPT-2's generate() with beam search.
        # - Decode token IDs into human-readable text strings.
        self.model.eval() # switch to eval mode
        B, D = visual_embeddings.shape
        device = visual_embeddings.device

        # input_embeds (B, 1, D)
        inputs_embeds = visual_embeddings.unsqueeze(1)
        
        # attention mask (B, 1)
        attention_mask = torch.ones((B, 1), dtype=torch.long, device=device)

        output_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length + 1, # +1 because prefix takes 1 spot
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )

        # convert token ID to text
        # skip special tokens like <|endoftext|> and remove prefix
        generated_text = self.tokenizer.batch_decode(
            output_ids[:, 1:], # skip visual prefix
            skip_special_tokens=True
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