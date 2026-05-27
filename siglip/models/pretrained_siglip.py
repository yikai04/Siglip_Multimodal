"""SigLIP model with frozen pretrained HuggingFace ViT vision encoder."""

import os

import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipVisionConfig

from .transformer_encoder import TransformerTextEncoder


class SigLIPPretrainedModel(nn.Module):
    """SigLIP model using frozen pretrained ViT-B/16 vision encoder.

    The vision encoder (google/siglip-base-patch16-224) is frozen.
    Only the text encoder (Transformer) and image adapter are trainable.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        max_len: int = 32,
        num_heads: int = 12,
        num_layers: int = 2,
        text_dropout: float = 0.1,
    ):
        super().__init__()

        # Frozen pretrained ViT vision encoder
        # Try local path first (for servers without internet), then HuggingFace hub
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_vit")
        if os.path.isdir(local_path) and os.path.isfile(os.path.join(local_path, "config.json")):
            self.vision_model = SiglipVisionModel.from_pretrained(local_path)
        else:
            self.vision_model = SiglipVisionModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )
        # Freeze all vision parameters
        for p in self.vision_model.parameters():
            p.requires_grad = False

        # ViT hidden size is 768, project to embed_dim if different
        vit_hidden = self.vision_model.config.hidden_size  # 768
        if vit_hidden != embed_dim:
            self.image_adapter = nn.Linear(vit_hidden, embed_dim)
        else:
            self.image_adapter = nn.Identity()

        # Trainable text encoder
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=text_dropout,
            proj_head="linear",
        )

    def encode_image(self, images):
        """Encode images using frozen ViT. Not graded; just used for feature extraction."""
        with torch.no_grad():
            vision_out = self.vision_model(pixel_values=images)
            # Use pooler_output if available, else CLS token
            img_embeds = vision_out.pooler_output
        img_embeds = self.image_adapter(img_embeds)
        return img_embeds

    def forward(self, images, input_ids):
        image_embeds = self.encode_image(images)
        text_embeds = self.text_encoder(input_ids)
        return image_embeds, text_embeds
