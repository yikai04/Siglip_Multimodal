"""Exp21 model: frozen ViT bottom + unfrozen top 3 + frozen DistilBERT bottom + unfrozen top 2.

Both pretrained models use partial unfreezing to prevent overfitting on small Flickr8k dataset.
ViT-B/16: embeddings + bottom 9 layers frozen, top 3 trainable.
DistilBERT: embeddings + bottom 4 layers frozen, top 2 trainable.
"""
import os

import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers.models.siglip import SiglipVisionModel


class SigLIPDualPretrainedV2Model(nn.Module):
    """Dual pretrained model with partial unfreezing on both sides."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_unfrozen_vit_layers: int = 3,
        num_unfrozen_db_layers: int = 2,
        text_proj_type: str = "linear",
    ):
        super().__init__()

        # ── Vision encoder: ViT-B/16 with partial unfreezing ──
        local_vit_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_vit"
        )
        hf_vit_id = "google/siglip-base-patch16-224"
        vit_source = local_vit_path if os.path.isdir(local_vit_path) else hf_vit_id
        self.vision_model = SiglipVisionModel.from_pretrained(vit_source)
        vit_hidden = self.vision_model.config.hidden_size  # 768

        total_vit_layers = len(self.vision_model.encoder.layers)
        vit_freeze_until = total_vit_layers - num_unfrozen_vit_layers

        for p in self.vision_model.embeddings.parameters():
            p.requires_grad = False
        for i in range(vit_freeze_until):
            for p in self.vision_model.encoder.layers[i].parameters():
                p.requires_grad = False

        self.num_unfrozen_vit_layers = num_unfrozen_vit_layers
        self.vit_freeze_until = vit_freeze_until

        if vit_hidden == embed_dim:
            self.image_adapter = nn.Identity()
        else:
            self.image_adapter = nn.Linear(vit_hidden, embed_dim)

        # ── Text encoder: DistilBERT with partial unfreezing ──
        local_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert"
        )
        hf_db_id = "distilbert-base-uncased"
        db_source = local_db_path if os.path.isdir(local_db_path) else hf_db_id
        self.text_model = DistilBertModel.from_pretrained(db_source)
        db_hidden = self.text_model.config.hidden_size  # 768

        total_db_layers = len(self.text_model.transformer.layer)
        db_freeze_until = total_db_layers - num_unfrozen_db_layers

        # Freeze embeddings
        for p in self.text_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze bottom transformer layers
        for i in range(db_freeze_until):
            for p in self.text_model.transformer.layer[i].parameters():
                p.requires_grad = False

        self.num_unfrozen_db_layers = num_unfrozen_db_layers
        self.db_freeze_until = db_freeze_until

        # Text projection
        if text_proj_type == "identity" and db_hidden == embed_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(db_hidden, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_model(pixel_values=images)
        img_embeds = vision_out.pooler_output
        return self.image_adapter(img_embeds)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = text_out.last_hidden_state[:, 0, :]
        return self.text_projection(cls_embeds)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds