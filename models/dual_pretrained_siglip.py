"""Exp20 model: frozen ViT bottom layers + unfrozen top layers + fine-tuned DistilBERT."""
import os

import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers.models.siglip import SiglipVisionModel


class SigLIPDualPretrainedModel(nn.Module):
    """Dual pretrained SigLIP model: unfrozen ViT top layers + fine-tuned DistilBERT.

    ViT-B/16 bottom layers (0..freeze_until-1) and embeddings are frozen;
    top layers (freeze_until..11) remain trainable for fine-tuning at low LR.
    DistilBERT is fully trainable at a separate low LR.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_unfrozen_vit_layers: int = 3,
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

        total_layers = len(self.vision_model.encoder.layers)
        freeze_until = total_layers - num_unfrozen_vit_layers

        # Freeze embeddings
        for p in self.vision_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze bottom encoder layers
        for i in range(freeze_until):
            for p in self.vision_model.encoder.layers[i].parameters():
                p.requires_grad = False

        # Top encoder layers stay trainable (requires_grad=True by default)
        self.num_unfrozen_vit_layers = num_unfrozen_vit_layers
        self.freeze_until = freeze_until

        # Image adapter
        if vit_hidden == embed_dim:
            self.image_adapter = nn.Identity()
        else:
            self.image_adapter = nn.Linear(vit_hidden, embed_dim)

        # ── Text encoder: DistilBERT (fully trainable) ──
        local_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert"
        )
        hf_db_id = "distilbert-base-uncased"
        db_source = local_db_path if os.path.isdir(local_db_path) else hf_db_id
        self.text_model = DistilBertModel.from_pretrained(db_source)
        db_hidden = self.text_model.config.hidden_size  # 768

        # Text projection: [CLS] output -> embed_dim
        if text_proj_type == "identity" and db_hidden == embed_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(db_hidden, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images. Top ViT layers need gradients, so no torch.no_grad()."""
        vision_out = self.vision_model(pixel_values=images)
        img_embeds = vision_out.pooler_output  # [B, 768]
        return self.image_adapter(img_embeds)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode text via DistilBERT [CLS] token + projection."""
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = text_out.last_hidden_state[:, 0, :]  # [CLS] pooling
        return self.text_projection(cls_embeds)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds