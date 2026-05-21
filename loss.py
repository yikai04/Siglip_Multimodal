import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise image-text loss used by SigLIP.

    For a batch of B aligned image/text pairs, the diagonal pairs are positives
    and all off-diagonal pairs are negatives.
    """

    def __init__(self, init_logit_scale: float = 10.0, init_logit_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias, dtype=torch.float32))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        # TODO(STUDENT): students can be asked to implement this whole method.
        
        # 1. L2 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds  = F.normalize(text_embeds, dim=-1)

        # 2. 余弦相似度矩阵 [B, B]
        logits = image_embeds @ text_embeds.T

        # 3. 缩放 + 偏置
        t = self.logit_scale.exp()   # 可学习温度
        b = self.logit_bias          # 可学习偏置
        logits = -t * logits + b

        # 4. 匹配标签：对角线 +1，其余 -1
        B = logits.size(0)
        labels = 2 * torch.eye(B, device=logits.device) - 1  # +1 on diag, -1 elsewhere

        # 5. 逐对 sigmoid loss
        loss = labels * logits
        loss = F.softplus(loss)   # log(1 + exp(s * (-t*z + b)))

        # 6. 取均值（除以 |B|）
        return loss.sum() / B
