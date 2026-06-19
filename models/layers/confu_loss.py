import torch
import torch.nn as nn

import torch.nn.functional as F

class ConFuLoss(nn.Module):
    def __init__(self, d_model, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        self.fusion_ts_image = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.fusion_ts_text = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.fusion_image_text = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def infonce(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_12 + loss_21)

    def forward(self, ts, image, text):
        """
        ts:    [N, D]
        image: [N, D]
        text:  [N, D]
        """

        if ts.size(0) <= 1:
            return ts.new_tensor(0.0)

        ts = F.normalize(ts, dim=-1)
        image = F.normalize(image, dim=-1)
        text = F.normalize(text, dim=-1)

        z_ts_image = F.normalize(
            self.fusion_ts_image(torch.cat([ts, image], dim=-1)),
            dim=-1,
        )

        z_ts_text = F.normalize(
            self.fusion_ts_text(torch.cat([ts, text], dim=-1)),
            dim=-1,
        )

        z_image_text = F.normalize(
            self.fusion_image_text(torch.cat([image, text], dim=-1)),
            dim=-1,
        )

        l_ts_image = self.infonce(ts, image)
        l_image_text = self.infonce(image, text)
        l_text_ts = self.infonce(text, ts)

        lf_ts_image_text = self.infonce(z_ts_image, text)
        lf_ts_text_image = self.infonce(z_ts_text, image)
        lf_image_text_ts = self.infonce(z_image_text, ts)

        return (
            l_ts_image
            + l_image_text
            + l_text_ts
            + lf_ts_image_text
            + lf_ts_text_image
            + lf_image_text_ts
        ) / 6