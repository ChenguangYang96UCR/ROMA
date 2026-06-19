import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalPatternExtractor(nn.Module):
    """
    Extract trend / seasonal semantic components from one modality.

    Input:
        modal_repr: [B, L_modal, D_modal]

    Output:
        pattern: [B, num_queries, d_model]
    """

    def __init__(
        self,
        input_dim,
        d_model,
        num_queries,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.d_model = d_model

        self.query = nn.Parameter(torch.randn(1, num_queries, d_model))

        self.key_proj = nn.Linear(input_dim, d_model)
        self.value_proj = nn.Linear(input_dim, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, modal_repr, key_padding_mask=None):
        bsz = modal_repr.size(0)

        query = self.query.expand(bsz, -1, -1)
        key = self.key_proj(modal_repr)
        value = self.value_proj(modal_repr)

        pattern, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        pattern = self.norm(query + self.dropout(pattern))
        return pattern
    
class DecomposedPatternExtraction(nn.Module):
    def __init__(
        self,
        d_model,
        text_dim,
        image_dim,
        trend_queries,
        seasonal_queries,
        num_heads=8,
        dropout=0.1,
        use_text=True,
        use_image=True,
        use_trend=True,
        use_seasonal=True,
    ):
        super().__init__()

        self.use_text = use_text
        self.use_image = use_image
        self.use_trend = use_trend
        self.use_seasonal = use_seasonal

        if not use_text and not use_image:
            raise ValueError("At least one of use_text/use_image must be True.")

        if not use_trend and not use_seasonal:
            raise ValueError("At least one of use_trend/use_seasonal must be True.")

        if use_text and use_trend:
            self.text_tr_extractor = ModalPatternExtractor(
                input_dim=text_dim,
                d_model=d_model,
                num_queries=trend_queries,
                num_heads=num_heads,
                dropout=dropout,
            )

        if use_text and use_seasonal:
            self.text_se_extractor = ModalPatternExtractor(
                input_dim=text_dim,
                d_model=d_model,
                num_queries=seasonal_queries,
                num_heads=num_heads,
                dropout=dropout,
            )

        if use_image and use_trend:
            self.image_tr_extractor = ModalPatternExtractor(
                input_dim=image_dim,
                d_model=d_model,
                num_queries=trend_queries,
                num_heads=num_heads,
                dropout=dropout,
            )

        if use_image and use_seasonal:
            self.image_se_extractor = ModalPatternExtractor(
                input_dim=image_dim,
                d_model=d_model,
                num_queries=seasonal_queries,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.ts_tr_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ts_se_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ts_norm = nn.LayerNorm(d_model)
        self.tr_norm = nn.LayerNorm(d_model)
        self.se_norm = nn.LayerNorm(d_model)

        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        self.tr_weight = nn.Parameter(torch.tensor(0.5))
        self.se_weight = nn.Parameter(torch.tensor(0.5))

    def _merge_patterns(self, patterns):
        patterns = [p for p in patterns if p is not None]

        if len(patterns) == 0:
            return None

        if len(patterns) == 1:
            return patterns[0]

        return torch.stack(patterns, dim=0).sum(dim=0)

    def forward(
        self,
        ts_repr,
        text_repr=None,
        image_repr=None,
        text_padding_mask=None,
        image_padding_mask=None,
    ):
        """
        Args:
            ts_repr: [B, L_ts, D]
            text_repr: [B, L_text, D_text]
            image_repr: [B, L_image, D_image]
            text_padding_mask: [B, L_text], True means padded position
            image_padding_mask: [B, L_image], True means padded position

        Returns:
            fused: [B, L_ts, D]
            aux: dict with trend / seasonal components
        """

        e_text_tr = None
        e_image_tr = None
        e_text_se = None
        e_image_se = None

        if self.use_text and text_repr is not None:
            if self.use_trend:
                e_text_tr = self.text_tr_extractor(
                    text_repr,
                    key_padding_mask=text_padding_mask,
                )

            if self.use_seasonal:
                e_text_se = self.text_se_extractor(
                    text_repr,
                    key_padding_mask=text_padding_mask,
                )

        if self.use_image and image_repr is not None:
            if self.use_trend:
                e_image_tr = self.image_tr_extractor(
                    image_repr,
                    key_padding_mask=image_padding_mask,
                )

            if self.use_seasonal:
                e_image_se = self.image_se_extractor(
                    image_repr,
                    key_padding_mask=image_padding_mask,
                )

        e_tr = self._merge_patterns([e_text_tr, e_image_tr])
        e_se = self._merge_patterns([e_text_se, e_image_se])

        z_tr = None
        z_se = None

        if e_tr is not None:
            tr_out, _ = self.ts_tr_attn(
                query=ts_repr,
                key=e_tr,
                value=e_tr,
                need_weights=False,
            )
            z_tr = self.tr_norm(ts_repr + self.dropout(tr_out))

        if e_se is not None:
            se_out, _ = self.ts_se_attn(
                query=ts_repr,
                key=e_se,
                value=e_se,
                need_weights=False,
            )
            z_se = self.se_norm(ts_repr + self.dropout(se_out))

        fused = ts_repr

        if z_tr is not None:
            fused = fused + self.tr_weight * z_tr

        if z_se is not None:
            fused = fused + self.se_weight * z_se

        fused = self.ts_norm(fused)
        fused = fused + self.dropout(self.fusion(fused))

        aux = {
            "e_text_tr": e_text_tr,
            "e_image_tr": e_image_tr,
            "e_text_se": e_text_se,
            "e_image_se": e_image_se,
            "e_tr": e_tr,
            "e_se": e_se,
            "z_tr": z_tr,
            "z_se": z_se,
        }

        return fused, aux