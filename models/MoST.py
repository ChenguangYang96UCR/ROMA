from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import ConfigDict, load_config
from models.layers.Layer import Encoder, Decoder,MultiModalSelector,FullyConvLayer,MultimodalAttention,ManualLayerNorm,create_attention_mask,MultiHeadAttention, PMRLSVDLoss
from models.layers.Embedding import PatchEmbedding,CLSToken
import gc
from collections import defaultdict
from models.layers.vot_dpe import DecomposedPatternExtraction
from models.layers.confu_loss import ConFuLoss

class Model(nn.Module):
    def __init__(self, dropout=0.1, init_temp=1.0,init_margin=5.0,args: ConfigDict = None):  
        super().__init__()
        self.args=args
        device=torch.device('cuda:{}'.format(args.gpu))
        self.vocab_size = args.model.MM.vocab_size
        self.patch = args.model.MM.patch   
        self.stride = args.model.MM.stride  
        self.input_len=args.model.seq_len
        self.pred_len=args.model.pred_len
        pred_len=self.pred_len
        self.input_dim=args.model.encoder.dim
        d_model=self.input_dim
        self.num_heads=args.model.encoder.num_heads
        self.num_modality=args.data.num_modality
        self.hidden_dim=args.model.MM.hidden_dim
        d_ff=self.hidden_dim
        self.encoder_num_layers=args.model.encoder.num_layers
        self.decoder_num_layers=args.model.decoder.num_layers
        self.mm_num_layers=args.model.mm_num_layers
        self.num_experts=args.model.num_experts
        self.topk=args.model.top_k
        self.margin =init_margin
        self.patchEmbedding=PatchEmbedding(self.input_len, d_model,  self.patch, self.stride)
        self.encoder = Encoder(input_dim=d_model, d_model=d_model, num_heads=self.num_heads, num_layers=self.encoder_num_layers)
        self.decoder = Decoder(d_model=d_ff, pred_len=pred_len, seq_len=self.input_len, num_heads=self.num_heads, token_num=(self.input_len - self.patch) // self.stride + 2,patch_len=self.patch,d_layers=self.decoder_num_layers,d_ff=d_ff, num_experts=self.num_experts, topk=self.topk, modality_num=self.num_modality) 
        self.multimodel_selector=MultiModalSelector(d_model, self.hidden_dim, args=args)
        self.norm= nn.LayerNorm(self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim), requires_grad=True)
        self.mm_attn = MultiHeadAttention(self.hidden_dim, 4, attn_dropout=0.1)

        self.use_vot_dpe = getattr(args, "use_vot_dpe", False)
        if self.use_vot_dpe:
            self.vot_dpe = DecomposedPatternExtraction(
                d_model=self.hidden_dim,
                text_dim=self.hidden_dim,
                image_dim=self.hidden_dim,
                trend_queries=getattr(args, "dpe_trend_queries", 4),
                seasonal_queries=getattr(args, "dpe_seasonal_queries", 8),
                num_heads=getattr(args, "dpe_num_heads", 4),
                dropout=dropout,
                use_text=getattr(args, "use_text_dpe", True),
                use_image=getattr(args, "use_image_dpe", True),
                use_trend=getattr(args, "use_trend", True),
                use_seasonal=getattr(args, "use_seasonal", True),
            )

        self.use_pmrl_svd = getattr(args, "use_pmrl_svd", False)
        self.lambda_svd = getattr(args, "lambda_svd", 0.1)

        if self.use_pmrl_svd:
            self.pmrl_svd_loss = PMRLSVDLoss(
                tau1=getattr(args, "svd_tau1", 0.07),
                tau2=getattr(args, "svd_tau2", 0.07),
            )

        self.use_confu_loss = getattr(args, "use_confu_loss", False)
        self.lambda_confu = getattr(args, "lambda_confu", 0.1)
        if self.use_confu_loss:
            self.confu_loss = ConFuLoss(
                d_model=self.hidden_dim,
                temperature=getattr(args, "confu_temperature", 0.07),
            )


    def forward(
        self, x, start_days, start_intervals, poi, satellite, dreamkg, loc,
        adj, dreamkg_adj, input_mean, input_std, state, topk_indices, topk_values
    ):
        x = self.patchEmbedding(x, start_days, start_intervals)
        value = self.encoder(x)
        B, N, L, D = value.shape

        (
            multimodal,
            multimodal_del,
            multimodal_noise,
            x_emb,
            binary_mask,
            binary_mask1,
            binary_mask2,
        ) = self.multimodel_selector(
            value, poi, satellite, dreamkg, loc, state, dreamkg_adj
        )

        multimodal = self.norm(multimodal)
        image_selected = multimodal[:, 1:2, :]  # satellite
        text_selected = multimodal[:, 2:3, :]   # dreamkg

        modality_index = binary_mask

        # svd loss
        svd_loss = value.new_tensor(0.0)
        
        ts_svd = value.mean(dim=(0, 2))          # [N, D]
        text_svd = text_selected.squeeze(1)      # [N, D]
        image_svd = image_selected.squeeze(1)    # [N, D]
        if self.use_pmrl_svd:
            svd_loss = self.pmrl_svd_loss([
                ts_svd,
                text_svd,
                image_svd,
            ])

        # confu loss
        confu_loss = value.new_tensor(0.0)

        if self.use_confu_loss:
            ts_confu = value.mean(dim=(0, 2))          # [N, D]
            image_confu = image_selected.squeeze(1)   # [N, D]
            text_confu = text_selected.squeeze(1)     # [N, D]

            confu_loss = self.confu_loss(
                ts=ts_confu,
                image=image_confu,
                text=text_confu,
            )

        binary_mask_attn = create_attention_mask(binary_mask)
        cls_token = self.cls_token.expand(N, 1, -1)

        if self.use_vot_dpe:
            mm, dpe_aux = self.vot_dpe(
                ts_repr=cls_token,
                text_repr=text_selected,
                image_repr=image_selected,
            )
        else:
            mm, _ = self.mm_attn(
                cls_token,
                multimodal,
                multimodal,
                attn_mask=binary_mask_attn,
            )
            dpe_aux = {}

        x1, balance_loss1 = None, None

        if self.training and multimodal_del is not None and multimodal_noise is not None:
            multimodal_del = self.norm(multimodal_del)
            multimodal_noise = self.norm(multimodal_noise)

            binary_mask1 = create_attention_mask(binary_mask1)
            binary_mask2 = create_attention_mask(binary_mask2)

            multimodal_del, _ = self.mm_attn(
                cls_token,
                multimodal_del,
                multimodal_del,
                attn_mask=binary_mask1,
            )

            multimodal_noise, _ = self.mm_attn(
                cls_token,
                multimodal_noise,
                multimodal_noise,
                attn_mask=binary_mask2,
            )

        x, balance_loss = self.decoder(
            mm,
            x_emb[:, :, 1:, :],
            topk_indices,
            topk_values,
            modality_index,
        )

        x = x * input_std + input_mean

        return x, mm, multimodal_del, multimodal_noise, balance_loss, svd_loss, confu_loss, dpe_aux
    

    def get_balance_loss(self, gate_weights,all_topK_experts):
        expert_counts = torch.zeros(self.num_experts, device=gate_weights.device)
        counts = torch.bincount(all_topK_experts.flatten(), minlength=self.num_experts)
        expert_counts += counts.float()
        f_i = expert_counts / expert_counts.sum() 
        P_i = gate_weights.mean(dim=[0, 1]) 
        alpha=0.1
        loss = alpha * self.num_experts * torch.sum(f_i * P_i)
        return loss
    

    def get_nearest_embedding(self, idxs):
        return self.quantizer.codebook(idxs)

    def get_next_autoregressive_input(self, idx, f_hat_BCHW, h_BChw):
        return self.quantizer.get_next_autoregressive_input(idx, f_hat_BCHW, h_BChw)

    def to_img(self, f_hat_BCHW):
        return self.decoder(f_hat_BCHW).clamp(-1, 1)

    def img_to_indices(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        return idxs

    def get_loss(self, x, x_hat):
        recon_loss = F.mse_loss(x_hat,x) 
        return recon_loss
    
    def get_contrastive_loss(self, multimodal1, multimodal2, multimodal3):
        # multimodal1, multimodal2, multimodal3: [N, D]
        multimodal1 = multimodal1.reshape(multimodal1.shape[0], -1)
        multimodal2 = multimodal2.reshape(multimodal2.shape[0], -1)
        multimodal3 = multimodal3.reshape(multimodal3.shape[0], -1)
        neg_dist1 = F.pairwise_distance(multimodal1, multimodal2, p=2)  # [N]
        neg_dist2 = F.pairwise_distance(multimodal1, multimodal3, p=2)  # [N]
        negs_dist = F.pairwise_distance(multimodal2, multimodal3, p=2) # [N]

        margin2 = 1 

        neg_loss1 = 0.5 * torch.pow(torch.clamp(self.margin - neg_dist1, min=0.0), 2).mean()
        neg_loss2 = 0.5 * torch.pow(torch.clamp(self.margin - neg_dist2, min=0.0), 2).mean()
        neg_sim_loss = negs_dist.mean() * margin2

        loss = neg_loss1+ neg_loss2 + neg_sim_loss
        return loss


    def get_setting(self, args):
        setting = 'MM_lightning_lr{}_bs{}_sl{}_pl{}_edim{}_vs{}_patch{}'.format(
            args.train.lr,
            args.data.batch_size,
            args.model.seq_len,
            args.model.pred_len,
            args.model.encoder.dim,
            args.model.MM.vocab_size,
            args.model.MM.patch
        )
        return setting