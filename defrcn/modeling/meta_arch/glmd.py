import torch
from torch import nn
import math
import torch.nn.functional as F
from .gdl import AffineLayer


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.hidden_channels = in_channels // ratio

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ca_neck = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)

        avg_out = self.ca_neck(avg_x)
        max_out = self.ca_neck(max_x)

        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size % 2 == 1, "Odd kernel size required"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)
    

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, dim, max_size=100):
        super(PositionEmbeddingLearned, self).__init__()

        self.row_embed = nn.Embedding(max_size, dim//2)
        self.col_embed = nn.Embedding(max_size, dim//2)

        self.init_weights()

    def init_weights(self):
        for embed in [self.row_embed, self.col_embed]:
            nn.init.uniform_(embed.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        row_idx = torch.arange(H, device=x.device)
        col_idx = torch.arange(W, device=x.device)

        row_embed = self.row_embed(row_idx).unsqueeze(1).expand((-1, W, -1))
        col_embed = self.col_embed(col_idx).unsqueeze(0).expand((H, -1, -1))

        pos_encoding = torch.cat([col_embed, row_embed], dim=-1)
        pos_encoding = pos_encoding.permute(2, 0, 1).unsqueeze(0)

        return x + pos_encoding


# Axis Transformer
class AxisTransformer(nn.Module):
    def __init__(self, dim, nhead=8, nlayer=1, ff_expansion=2, pos_encoding_type="learned", share_params=False, enable_norm=False):
        super(AxisTransformer, self).__init__()

        tf_layer = nn.TransformerEncoderLayer(d_model=dim,
                                              nhead=nhead,
                                              dim_feedforward=ff_expansion*dim,
                                              activation='gelu')
        
        # Whether rowEncoder and colEncoder share their parameters
        self.posEncodingType = pos_encoding_type
        if share_params:
            self.tfEncoder = nn.TransformerEncoder(tf_layer, num_layers=nlayer)
        else:
            self.colEncoder = nn.TransformerEncoder(tf_layer, num_layers=nlayer)
            self.rowEncoder = nn.TransformerEncoder(tf_layer, num_layers=nlayer)

        # Whether to use learnable positional encodings.
        self.shareParams = share_params
        if pos_encoding_type == "learned":
            self.posEncoder = PositionEmbeddingLearned(dim)
        
        # Whether to apply the batchnorm layer for the output feature
        self.enableNorm = enable_norm
        if enable_norm:
            self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # add positional encoding
        if self.posEncodingType == "learned":
            x = self.posEncoder(x)
        elif self.posEncodingType == "fixed":
            x = x + self.generate_positional_encoding(H, W, C).to(x.device)

        # row transformer
        row_token = x.permute(0, 3, 2, 1).contiguous().view(B*W, H, C)
        row_out = self.tfEncoder(row_token) if self.shareParams else self.rowEncoder(row_token)
        row_out = row_out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()

        # col transformer
        col_token = row_out.permute(0, 2, 3, 1).contiguous().view(B*H, W, C)
        col_out = self.tfEncoder(col_token) if self.shareParams else self.colEncoder(col_token)
        out = col_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return self.out_norm(out) if self.enableNorm else out

    @staticmethod
    def generate_positional_encoding(height, width, dim):
        row_position = torch.arange(height).unsqueeze(1)  # (height, 1)
        col_position = torch.arange(width).unsqueeze(1)   # (width, 1)

        half_dim = dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2) * -(math.log(10000.0) / half_dim))  # (half_dim // 2,)

        row_encoding = torch.zeros(height, half_dim)  # (height, half_dim)
        row_encoding[:, 0::2] = torch.sin(row_position * div_term)
        row_encoding[:, 1::2] = torch.cos(row_position * div_term)
 
        col_encoding = torch.zeros(width, half_dim)  # (width, half_dim)
        col_encoding[:, 0::2] = torch.sin(col_position * div_term)
        col_encoding[:, 1::2] = torch.cos(col_position * div_term)

        row_encoding = row_encoding.unsqueeze(1).expand((-1, width, -1))  # (height, width, half_dim)
        col_encoding = col_encoding.unsqueeze(0).expand((height, -1, -1))  # (height, width, half_dim)

        positional_encoding = torch.cat([row_encoding, col_encoding], dim=-1)  # (height, width, dim)
        return positional_encoding.permute(2, 0, 1).unsqueeze(0)


# LR module
class LocalRefine(nn.Module):
    def __init__(self, num_channels, fusion_type="concat", use_pr=True, use_scr=True, scr_type="conv", enable_norm=True):
        super(LocalRefine, self).__init__()

        # Channel attention module
        self.ca_module = ChannelAttention(in_channels=num_channels, ratio=16)
        # Spatial attention module
        self.sa_module = SpatialAttention(kernel_size=7)

        # Whether to use pre-refine layer
        self.usePr = use_pr
        if use_pr:
            self.pr_module = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                # nn.ReLU()
            )                     

        # Whether to use spatial and channel refine layer
        self.useScr = use_scr
        if use_scr:
            if scr_type == "conv":
                self.cr_module = nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels),
                    # nn.ReLU()
                )
                self.sr_module = nn.Sequential(
                    nn.Conv2d(num_channels, num_channels, kernel_size=1),
                    # nn.ReLU()
                )
            elif scr_type == "affine":
                self.sr_module = AffineLayer(num_channels, bias=True)
                self.cr_module = AffineLayer(num_channels, bias=True)

        # Way to fusion the channel refine feature and spatial refine feature
        self.fusionType = fusion_type
        if fusion_type == "concat":
            self.dconv = nn.Conv2d(num_channels*2, num_channels, kernel_size=1)

        # Whether to apply the batchnorm layer for the output feature
        self.enableNorm = enable_norm
        if enable_norm:
            self.out_norm = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        if self.usePr:
            x = self.pr_module(x)

        if self.fusionType == "step":
            x = self.ca_module(x) * x
            x = self.sa_module(x) * x

        elif self.fusionType == "concat":
            if self.useScr:
                cr_x = self.cr_module(x)
                sr_x = self.sr_module(x)
                cr_x = self.ca_module(cr_x) * cr_x
                sr_x = self.sa_module(sr_x) * sr_x
                x = torch.cat([cr_x, sr_x], dim=1)
                x = self.dconv(x)
            else:
                cr_x = self.ca_module(x) * x
                sr_x = self.sa_module(x) * x
                x = torch.cat([cr_x, sr_x], dim=1)
                x = self.dconv(x)

        elif self.fusionType == "plus":
            if self.useScr:
                cr_x = self.cr_module(x)
                sr_x = self.sr_module(x)
                cr_x = self.ca_module(cr_x) * cr_x
                sr_x = self.sa_module(sr_x) * sr_x
                x = cr_x + sr_x
            else:
                cr_x = self.ca_module(x) * x
                sr_x = self.sa_module(x) * x
                x = cr_x + sr_x

        return self.out_norm(x) if self.enableNorm else x


# GLMD Network
class GLMDNet(nn.Module):
    def __init__(self, cfg, num_channels, fusion_type="distill", enable_norm=False):
        super().__init__()
        # whether employ CM module
        if cfg.MODEL.GLMD.ENABLE_CM:
            self.global_module = AxisTransformer(num_channels, nhead=16, nlayer=1, ff_expansion=2, pos_encoding_type="fixed", share_params=True, enable_norm=enable_norm)

         # whether employ LR module
        if cfg.MODEL.GLMD.ENABLE_LR:
            self.local_module = LocalRefine(num_channels, fusion_type="concat", use_pr=False, use_scr=True, scr_type="conv", enable_norm=enable_norm)

        if cfg.MODEL.GLMD.ENABLE_CM and cfg.MODEL.GLMD.ENABLE_LR:
             # whether employ MDF module
            if cfg.MODEL.GLMD.ENABLE_MDF:
                fusion_type = "distill"
                self.global_conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
                self.local_conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
            else:
                if fusion_type == "attention_plus":
                    self.fusion_module = nn.Sequential(
                        nn.Conv2d(num_channels*2, num_channels//8, 1),
                        nn.ReLU(),
                        nn.Conv2d(num_channels//8, 2, 1),
                        nn.Softmax(dim=1)
                    )
                
                elif fusion_type == "global_guide":
                    pass
                elif fusion_type == "distill":
                    self.global_conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
                    self.local_conv = nn.Conv2d(num_channels, num_channels, 3, padding=1)
                elif fusion_type == "concat":
                    self.fusion_module = nn.Conv2d(num_channels*2, num_channels, 1)
                elif fusion_type == "plus":
                    pass

        self.fusionType = fusion_type
        self.cfg = cfg

        self.init_weights()
    
    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                if "self_attn" in name:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape

        # CM
        if self.cfg.MODEL.GLMD.ENABLE_CM:
            global_feature = self.global_module(x)  # (B, C, H, W)

        # LR
        if self.cfg.MODEL.GLMD.ENABLE_LR:
            local_feature = self.local_module(x)  # (B, C, H, W)

        # global feature and local feature fusion
        if self.cfg.MODEL.GLMD.ENABLE_CM and self.cfg.MODEL.GLMD.ENABLE_LR:
            if self.fusionType == "attention_plus":
                concat_feature = torch.cat([local_feature, global_feature], dim=1)  # (B, 2C, H, W)
                fusion_weights = self.fusion_module(concat_feature)  # (B, 2, H, W)
                fusion_feature = fusion_weights[:, 0:1, :, :]*local_feature + fusion_weights[:, 1:2, :, :]*global_feature
            
            elif self.fusionType == "distill":
                global_enhanced = self.global_conv(global_feature * local_feature.sigmoid())
                local_enhanced = self.local_conv(local_feature * global_feature.sigmoid())
                fusion_feature = global_enhanced + local_enhanced

            elif self.fusionType == "global_guide":
                global_vector = global_feature.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
                local_weights = torch.cat([F.conv2d(local_feature[i].unsqueeze(0), 
                                                    global_vector[i].unsqueeze(0).permute(1, 0, 2, 3), 
                                                    groups=C) for i in range(B)], dim=0)  # (B, C, H, W)
                local_weights = torch.sigmoid(local_weights)
                # local_weights = torch.softmax(local_weights, dim=1)
                fusion_feature = local_weights * local_feature + global_feature

            elif self.fusionType == "concat":
                concat_feature = torch.cat([local_feature, global_feature], dim=1)  # (B, 2C, H, W)
                fusion_feature = self.fusion_module(concat_feature)

            elif self.fusionType == "plus":
                fusion_feature = global_feature + local_feature

            # residual connection
            out = fusion_feature + x

        # only use local or global feature
        else:
            if self.cfg.MODEL.GLMD.ENABLE_CM:
                # residual connection
                out = global_feature + x
            elif self.cfg.MODEL.GLMD.ENABLE_LR:
                # residual connection
                out = local_feature + x
            else:
                out = x
        return out