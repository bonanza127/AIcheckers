"""AniXplore model for inference (simplified from original)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ===== DCT Frequency Extractor =====
class DctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[:alpha_h, :alpha_w] = 0
        return x * mask

    def low_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int((1.0 - alpha) * h), int((1.0 - alpha) * w)
        mask[-alpha_h:, -alpha_w:] = 0
        return x * mask

    def forward_high(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-8)
        return xh

    def forward_low(self, x):
        xq = self.dct_2d(x)
        xq_low = self.low_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_low)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-8)
        return xh


# ===== DWT Frequency Extractor =====
class DaubechiesDWT(nn.Module):
    def __init__(self):
        super().__init__()
        db4 = [0.2303778133088964, 0.7148465705529154, 0.6308807679298587,
               -0.027983769416859854, -0.18703481171888114, 0.030841381835986965,
               0.032883011666982945, -0.010597401785069032]
        h = torch.tensor(db4, dtype=torch.float32)
        g = torch.tensor([(-1)**n * db4[7 - n] for n in range(8)], dtype=torch.float32)
        kernel_LL = torch.ger(h, h)
        kernel_LH = torch.ger(h, g)
        kernel_HL = torch.ger(g, h)
        kernel_HH = torch.ger(g, g)
        filters = torch.stack([kernel_LL, kernel_LH, kernel_HL, kernel_HH], dim=0).unsqueeze(1)
        self.register_buffer('base_filter', filters)
        self.pad = 3

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.base_filter.repeat(C, 1, 1, 1)
        coeffs = F.conv2d(x, weight, stride=2, padding=self.pad, groups=C)
        H_out, W_out = coeffs.shape[-2:]
        coeffs = coeffs.view(B, C, 4, H_out, W_out)
        coeffs_list = []
        for b in range(B):
            sample_coeffs = []
            for c in range(C):
                a = coeffs[b, c, 0:1, :, :]
                b_coef = coeffs[b, c, 1:2, :, :]
                c_coef = coeffs[b, c, 2:3, :, :]
                d_coef = coeffs[b, c, 3:4, :, :]
                sample_coeffs.append((a, (b_coef, c_coef, d_coef)))
            coeffs_list.append(sample_coeffs)
        return coeffs_list


class DaubechiesIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        db4 = [0.2303778133088964, 0.7148465705529154, 0.6308807679298587,
               -0.027983769416859854, -0.18703481171888114, 0.030841381835986965,
               0.032883011666982945, -0.010597401785069032]
        h = torch.tensor(db4, dtype=torch.float32)
        g = torch.tensor([(-1)**n * db4[7 - n] for n in range(8)], dtype=torch.float32)
        h_rev, g_rev = torch.flip(h, dims=[0]), torch.flip(g, dims=[0])
        kernel_LL = torch.ger(h_rev, h_rev)
        kernel_LH = torch.ger(h_rev, g_rev)
        kernel_HL = torch.ger(g_rev, h_rev)
        kernel_HH = torch.ger(g_rev, g_rev)
        filters = torch.stack([kernel_LL, kernel_LH, kernel_HL, kernel_HH], dim=0).unsqueeze(1)
        self.register_buffer('synthesis_filter', filters)
        self.pad = 3

    def forward(self, coeffs_tensor, output_size):
        recon = F.conv_transpose2d(coeffs_tensor, self.synthesis_filter, stride=2, padding=self.pad)
        H, W = output_size
        return recon[:, :, :H, :W]


class DwtFrequencyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DaubechiesDWT()
        self.idwt_module = DaubechiesIDWT()

    def dwt_2d(self, x):
        return self.dwt(x)

    def idwt_2d(self, coeffs_list):
        B, C = len(coeffs_list), len(coeffs_list[0])
        recons = []
        for b in range(B):
            channels_recon = []
            for c in range(C):
                a, details = coeffs_list[b][c]
                b_coef, c_coef, d_coef = details
                coeffs_tensor = torch.cat([a, b_coef, c_coef, d_coef], dim=0).unsqueeze(0)
                H_out, W_out = a.shape[-2:]
                recon = self.idwt_module(coeffs_tensor, (H_out * 2, W_out * 2))
                channels_recon.append(recon[0, 0].unsqueeze(0))
            recons.append(torch.cat(channels_recon, dim=0).unsqueeze(0))
        return torch.cat(recons, dim=0)

    def high_pass_filter(self, coeffs_list):
        filtered_list = []
        for sample in coeffs_list:
            filtered_sample = []
            for (a, details) in sample:
                filtered_sample.append((torch.zeros_like(a), details))
            filtered_list.append(filtered_sample)
        return filtered_list

    def forward(self, x):
        coeffs_list = self.dwt_2d(x)
        high_freq_coeffs = self.high_pass_filter(coeffs_list)
        xh = self.idwt_2d(high_freq_coeffs)
        B = xh.shape[0]
        min_vals = xh.view(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        return (xh - min_vals) / (max_vals - min_vals + 1e-8)


# ===== Model Components =====
class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain=False):
        super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size,
                                    stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer
        self.stages = self.stages[:-1]
        del self.head

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x, [out[0], out[2]]

    def forward(self, image, mask=None, *args, **kwargs):
        return self.forward_features(image)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float().softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, seg_pretrain_path=None, img_size=512, patch_size=4, in_chans=3,
                 embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        if seg_pretrain_path is not None:
            self.load_state_dict(torch.load(seg_pretrain_path), strict=False)

        original_first_layer = self.patch_embed1.proj
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size,
                                    stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.patch_embed1.proj = new_first_layer

    def stage1(self, B, x):
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def stage2(self, B, x):
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def stage3(self, B, x):
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleFeaturePyramid(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors, input_stride=16, top_block=None, norm=None):
        super().__init__()
        dim = in_channels
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            out_dim = dim
            if scale == 4.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2), LayerNorm(dim // 2),
                          nn.GELU(), nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError
            layers.extend([torch.nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False), LayerNorm(out_channels),
                           torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), LayerNorm(out_channels)])
            self.stages.append(nn.Sequential(*layers))

    def forward(self, features):
        results = [stage(features) for stage in self.stages]
        results.append(F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0))
        return results


class PredictHead(nn.Module):
    def __init__(self, feature_channels, embed_dim=256, predict_channels=1, norm="BN"):
        super().__init__()
        self.linear_fuse = nn.Conv2d(in_channels=embed_dim * 5, out_channels=embed_dim, kernel_size=1)
        if norm == "LN":
            self.norm = LayerNorm(embed_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)
        self.dropout = nn.Dropout()
        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4, c5 = x
        n, _, h, w = c1.shape
        _c1 = F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        _c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        _c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        _c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        _c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, _c5], dim=1))
        _c = self.norm(_c)
        x = self.dropout(_c)
        return self.linear_predict(x)


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))


class AniXplore(nn.Module):
    """AniXplore model for AI-generated anime image detection (inference only)"""
    def __init__(self, seg_pretrain_path=None, conv_pretrain=False, image_size=512):
        super().__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.dct = DctFrequencyExtractor()
        self.high_dwt = DwtFrequencyExtractor()
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(96 + 64, 96, kernel_size=1),
            nn.Conv2d(192 + 128, 192, kernel_size=1),
            nn.Conv2d(384 + 320, 384, kernel_size=1),
        ])
        self.predict_head = PredictHead(feature_channels=[256 for _ in range(5)], embed_dim=256, norm="BN")
        self.featurePyramid_net = SimpleFeaturePyramid(in_channels=384, out_channels=256,
                                                        scale_factors=(4.0, 2.0, 1.0, 0.5), top_block=None, norm="LN")
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(384, 1))
        self.auto_weight = AutomaticWeightedLoss(num=2)

    @torch.no_grad()
    def predict(self, image):
        """Inference-only forward pass. Returns probability of being AI-generated."""
        high_dct_freq = self.dct.forward_high(image)
        high_dwt_freq = self.high_dwt.forward(image)
        high_freq = high_dct_freq * 0.5 + high_dwt_freq * 0.5
        low_freq = self.dct.forward_low(image)
        input_high = torch.concat([image, high_freq], dim=1)
        input_low = torch.concat([image, low_freq], dim=1)

        B = input_low.shape[0]
        x = self.convnext.stem(input_high)
        x = self.convnext.stages[0](x)
        y = self.segformer.stage1(B, input_low)
        fused_feat = self.fusion_layers[0](torch.cat([x, y], dim=1))

        x = self.convnext.stages[1](fused_feat)
        y = self.segformer.stage2(B, y)
        fused_feat = self.fusion_layers[1](torch.cat([x, y], dim=1))

        x = self.convnext.stages[2](fused_feat)
        y = self.segformer.stage3(B, y)
        fused_feat = self.fusion_layers[2](torch.cat([x, y], dim=1))

        raw_cls_logit = self.cls_head(fused_feat)
        prob = torch.sigmoid(raw_cls_logit).squeeze()
        return prob.item() if prob.dim() == 0 else prob.tolist()
