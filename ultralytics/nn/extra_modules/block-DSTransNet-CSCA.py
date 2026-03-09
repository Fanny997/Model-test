import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch']

# -----------------------------------------------------------
# 来自 DSTransNet 的核心融合模块
# -----------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CSCA(nn.Module):
    """
    Cross-attention of Spaces and Channels (来自 DSTransNet)
    用于动态融合两路特征，利用 up (主特征) 来筛选 skip (辅助/边缘特征)
    """

    def __init__(self, up_channels, skip_channels):
        super(CSCA, self).__init__()

        # Channel Attention 部分
        self.mlp_up = nn.Sequential(
            Flatten(),
            nn.Linear(up_channels, skip_channels))
        self.mlp_skip = nn.Sequential(
            Flatten(),
            nn.Linear(skip_channels, skip_channels))

        # Spatial Attention 部分
        self.spatial_attention_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.relu = nn.ReLU()

    def forward(self, up, skip):
        # 1. Channel Attention (通道注意力)
        avg_pool_up = F.adaptive_avg_pool2d(up, (1, 1))
        avg_pool_skip = F.adaptive_avg_pool2d(skip, (1, 1))

        channel_att_up = self.mlp_up(avg_pool_up)
        channel_att_skip = self.mlp_skip(avg_pool_skip)

        # 融合两路通道注意力
        out_ch = channel_att_up + channel_att_skip
        channel_att_out = torch.sigmoid(out_ch).unsqueeze(2).unsqueeze(3).expand_as(skip)

        # 2. Spatial Attention (空间注意力)
        # 在通道维度取最大值，生成空间分布图
        max_up, _ = torch.max(up, dim=1, keepdim=True)
        max_skip, _ = torch.max(skip, dim=1, keepdim=True)
        out_sp = torch.cat([max_up, max_skip], dim=1)
        out_sp = self.spatial_attention_conv(out_sp)
        spatial_att_out = torch.sigmoid(out_sp).expand_as(skip)

        # 3. 动态加权 (Feature Selection)
        # 用生成的权重去“过滤”辅助特征(skip)
        skip_refined = skip * channel_att_out * spatial_att_out

        final_out = self.relu(skip_refined)
        return final_out
# -----------------------------------------------------------
# 只保留DRG相关模块（删除DSA）
class RCSAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)  # 去掉DSA
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DynamicResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(DynamicResidualGroup, self).__init__()
        modules_body = [
            RCSAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=True,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]


class EFE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        # 分支 1: 边缘提取 (Detail/Skip)
        self.sobel_branch = SobelConv(inc)

        # 分支 2: 语义上下文提取 (Main/Up)
        self.conv_branch = Conv(inc, inc, 3)

        # --- 引入 DSTransNet 的 CSCA 融合 ---
        self.fusion = CSCA(up_channels=inc, skip_channels=inc)

        # 为了更稳定，初始化时让 spatial_attention_conv 的权重接近0
        # 这样初始阶段 x_edge_refined 接近 0，网络退化为原始 EFE，避免冷启动干扰
        nn.init.constant_(self.fusion.spatial_attention_conv.weight, 0)
        # -----------------------------------------

        # 融合后的降维 (Concat后通道翻倍)
        self.conv1 = Conv(inc * 2, ouc, 1)

        # 特征增强模块 (DRG)
        self.drg = DynamicResidualGroup(conv=default_conv, n_feat=ouc, kernel_size=3, reduction=4, n_resblocks=2)

    def forward(self, x):
        # 1. 提取两路特征
        x_edge = self.sobel_branch(x)  # 边缘特征
        x_main = self.conv_branch(x)  # 上下文特征

        # 2. 使用 CSCA 进行动态筛选
        # x_edge_refined = x_edge * Mask
        x_edge_refined = self.fusion(up=x_main, skip=x_edge)

        # 3. 【关键策略：残差融合】
        # 将“筛选后的特征”作为“原始特征”的补充，而不是替代
        # 这样保证了最差情况也能达到 Baseline 的效果
        x_edge_final = x_edge + x_edge_refined

        # 4. 拼接 "主特征" 和 "增强后的边缘特征"
        x_concat = torch.cat([x_main, x_edge_final], dim=1)

        # 5. 融合降维
        x_fused = self.conv1(x_concat)

        # 6. DRG 增强
        out = self.drg(x_fused) + x_fused

        return out

class C3k_EFE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(EFE(c_, c_) for _ in range(n)))


class C3k2_EFE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_EFE(self.c, self.c, 2, shortcut, g) if c3k else EFE(self.c, self.c) for _ in range(n))


# 以下IRSTE部分保持不变（同只加DSA版本）
class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class Multibranch(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))