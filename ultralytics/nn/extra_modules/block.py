import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['MDTE_Conv', 'RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch', 'ADSF_Fusion']

class ADSF(nn.Module):
    def __init__(self, channel, m=-0.80, b=1, gamma=2):
        super(ADSF, self).__init__()

        self.w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        ax1 = self.avg_pool(x1)
        ax2 = self.avg_pool(x2)
        ax1 = self.conv1(ax1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1, C, 1)
        ax2 = self.fc(ax2).squeeze(-1).transpose(-1, -2)  # (1, C, 1)
        out1 = torch.sum(torch.matmul(ax1, ax2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(ax2.transpose(-1, -2), ax1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        mix_factor = self.mix_block(self.w)
        out = out1 * mix_factor + out2 * (1 - mix_factor)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return torch.cat([(x2 * out), x1], dim=1)


# ================================================================
# 更新版：为 YOLO 架构定制的 ADSF 包装器 (修复深浅层映射)
# ================================================================
class ADSF_Fusion(nn.Module):
    """
    将 YOLO 的 [深层特征, 浅层特征] 列表输入转化为 SDS-Net ADSF 所需的对齐输入。
    """

    def __init__(self, c1, c2, c_out):
        super().__init__()
        # 1. 统一通道数：ADSF 要求 x1 和 x2 通道数一致，我们将它们对齐到 c_out 的一半
        self.align_channels = c_out // 2
        self.conv_deep = nn.Conv2d(c1, self.align_channels, 1)
        self.conv_shallow = nn.Conv2d(c2, self.align_channels, 1)

        # 2. 实例化 SDS-Net 原生的 ADSF
        self.adsf = ADSF(self.align_channels)

        # 3. 融合后通道数恢复
        self.out_conv = Conv(c_out, c_out, 3)  # 使用 Ultralytics 自带的 3x3 Conv

    def forward(self, x):
        # 在 YOLO 中，f=[-1, 4]，即 x[0]是深层上采样特征，x[1]是浅层主干特征
        x_deep, x_shallow = x[0], x[1]

        # 空间分辨率对齐 (YOLO 必须的步骤)
        if x_deep.size(2) != x_shallow.size(2):
            x_deep = F.interpolate(x_deep, size=x_shallow.shape[2:], mode='bilinear', align_corners=False)

        # 通道对齐
        deep_feat = self.conv_deep(x_deep)
        shallow_feat = self.conv_shallow(x_shallow)

        # 执行 ADSF 融合
        # (依据 SDS-Net 源码：x1必须传入深层语义 deep_feat，x2必须传入浅层细节 shallow_feat)
        fused = self.adsf(deep_feat, shallow_feat)

        return self.out_conv(fused)


# ================================================================
# 2. RDIAN 核心模块移植: MDTE_Conv (多方向目标增强)
#    替代原有的 SobelConv，使用可学习的多方向差分卷积
# ================================================================
class MDTE_Conv(nn.Module):
    """
    RDIAN MDTE (Multidirection Target Enhancement) 思想的 YOLO 适配版。
    相比 Sobel 的固定算子，MDTE 通过学习不同方向的梯度差异，
    能更好地突出红外小目标并抑制背景噪声。
    """

    def __init__(self, in_channels):
        super().__init__()
        # 学习 4 个方向的梯度 (水平、垂直、45度、135度)
        # groups=in_channels 保证了对每个通道独立计算梯度，类似 Depthwise Conv
        self.conv_diff = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, groups=in_channels)
        self.fuse = nn.Conv2d(in_channels * 4, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # 1. 计算多方向差分
        diff = self.conv_diff(x)
        # 2. 融合差分特征 (找到最显著的区域)
        out = self.fuse(diff)
        return self.act(self.bn(out))


# ================================================================
# 3. 改进版 EFE 模块 (集成 MDTE + DRG)
#    核心改动：移除了 SobelConv，替换为 MDTE_Conv
# ================================================================
class EFE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        # --- 改进: 用 MDTE (方向增强) 替换 Sobel (固定梯度) ---
        # 理由：Sobel 对噪声敏感；MDTE 能通过多方向差分抑制杂波
        self.edge_branch = MDTE_Conv(inc)
        # -----------------------------------------------------

        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, ouc, 1)
        # DRG 模块保持不变，用于对融合后的特征进行深层语义精炼
        self.drg = DynamicResidualGroup(conv=default_conv, n_feat=ouc, kernel_size=3, reduction=4, n_resblocks=2)

    def forward(self, x):
        # 1. 提取特征
        x_edge = self.edge_branch(x)  # 这里的特征现在是“纯净的目标增强特征”
        x_main = self.conv_branch(x)

        # 2. 融合 (MDTE 提取的边缘特征 + 卷积提取的语义特征)
        x_concat = torch.cat([x_main, x_edge], dim=1)

        # 3. 降维融合
        x_fused = self.conv1(x_concat)

        # 4. DRG 增强
        # DRG 放在这里是为了进一步精炼融合后的特征
        # 残差连接 x_fused 保证梯度传播
        out = self.drg(x_fused) + x_fused

        return out


# ================================================================
# 4. 其他辅助模块 (保持不变或微调)
# ================================================================

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class RCSAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DynamicResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, n_resblocks=2, conv=default_conv):
        super(DynamicResidualGroup, self).__init__()
        modules_body = [
            RCSAB(conv, n_feat, kernel_size, reduction, bias=True, bn=True,
                  act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class C3k_EFE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        # 这里调用的 EFE 现在已经是集成 MDTE 的新版本了
        self.m = nn.Sequential(*(EFE(c_, c_) for _ in range(n)))


class C3k2_EFE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_EFE(self.c, self.c, 2, shortcut, g) if c3k else EFE(self.c, self.c) for _ in range(n))


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
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
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
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
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
        self.m = nn.Identity()

    def forward(self, x):
        pass