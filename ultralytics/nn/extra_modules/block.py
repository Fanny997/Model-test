import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['DSA', 'RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch']

# =======================================================
# 1. 提前声明 default_conv，防止下面类找不到它报错
# =======================================================
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

###################################################### C3k2-EFE start #########################################################################
# =======================================================
# M3FD 满血版注意力机制与残差组 (直接覆盖原有的这三个类)
# =======================================================
class DSA(nn.Module):
    """动态空间与通道注意力 (大目标感知特化 7x7)"""

    def __init__(self, channel, reduction=16):
        super(DSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_c = self.avg_pool(x).view(b, c)
        y_c = self.fc(y_c).view(b, c, 1, 1)
        x_c = x * y_c.expand_as(x)

        avg_out = torch.mean(x_c, dim=1, keepdim=True)
        max_out, _ = torch.max(x_c, dim=1, keepdim=True)
        y_s = torch.cat([avg_out, max_out], dim=1)
        y_s = self.sigmoid_spatial(self.spatial_conv(y_s))

        return x_c * y_s


class RCSAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

        # 【核心修复】：直接传 n_feat！再也没有烦人的 in_channels 报错了
        self.dsa = DSA(n_feat)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.dsa(res)
        res += x
        return res


class DynamicResidualGroup(nn.Module):
    def __init__(self, n_feat, conv=default_conv, kernel_size=3, reduction=4, n_resblocks=2):
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
# ================================================================
# 恢复：MDTE_Conv (多方向目标增强边缘提取) - 专治 M3FD 复杂背景
# ================================================================
class MDTE_Conv(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 使用 4 个方向的可学习差分卷积 (利用 groups 实现独立通道处理，极轻量)
        self.dir_conv = nn.Conv2d(c, c * 4, kernel_size=3, padding=1, groups=c, bias=False)
        self.fuse = nn.Conv2d(c * 4, c, kernel_size=1)
        self.act = nn.GELU()  # 使用 GELU 平滑特征

    def forward(self, x):
        # 提取多方向软响应并融合，过滤各向异性的背景干扰
        out = self.act(self.dir_conv(x))
        return self.fuse(out)

class EFE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        # 【换回 MDTE】
        self.edge_branch = MDTE_Conv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, ouc, 1)
        self.drg = DynamicResidualGroup(n_feat=ouc, conv=default_conv, kernel_size=3, reduction=4, n_resblocks=2)

    def forward(self, x):
        x_edge = self.edge_branch(x)
        x_main = self.conv_branch(x)
        x_concat = torch.cat([x_main, x_edge], dim=1)
        x_fused = self.conv1(x_concat)
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
###################################################### C3k2-EFE end #########################################################################


###################################################### IRSTE start #########################################################################
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
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
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
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
###################################################### IRSTE end #########################################################################