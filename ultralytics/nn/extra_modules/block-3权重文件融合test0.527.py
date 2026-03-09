import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['MDTE_Simplified', 'GCAI_Fusion', 'RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch']


# ================================================================
# 1. 兼容版 GCAI_Fusion (能同时跑 RDIAN2/4 的旧模型 和 RDIAN3 的新模型)
# ================================================================
class GCAI_Fusion(nn.Module):
    def __init__(self, c_low, c_high, c_out):
        super().__init__()
        # 公共组件
        self.conv_low = nn.Conv2d(c_low, c_out, 1)
        self.conv_high = nn.Conv2d(c_high, c_out, 1)
        self.out_conv = nn.Conv2d(c_out, c_out, 3, padding=1)

        # === 新版本组件 (Code Current Version) ===
        self.att_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c_out // 4, c_out, 1),
            nn.Sigmoid()
        )
        self.att_spatial = nn.Sequential(
            nn.Conv2d(c_out, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # === 旧版本组件 (Legacy Support) ===
        # 注意：这里定义是为了结构完整，实际加载旧权重时，这些层会直接从文件中恢复
        self.att_conv = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out // 2, 1),
            nn.BatchNorm2d(c_out // 2),
            nn.ReLU(),
            nn.Conv2d(c_out // 2, c_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_low, x_high = x
        if x_low.size(2) != x_high.size(2):
            x_high = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)

        low_feat = self.conv_low(x_low)
        high_feat = self.conv_high(x_high)

        # 【核心修复逻辑】自动检测权重版本
        if hasattr(self, 'att_high'):
            # --- 新版本逻辑 (RDIAN3) ---
            high_att = self.att_high(high_feat) * high_feat
            high_att = self.att_spatial(high_att) * high_att
            fused = low_feat + high_att
        elif hasattr(self, 'att_conv'):
            # --- 旧版本逻辑 (RDIAN2/4) ---
            concat_feat = torch.cat([low_feat, high_feat], dim=1)
            attention_map = self.att_conv(concat_feat)
            fused = low_feat * attention_map + high_feat * (1 - attention_map)
        else:
            # 兜底逻辑 (直接相加)
            fused = low_feat + high_feat

        return self.out_conv(fused)


# ================================================================
# 2. 兼容版 DynamicSpatialAttention (之前修好的那个)
# ================================================================
class DynamicSpatialAttention(nn.Module):
    def __init__(self, n_feat, reduction=4, k_size=7):
        super(DynamicSpatialAttention, self).__init__()
        # 0.520 旧模型结构
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, 9, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

        # 0.494/0.477 新模型结构
        self.conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // 16, n_feat, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, k_size, padding=k_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 自动检测使用哪种逻辑
        if hasattr(self, 'kernel_generator'):
            return self.forward_dynamic_conv(x)
        else:
            return self.forward_standard(x)

    def forward_dynamic_conv(self, x):
        """0.520 Model Logic"""
        b, c, h, w = x.size()
        mask = self.kernel_generator(x)
        mask = self.sigmoid(mask)
        mask = mask.view(b, 1, 9, 1, 1)
        x_unfold = F.unfold(x, kernel_size=3, padding=1)
        x_unfold = x_unfold.view(b, c, 9, h, w)
        out = (x_unfold * mask).sum(dim=2)
        return out

    def forward_standard(self, x):
        """0.494/0.477 Model Logic"""
        # 通道注意力
        avg_out = F.adaptive_avg_pool2d(x, 1)
        max_out = F.adaptive_max_pool2d(x, 1)
        y = avg_out + max_out

        if hasattr(self, 'conv'):
            y = self.conv(y)
            x = x * y
        elif hasattr(self, 'fc'):
            b, c, _, _ = x.size()
            y = y.view(b, -1)
            y = self.fc(y)
            y = y.view(b, c, 1, 1)
            x = x * y

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)

        if hasattr(self, 'spatial'):
            scale = self.spatial(scale)
            x = x * scale
        return x


# ================================================================
# 3. 其他模块 (保持不变)
# ================================================================

class MDTE_Simplified(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.direction_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        diff = self.direction_conv(x)
        att = self.sigmoid(diff)
        return x * att + x


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
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[
            :, :, 0]


class EFE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)
        self.drg = DynamicResidualGroup(conv=default_conv, n_feat=ouc, kernel_size=3, reduction=4, n_resblocks=2)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x_fuse = self.conv2(x_feature + x)
        x_drg = self.drg(x_fuse - x) + x
        return x_drg


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
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(x.size(1) * self.e), int(x.size(1) * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))