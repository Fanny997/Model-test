import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = [
    'GCAI_Fusion', 'RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'C3k_EFE',
    'MDTE_Conv', 'SPDConv', 'Multibranch', 'SobelConv', 'EFE', 'DynamicSpatialAttention'
]

class GCAI_Fusion(nn.Module):
    def __init__(self, c_low, c_high, c_out):
        super().__init__()
        self.align_high = Conv(c_high, c_low, 1) if c_high != c_low else nn.Identity()

        self.mask_gen = nn.Sequential(
            Conv(c_low * 2, c_low, 1),
            nn.Conv2d(c_low, c_low, 1),
            nn.Sigmoid()
        )
        self.out_conv = Conv(c_low, c_out, 3)

    def forward(self, x_low, x_high):
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:], mode='nearest')
        x_high_aligned = self.align_high(x_high_up)

        w = self.mask_gen(torch.cat([x_low, x_high_aligned], dim=1))
        gated_feat = x_low * w + x_high_aligned * (1 - w)

        # 恒等残差注入，防止梯度截断
        fused = x_low + gated_feat
        return self.out_conv(fused)

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

class MDTE_Conv(nn.Module):
    """
    采用多方向局部差分先验，替代死板的 Sobel，实现动态源头清洗
    """
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = Conv(c1, c2, k, p=k//2)
        self.dir_att = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 物理先验：目标是局部极值点。利用差分突出目标，并用注意力掩码过滤杂波
        local_diff = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.conv(x * self.dir_att(local_diff))

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class RCSAB(nn.Module):
    def __init__(self, in_channels):
        super(RCSAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.ca(out)
        out = out * self.sa(out)
        out += residual
        return self.relu(out)

class DynamicResidualGroup(nn.Module):
    def __init__(self, c1, c2=None, num_blocks=2):
        super(DynamicResidualGroup, self).__init__()
        c2 = c2 or c1
        self.reduce_conv = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*[RCSAB(c2) for _ in range(num_blocks)])
        self.conv_out = Conv(c2, c2, 3)

    def forward(self, x):
        x = self.reduce_conv(x)
        residual = x
        out = self.blocks(x)
        out = self.conv_out(out)
        out += residual
        return out

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        # 标准 Sobel 算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        # 重塑为 PyTorch 权重格式 (C, 1, 3, 3)
        weight_x = torch.from_numpy(sobel_x).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        weight_y = torch.from_numpy(sobel_y).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)

        self.conv_x.weight.data = weight_x
        self.conv_y.weight.data = weight_y
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        # 取 x 和 y 方向的梯度之和
        return self.conv_x(x) + self.conv_y(x)

class EFE(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)
        self.drg = DynamicResidualGroup(c1=ouc, c2=ouc, num_blocks=2)

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