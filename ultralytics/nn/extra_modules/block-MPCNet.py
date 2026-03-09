import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch', 'SCFM']


class SCFM(nn.Module):
    """
    Semantic-Context Feature Modulation (基于 MPCNet 改进)
    解决特征稀释问题的核心模块
    """

    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()

        # 1. 降维/对齐通道
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        # 2. 交叉注意力调制 (Cross-Attention Modulation)
        # 这是一个轻量级的注意力生成器
        self.attention_generator = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, 1),  # 融合两路信息
            nn.ReLU(True),
            nn.Conv2d(out_channels // 2, 2, 1),  # 生成两个通道的权重: [Weight_Low, Weight_High]
            nn.Softmax(dim=1)  # 保证权重和为1 (软融合，不会导致梯度消失)
        )

        # 3. 融合后的处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        # --- 【关键技巧：零初始化】 ---
        # 初始化注意力生成器的最后一层为0，使得初始权重为 0.5/0.5
        # 这样训练初期相当于简单的 Add 融合，避免了“瞎指挥”导致的掉点
        nn.init.constant_(self.attention_generator[2].weight, 0)
        nn.init.constant_(self.attention_generator[2].bias, 0)

    def forward(self, x_low, x_high):
        # x_low: 边缘特征 (Sobel)
        # x_high: 语义特征 (Conv/DRG)

        # 1. 对齐特征
        feat_low = self.low_proj(x_low)
        feat_high = self.high_proj(x_high)

        # 2. 拼接并生成动态权重
        # 这里的逻辑是：网络通过看两路特征，自己决定每一路该占多少比例
        concat = torch.cat([feat_low, feat_high], dim=1)
        weights = self.attention_generator(concat)  # [B, 2, H, W]

        # 3. 分离权重
        w_low = weights[:, 0:1, :, :]
        w_high = weights[:, 1:2, :, :]

        # 4. 加权融合 (Soft Fusion)
        # 不会像乘法门控那样把特征搞成0，因为 w_low + w_high = 1
        feat_fused = w_low * feat_low + w_high * feat_high

        # 5. 进一步提取
        out = self.fusion_conv(feat_fused)

        # 6. 残差连接 (兜底策略)
        # 如果输入输出通道一致，建议加个残差；如果不一致就算了
        return out

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
        # 边缘分支
        self.sobel_branch = SobelConv(inc)
        # 语义分支
        self.conv_branch = Conv(inc, inc, 3)

        # --- 使用 MPCNet 的 SCFM 模块 ---
        # 注意：这里我们让 SCFM 直接输出 ouc 通道，省去了后面的 conv1
        self.fusion = SCFM(low_channels=inc, high_channels=inc, out_channels=ouc)
        # --------------------------------

        # DRG 模块
        self.drg = DynamicResidualGroup(conv=default_conv, n_feat=ouc, kernel_size=3, reduction=4, n_resblocks=2)

    def forward(self, x):
        x_edge = self.sobel_branch(x)
        x_main = self.conv_branch(x)

        # 使用 SCFM 进行软融合
        # 它会自动平衡 边缘信息 和 语义信息
        x_fused = self.fusion(x_low=x_edge, x_high=x_main)

        # DRG 增强 + 残差
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