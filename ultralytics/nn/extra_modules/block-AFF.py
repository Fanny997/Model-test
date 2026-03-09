import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['Flatten','GatedResidualFusion', 'RCSAB', 'DynamicResidualGroup', 'C3k2_EFE', 'SPDConv', 'Multibranch']



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GatedResidualFusion(nn.Module):
    """
    改进版融合模块：基于 DSTransNet 的 CSCA，但加入了 AFF 的残差思想
    解决问题：防止 EFE 的强边缘特征被随机初始化的权重“稀释”或“破坏”
    """

    def __init__(self, main_channels, aux_channels):
        super(GatedResidualFusion, self).__init__()

        # 1. 空间注意力 (Spatial Attention) - 决定“哪里”是重要的
        # 初始时，我们希望这个 Conv 输出极小，经过 Sigmoid 后 Mask 接近 0.5 (或通过 bias 控制)
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

        # 2. 通道注意力 (Channel Attention) - 决定“哪个通道”是重要的
        self.mlp_main = nn.Sequential(
            Flatten(),
            nn.Linear(main_channels, aux_channels)  # 跨通道交互
        )
        self.mlp_aux = nn.Sequential(
            Flatten(),
            nn.Linear(aux_channels, aux_channels)
        )

        # --- 【关键点 1：零初始化技巧】 ---
        # 强制初始化最后一层权重为 0。
        # 效果：训练开始时，gate_weight 全为 0，Mask 接近 0.5 (Sigmoid(0)) 或受 bias 控制。
        # 配合下文的残差连接，保证初始状态下，融合模块 = 直接相加 (Baseline状态)，绝不会掉点。
        nn.init.constant_(self.spatial_gate.weight, 0)
        nn.init.constant_(self.spatial_gate.bias, 0)

        nn.init.constant_(self.mlp_main[1].weight, 0)
        nn.init.constant_(self.mlp_main[1].bias, 0)
        nn.init.constant_(self.mlp_aux[1].weight, 0)
        nn.init.constant_(self.mlp_aux[1].bias, 0)

    def forward(self, x_main, x_aux):
        """
        x_main: 上下文特征 (DRG/Conv 输出) - 语义强，定位准
        x_aux : 边缘特征 (EFE Sobel 输出) - 细节多，但有噪声
        """

        # --- A. 计算通道权重 ---
        # 利用全局平均池化捕捉全局上下文
        pool_main = F.adaptive_avg_pool2d(x_main, (1, 1))
        pool_aux = F.adaptive_avg_pool2d(x_aux, (1, 1))

        # 计算两路特征的通道相关性
        w_main = self.mlp_main(pool_main)
        w_aux = self.mlp_aux(pool_aux)
        channel_weight = torch.sigmoid(w_main + w_aux).unsqueeze(2).unsqueeze(3)

        # --- B. 计算空间权重 ---
        # 在通道维度取最大值，聚焦显著区域
        max_main, _ = torch.max(x_main, dim=1, keepdim=True)
        max_aux, _ = torch.max(x_aux, dim=1, keepdim=True)
        spatial_cat = torch.cat([max_main, max_aux], dim=1)

        spatial_weight = torch.sigmoid(self.spatial_gate(spatial_cat))

        # --- C. 动态筛选 (Attention) ---
        # 综合 空间+通道 权重，对边缘特征(x_aux)进行“去噪”
        # 这一步会抑制掉云层边缘 (Spatial=0) 和 无效通道 (Channel=0)
        x_aux_refined = x_aux * channel_weight * spatial_weight

        # --- 【关键点 2：残差连接 (The Fix)】 ---
        # 即使 x_aux_refined 被算坏了，我们也保留了 x_main 和 x_aux 的原始信息。
        # 这里的逻辑是：主特征 + 原始边缘 + 提纯后的边缘
        # 这样保证了 EFE 的强边缘绝对不会被“稀释”，只会通过 Refined 项被“增强”。
        out = x_main + x_aux + x_aux_refined

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
    def __init__(self, inc, ouc):
        super().__init__()
        self.sobel = SobelConv(inc)  # 边缘分支
        self.conv = Conv(inc, inc, 3)  # 语义分支

        # 使用上面的改进融合模块
        self.fusion = GatedResidualFusion(main_channels=inc, aux_channels=inc)

        self.conv_out = Conv(inc, ouc, 1)  # 注意：这里输入是 inc (因为是相加)，不是 inc*2
        self.drg = DynamicResidualGroup(default_conv, ouc, 3, 8, 2)

    def forward(self, x):
        x_edge = self.sobel(x)
        x_semantic = self.conv(x)

        # 使用改进的融合方式
        # 这一步由于有残差连接，绝对安全，不会掉点
        x_fused = self.fusion(x_main=x_semantic, x_aux=x_edge)

        # 输出
        out = self.conv_out(x_fused)
        out = self.drg(out) + out
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