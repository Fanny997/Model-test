import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# 设置通用绘图风格
plt.style.use('default')


def create_figure(title, size=(10, 6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def draw_box(ax, xy, width, height, color, label, text_color='black', fontsize=10):
    rect = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02",
                                  edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)
    return rect


def draw_arrow(ax, xy_from, xy_to, color='black'):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


# ==============================================================================
# 1. & 2. Sobel 噪声示意图与对比图
# ==============================================================================
def draw_sobel_noise():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图：Sobel 处理后的嘈杂特征图 ---
    # 生成模拟的云层背景噪声 + 弱目标
    np.random.seed(42)
    noise = np.random.normal(0.5, 0.1, (100, 100))
    # 模拟云层边缘 (梯度)
    for i in range(100):
        noise[i, :] += np.sin(i / 10) * 0.2

    axes[0].imshow(noise, cmap='gray')
    axes[0].set_title("Sobel Output (Simulated)\nHigh Background Noise", fontsize=12)
    # 打一个大红叉
    axes[0].text(50, 50, "❌", fontsize=100, color='red', ha='center', va='center', alpha=0.7)
    axes[0].axis('off')

    # --- 右图：当前模型架构简图 (噪声放大) ---
    ax2 = axes[1]
    ax2.set_xlim(0, 1);
    ax2.set_ylim(0, 1);
    ax2.axis('off')

    # 绘制模块
    draw_box(ax2, (0.1, 0.4), 0.2, 0.2, '#ADD8E6', 'EFE\n(Sobel)')
    draw_box(ax2, (0.7, 0.4), 0.2, 0.2, '#FFB6C1', 'DRG\n(Attention)')

    # 连接箭头
    ax2.annotate("", xy=(0.7, 0.5), xytext=(0.3, 0.5), arrowprops=dict(arrowstyle="->", lw=2))

    # 红色箭头标出噪声放大
    ax2.annotate("Noise Amplification", xy=(0.5, 0.52), xytext=(0.5, 0.7),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. & 6. & 10. 改进后的 EFE 结构图 (含 MDTE)
# ==============================================================================
def draw_improved_efe():
    fig, ax = create_figure("Improved EFE Module Structure (Slide 4)", size=(12, 6))

    # EFE 大框
    efe_frame = patches.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='gray', linestyle='--', lw=2)
    ax.add_patch(efe_frame)
    ax.text(0.07, 0.92, "EFE Module", fontsize=12, fontweight='bold', color='gray')

    # 输入
    draw_box(ax, (0.08, 0.45), 0.08, 0.1, '#D3D3D3', 'Input')

    # 分支路径
    # 上路: Conv 3x3 (语义)
    draw_arrow(ax, (0.16, 0.55), (0.25, 0.75))
    draw_box(ax, (0.25, 0.7), 0.15, 0.1, '#FFD700', 'Conv 3x3\n(Semantic)')

    # 下路: MDTE (边缘) - 绿色高亮
    draw_arrow(ax, (0.16, 0.45), (0.25, 0.25))
    # MDTE Block 内部细节
    mdte_rect = patches.Rectangle((0.24, 0.1), 0.35, 0.3, color='#90EE90', alpha=0.3)  # 绿色背景
    ax.add_patch(mdte_rect)
    ax.text(0.4, 0.05, "MDTE Block (Highlighted)", color='green', ha='center', fontsize=10)

    # 4个方向卷积示意
    for i, angle in enumerate([0, 45, 90, 135]):
        y_pos = 0.35 - i * 0.07
        ax.text(0.26, y_pos, f"Dir {angle}°", fontsize=8)
        draw_arrow(ax, (0.32, y_pos), (0.38, y_pos))

    draw_box(ax, (0.38, 0.15), 0.08, 0.2, '#98FB98', 'Fusion')
    draw_box(ax, (0.48, 0.15), 0.08, 0.2, '#98FB98', 'BN+ReLU')

    # Concat
    draw_arrow(ax, (0.4, 0.75), (0.62, 0.55))  # 上路下来
    draw_arrow(ax, (0.56, 0.25), (0.62, 0.45))  # 下路上来
    draw_box(ax, (0.62, 0.4), 0.08, 0.2, '#87CEEB', 'Concat')

    # 后处理 Conv 1x1 -> DRG -> Output
    draw_arrow(ax, (0.7, 0.5), (0.73, 0.5))
    draw_box(ax, (0.73, 0.45), 0.08, 0.1, '#FFA07A', 'Conv 1x1')
    draw_arrow(ax, (0.81, 0.5), (0.84, 0.5))
    draw_box(ax, (0.84, 0.45), 0.08, 0.1, '#FF69B4', 'DRG')
    draw_arrow(ax, (0.92, 0.5), (0.96, 0.5))
    ax.text(0.97, 0.5, "Output", fontsize=10)

    plt.show()


# ==============================================================================
# 4. GCAI_Fusion 逻辑流程图
# ==============================================================================
def draw_gcai_logic():
    fig, ax = create_figure("GCAI_Fusion Logic Flow", size=(8, 4))

    # High Input
    draw_box(ax, (0.1, 0.6), 0.15, 0.2, '#FFB6C1', 'High Input\n(Semantic)')

    # Sigmoid / Attention Generation
    draw_arrow(ax, (0.25, 0.7), (0.35, 0.7))
    draw_box(ax, (0.35, 0.6), 0.15, 0.2, '#FFD700', 'Sigmoid\nFunction')

    # Multiply
    draw_arrow(ax, (0.5, 0.7), (0.6, 0.7))
    draw_box(ax, (0.6, 0.6), 0.1, 0.2, 'white', 'X', fontsize=20)  # 乘号节点

    # Low Input
    draw_box(ax, (0.6, 0.2), 0.15, 0.2, '#ADD8E6', 'Low Input\n(Detail)')
    draw_arrow(ax, (0.675, 0.4), (0.65, 0.6))  # Low 连向乘号

    # Output
    draw_arrow(ax, (0.7, 0.7), (0.8, 0.7))
    draw_box(ax, (0.8, 0.6), 0.15, 0.2, '#90EE90', 'Fused\nOutput')

    plt.show()


# ==============================================================================
# 7. & 9. GCAI 原理图 (双流 + 3D 立方体示意)
# ==============================================================================
def draw_gcai_principle_3d():
    fig, ax = create_figure("GCAI Principle (Slide 5: Dual-Stream Masking)", size=(12, 7))

    def draw_3d_cube(ax, origin, size, color, label):
        x, y = origin
        w, h, d = size  # width, height, depth
        # Front
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color, edgecolor='k', alpha=0.9))
        # Top
        ax.add_patch(patches.Polygon([[x, y + h], [x + d, y + h + d], [x + w + d, y + h + d], [x + w, y + h]],
                                     facecolor=color, edgecolor='k', alpha=0.6))
        # Side
        ax.add_patch(patches.Polygon([[x + w, y], [x + w + d, y + d], [x + w + d, y + h + d], [x + w, y + h]],
                                     facecolor=color, edgecolor='k', alpha=0.4))
        ax.text(x + w / 2, y - 0.05, label, ha='center', fontsize=10, fontweight='bold')

    # 1. 左侧输入 (Cubes)
    draw_3d_cube(ax, (0.1, 0.6), (0.1, 0.1, 0.05), '#FF6347', 'High-Level\n(Semantic)')
    draw_3d_cube(ax, (0.1, 0.2), (0.1, 0.1, 0.05), '#4682B4', 'Low-Level\n(Detail)')

    # 2. 中间处理 (Attention Funnel)
    # High -> Attention Block
    ax.annotate("", xy=(0.35, 0.65), xytext=(0.22, 0.65), arrowprops=dict(arrowstyle="->", lw=2))
    # 漏斗形状
    polygon = patches.Polygon([[0.35, 0.75], [0.35, 0.55], [0.45, 0.63], [0.45, 0.67]],
                              facecolor='#FFD700', edgecolor='k')
    ax.add_patch(polygon)
    ax.text(0.4, 0.8, "Attention Block", ha='center')

    # 生成 Mask (平面)
    ax.add_patch(patches.Rectangle((0.5, 0.6), 0.1, 0.1, facecolor='gray', alpha=0.5, edgecolor='k'))
    ax.text(0.55, 0.65, "Mask\n(0-1)", ha='center', va='center', color='white')

    # 3. 融合操作 (Element-wise Multiply)
    ax.text(0.55, 0.45, "⊗", fontsize=30, ha='center')  # 乘号
    ax.annotate("", xy=(0.55, 0.58), xytext=(0.55, 0.50), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.55, 0.42), xytext=(0.22, 0.25),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    # 4. 残差连接 (Add)
    ax.text(0.75, 0.45, "⊕", fontsize=30, ha='center')  # 加号
    ax.annotate("", xy=(0.72, 0.46), xytext=(0.58, 0.46), arrowprops=dict(arrowstyle="->"))
    # High 残差连过来
    ax.annotate("", xy=(0.75, 0.48), xytext=(0.22, 0.68),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", linestyle="--"))

    # 5. 输出
    draw_3d_cube(ax, (0.85, 0.4), (0.1, 0.1, 0.05), '#32CD32', 'Fused\nFeature')

    plt.show()


# ==============================================================================
# 8. 整体架构展示 (YOLO11 Modified Topology)
# ==============================================================================
def draw_overall_architecture():
    fig, ax = create_figure("Modified YOLO11 Architecture (Topology)", size=(10, 8))

    # 定义层级位置
    layers = {
        'Input': (0.5, 0.95),
        'Backbone P1': (0.5, 0.85),
        'Backbone P2 (EFE)': (0.5, 0.75),
        'Backbone P3 (EFE)': (0.5, 0.65),
        'Backbone P4': (0.5, 0.55),
        'Backbone P5 (DRG)': (0.5, 0.45),
        'Neck (GCAI)': (0.2, 0.65),  # 侧边融合
        'Head': (0.5, 0.25),
        'Detect': (0.5, 0.1)
    }

    # 绘制骨干 (Backbone)
    draw_box(ax, (0.4, 0.92), 0.2, 0.05, '#D3D3D3', 'Input Image')
    draw_arrow(ax, (0.5, 0.92), (0.5, 0.88))

    draw_box(ax, (0.4, 0.82), 0.2, 0.06, '#D3D3D3', 'Stem / P1')
    draw_arrow(ax, (0.5, 0.82), (0.5, 0.78))

    # P2 EFE (改进点)
    draw_box(ax, (0.35, 0.72), 0.3, 0.06, '#90EE90', 'P2: MDTE-EFE\n(Clean Edge)', text_color='darkgreen')
    draw_arrow(ax, (0.5, 0.72), (0.5, 0.68))

    # P3 EFE (改进点)
    draw_box(ax, (0.35, 0.62), 0.3, 0.06, '#90EE90', 'P3: MDTE-EFE\n(Clean Edge)', text_color='darkgreen')
    draw_arrow(ax, (0.5, 0.62), (0.5, 0.58))

    # P4
    draw_box(ax, (0.4, 0.52), 0.2, 0.06, '#D3D3D3', 'P4: C3k2')
    draw_arrow(ax, (0.5, 0.52), (0.5, 0.48))

    # P5 DRG (改进点)
    draw_box(ax, (0.35, 0.42), 0.3, 0.06, '#FF69B4', 'P5: DRG + SPPF\n(Semantic Enhance)', text_color='maroon')

    # Neck / GCAI 连接
    # 从 P5 上采样回去
    ax.annotate("", xy=(0.3, 0.65), xytext=(0.35, 0.45),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", color='blue', linestyle='--'))

    # GCAI 模块
    draw_box(ax, (0.1, 0.62), 0.2, 0.06, '#FFD700', 'Neck: GCAI_Fusion', text_color='darkorange')
    # 连接线: P3 (Low) -> GCAI
    ax.annotate("", xy=(0.3, 0.65), xytext=(0.35, 0.65), arrowprops=dict(arrowstyle="->"))

    # Head
    draw_arrow(ax, (0.5, 0.42), (0.5, 0.3))
    draw_box(ax, (0.3, 0.2), 0.4, 0.1, '#ADD8E6', 'FPN / PANet Head')

    # Detect
    draw_arrow(ax, (0.5, 0.2), (0.5, 0.15))
    draw_box(ax, (0.4, 0.08), 0.2, 0.07, '#FF6347', 'Detect Head')

    plt.show()


# 执行所有绘图函数
if __name__ == "__main__":
    draw_sobel_noise()
    draw_improved_efe()
    draw_gcai_logic()
    draw_gcai_principle_3d()
    draw_overall_architecture()