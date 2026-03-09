import torch
import warnings

warnings.filterwarnings('ignore')

# 请替换为你那个 0.520 模型的真实路径
weight_path = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/train/exp-DRPCA7/weights/best.pt'

print(f"正在检查权重文件: {weight_path}")

try:
    ckpt = torch.load(weight_path, map_location='cpu')
    model = ckpt['model']

    print("\n=== 侦探报告: DynamicSpatialAttention 内部结构 ===")
    found = False
    for name, module in model.named_modules():
        # 找到第一个 DynamicSpatialAttention 模块
        if 'DynamicSpatialAttention' in str(type(module)):
            print(f"找到模块: {name}")
            print(f"模块类型: {type(module)}")
            print("该模块包含的子层（属性）:")
            for key, _ in module.named_children():
                print(f"  - self.{key}")

            print("\n该模块包含的参数键值:")
            for key, _ in module.state_dict().items():
                print(f"  - {key}")

            found = True
            break

    if not found:
        print("❌ 未在模型中找到 DynamicSpatialAttention 模块！可能名字变了？")

except Exception as e:
    print(f"❌ 读取失败: {e}")