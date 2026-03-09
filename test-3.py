import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
import torch
import types

# 1. 准备权重文件路径
# 模型 A: 0.477 (泛化好)
weight_path_a = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-RDIAN2/weights/best.pt'
# 模型 B: 0.494 (拟合好)
weight_path_b = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-RDIAN3/weights/best.pt'
# 模型 C: 0.520 (无GCAI的原版架构，精度最高) - 请确认路径是否正确
weight_path_c = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-RDIAN4/weights/best.pt' # 请修改这里的路径！

# 2. 初始化环境
# 随便加载一个模型来初始化 YOLO 环境对象
print("正在初始化融合模型环境...")
model = YOLO(weight_path_a)

# 3. 加载融合模型 (3合1)
print(f"正在融合3个权重...")
# 列表里可以放任意数量的模型权重 [a, b, c, d...]
ensemble_model = attempt_load_weights(
    [weight_path_a, weight_path_b, weight_path_c],
    device=model.device,
    fuse=True
)

# ==================== 【补丁 1: 修复 fuse 报错】 ====================
def dummy_fuse(verbose=True):
    return ensemble_model
ensemble_model.fuse = dummy_fuse

# ==================== 【补丁 2: 修复 embed 报错】 ====================
# 这个函数会自动遍历列表中的所有模型，无论有几个
def new_forward(self, x, augment=False, profile=False, visualize=False, embed=None, **kwargs):
    # 遍历 self (即包含3个模型的列表)，分别推理
    y = [module(x, augment, profile, visualize)[0] for module in self]
    # 将3个模型的预测结果在维度2上拼接 (NMS Ensemble)
    y = torch.cat(y, 2)
    return y, None

ensemble_model.forward = types.MethodType(new_forward, ensemble_model)
# =================================================================

# 4. 替换模型
model.model = ensemble_model

# 5. 运行验证
print("开始在测试集上验证 3模型融合结果...")
results = model.val(
    data='/home/Cug-Rs02/20251011/IRSTD-YOLO/datasets/config.yaml',
    split='test',
    imgsz=512,
    batch=8,  # 3个模型比较吃显存，如果报错请改为 2 或 1
    iou=0.5,
    save_json=True,
    project='runs/datasets/test',
    name='ensemble_3_models'
)

print("-" * 30)
print(f"3模型融合 (Ensemble) 的测试集 mAP50: {results.box.map50:.5f}")
print("-" * 30)