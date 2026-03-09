import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
import torch
import types  # 【关键】需要导入这个库来绑定方法

# 1. 准备权重文件路径
weight_path_a = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-RDIAN2/weights/best.pt'
weight_path_b = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-RDIAN4/weights/best.pt'

# 2. 初始化环境
print("正在初始化融合模型环境...")
model = YOLO(weight_path_a)

# 3. 加载融合模型
print(f"正在融合权重: {weight_path_a} 和 {weight_path_b}")
ensemble_model = attempt_load_weights([weight_path_a, weight_path_b], device=model.device, fuse=True)

# ==================== 【补丁 1: 修复 fuse 报错】 ====================
def dummy_fuse(verbose=True):
    return ensemble_model
ensemble_model.fuse = dummy_fuse

# ==================== 【补丁 2: 修复 embed 报错】 ====================
# 定义一个新的 forward 函数，专门接收 embed 和其他杂七杂八的参数
def new_forward(self, x, augment=False, profile=False, visualize=False, embed=None, **kwargs):
    # 这里我们只用核心参数调用子模型，忽略 embed
    y = [module(x, augment, profile, visualize)[0] for module in self]
    y = torch.cat(y, 2)  # nms ensemble
    return y, None

# 使用 types.MethodType 将这个新函数绑定给 ensemble_model 实例
ensemble_model.forward = types.MethodType(new_forward, ensemble_model)
# =================================================================

# 4. 替换模型
model.model = ensemble_model

# 5. 运行验证
print("开始在测试集上验证融合模型...")
results = model.val(
    data='/home/Cug-Rs02/20251011/IRSTD-YOLO/datasets/config.yaml',
    split='test',
    imgsz=512,
    batch=8,  # 显存够的话可以改回 8
    iou=0.5,
    save_json=True,
    project='runs/datasets/test',
    name='ensemble_result'
)

print("-" * 30)
print(f"融合模型 (Ensemble) 的测试集 mAP50: {results.box.map50:.5f}")
print("-" * 30)