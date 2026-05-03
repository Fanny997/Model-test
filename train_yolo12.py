import sys
import os

# 【核心修复1】强制将当前脚本所在的目录加入系统路径，确保读取你本地的 ultralytics 代码
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def main():
    # 1. 加载 YOLOv12 模型
    # 这里推荐使用预训练权重 'yolo12s.pt' (Small版本) 作为初始化。
    # 你也可以根据算力选择 'yolo12n.pt' (Nano), 'yolo12m.pt' (Medium) 等。
    # 框架会自动根据这个名字去匹配 cfg/models/12/yolo12.yaml 的结构
    model = YOLO('/home/Cug-Rs02/20251011/IRSTD-YOLO/ultralytics/cfg/models/12/yolo12.yaml')

    # 2. 开始训练
    results = model.train(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',         # 你的 VEDAI 数据集配置文件
        epochs=100,                # 训练轮数
        imgsz=512,                 # 图像尺寸 512x512
        batch=8,                   # 批次大小
        device='0',                # 显卡ID
        workers=4,                 # 数据加载线程数
        project='runs/M3FD/train',# 训练结果保存的主目录
        name='yolov12',        # 这一次实验的名称
        save=True,                 # 保存最佳权重
        val=True,                  # 开启验证
        amp=False                  # 【核心修复2】关闭混合精度及自动检查，避免后台静默下载检测文件报错
    )

if __name__ == '__main__':
    main()