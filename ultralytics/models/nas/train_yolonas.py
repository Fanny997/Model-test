import sys
import os

# 【核心修复1】强制将当前脚本所在的目录加入系统路径，确保读取你本地魔改过的 ultralytics 代码
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import warnings
warnings.filterwarnings('ignore')

# 导入 NAS 模型类
from ultralytics import NAS

def main():
    # 1. 加载 YOLO-NAS 模型
    # YOLO-NAS 提供了不同大小的预训练模型：'yolo_nas_s.pt' (小), 'yolo_nas_m.pt' (中), 'yolo_nas_l.pt' (大)
    # 推荐从 s 或 m 开始跑通
    model = NAS('yolo_nas_s.pt')

    # 2. 开始训练
    results = model.train(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',          # 刚才建好的 M3FD 数据集配置文件
        epochs=100,                # 训练轮数
        imgsz=512,                 # 你的图像尺寸是 512x512
        batch=8,                   # 批次大小，显存不够可降为 4
        device='0',                # 你的显卡ID，如果有两张卡可以写 '0,1'
        workers=4,                 # 数据加载线程数
        project='runs/M3FD/train', # 训练结果保存的主目录
        name='yolonas',        # 这一次实验的名称
        save=True,                 # 保存最佳权重
        val=True,                  # 每个 epoch 结束后做验证
        amp=False                  # 【核心修复2】关闭混合精度及自动检查，避免网络问题导致下载预训练权重报错
    )

if __name__ == '__main__':
    main()