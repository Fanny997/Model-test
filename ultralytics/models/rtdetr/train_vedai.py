import sys
import os

# 【核心修复】强制将当前脚本所在的目录加入系统路径，确保绝对能找到本地的 ultralytics 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

def main():
    # 1. 加载模型
    # 这里我们使用预训练的 rtdetr-l.pt 作为初始化，这比从头开始训练收敛更快
    # 如果你想使用更大的模型，可以换成 'rtdetr-x.pt' 或者 'rtdetr-resnet50.pt'
    model = RTDETR('/home/Cug-Rs02/20251011/IRSTD-YOLO/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')

    # 2. 开始训练
    results = model.train(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',     # 刚才建好的数据集配置文件
        epochs=100,            # 训练轮数，你可以根据情况修改，比如 200 或 300
        imgsz=512,             # 你的图像尺寸是 512x512
        batch=8,               # 批次大小，如果显存不够（OOM），请改成 4 或 2
        device='0',            # 指定使用第0张显卡（CUDA 12.4 支持完美运行）
        workers=8,             # 数据加载的线程数，根据你 CPU 的核心数调整
        project='runs/M3FD/train',  # 训练结果保存的主目录
        name='vedai_rtdetr',   # 这一次实验的名称
        save=True,             # 保存训练出来的权重
        val=True,               # 每个 epoch 结束后做验证
        amp=False
    )

if __name__ == '__main__':
    main()