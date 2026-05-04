import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # =========================================================================
    # 1. 选择你要训练的模型配置文件 (这里以 M3FD 最优的 DRG 版本为例)
    # 如果你要跑 VEDAI 数据集，请换成: 'ultralytics/cfg/models/11/yolo11-vedai-gcai.yaml'
    # =========================================================================
    yaml_path = '/home/Cug-Rs02/20251011/IRSTD-YOLO/ultralytics/cfg/models/11/yolo11-C3k2_EFE-IRSTE.yaml'
    model = YOLO(yaml_path)

    # 2. 开始训练
    model.train(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',
        cache=False,
        imgsz=512,
        epochs=300,
        batch=8,

        # ================== 【学术极度核心：关闭破坏性增强】 ==================
        mosaic=0.0,  # 彻底关闭马赛克拼接，保护小目标不被截断
        mixup=0.0,  # 彻底关闭混合，保护小目标能量不被稀释
        copy_paste=0.0,  # 关闭复制粘贴
        # =====================================================================

        workers=4,
        device='1',  # 你指定的 GPU 卡号
        optimizer='SGD',
        patience=50,
        amp=False,  # 关闭混合精度，防止红外微弱信号在 FP16 下下溢出(Underflow)

        project='runs/Newruns/M3FD/train',
        name='DRG-DSA-GCAI',  # 建议每次跑改个名字，比如 Ours_VEDAI_GCAI
    )