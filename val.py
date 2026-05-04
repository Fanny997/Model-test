import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 【注意】：这里的权重路径，必须改成你 train.py 刚跑完的 weights/best.pt 路径
    weight_path = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/M3FD/train/Ours_M3FD_DRG/weights/best.pt'
    model = YOLO(weight_path)

    model.val(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',
        split='val',
        imgsz=512,
        batch=8,
        iou=0.5,
        save_json=True,
        project='runs/Newruns/M3FD/val',
        name='DRG-DSA-GCAI',
    )