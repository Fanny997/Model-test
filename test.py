import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 【注意】：依然指向你训练出来的最好的那个权重
    weight_path = '/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/M3FD/train/Ours_M3FD_DRG/weights/best.pt'
    model = YOLO(weight_path)

    model.val(
        data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',
        split='test',  # 必须是 test 集，代表模型从未见过的数据
        imgsz=512,
        batch=8,
        iou=0.5,
        conf=0.001,  # 【学术建议】设置极低的置信度，能测出最真实的 PR 曲线和极限 mAP 值
        save_json=True,
        project='runs/Newruns/M3FD/test',
        name='DRG-DSA-GCAI',
    )