import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/datasets/train/exp-DRG/weights/best.pt')
    model.val(data='/home/Cug-Rs02/20251011/IRSTD-YOLO/datasets/config.yaml',
              # data='./datasets/InfraredUAV/infrareduav.yaml',
              split='test',
              imgsz=512,
              batch=8,
              # mosaic=0.0,  # 【核心】关闭马赛克
              # mixup=0.0,
              iou=0.5,
              save_json=True, 
              project='runs/datasets/test',
              name='exp-DRG',
              )