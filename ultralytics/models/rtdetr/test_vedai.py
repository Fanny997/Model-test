from ultralytics import RTDETR

# 加载你刚才训练好的最佳权重
model = RTDETR('/home/Cug-Rs02/20251011/IRSTD-YOLO/runs/M3FD/train/vedai_rtdetr2/weights/best.pt')

# 在测试集上进行验证
metrics = model.val(
    data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',
    split='test',  # 指定评估 test 集
    imgsz=512,
    device='0',
    project='runs/M3FD/test',
    name='exp-detr2'
)