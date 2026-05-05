import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/Cug-Rs02/20251011/IRSTD-YOLO/ultralytics/cfg/models/11/yolo11-C3k2_EFE-IRSTE.yaml')
    # model.load('runs/train/exp2/weights/last.pt') # loading pretrain weights
    # print(model)  # 打印模型结构，看各层输入输出通道数
    model.train(
                data='/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/config.yaml',
                # data='./datasets/AntiUAV310/antiuav310.yaml',
                cache=False,
                imgsz=512,
                epochs=300,
                # epochs=50,
                batch=8,
                workers=4,
                device='1',
                optimizer='SGD',
                patience=50,
                # resume=True,
                amp=False, # close amp
                project='runs/Newruns/M3FD/train',
                name='DRG-DSA-ADSF-GCAI',
                )