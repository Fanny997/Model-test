import os
import json
import cv2
import shutil

# 【需要您修改】请根据您的 M3FD/config.yaml 中的类别顺序填写
CLASSES = ['Bus', 'car', 'Lamp', 'Motorcycle', 'people', 'Truck']


def convert_yolo_to_coco(yolo_images_dir, yolo_labels_dir, output_json, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }

    annotation_id = 1
    image_files = [f for f in os.listdir(yolo_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_id, image_name in enumerate(image_files):
        img_path = os.path.join(yolo_images_dir, image_name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # 将图片复制到 DETR 期望的文件夹中
        shutil.copy(img_path, os.path.join(output_image_dir, image_name))

        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        label_path = os.path.join(yolo_labels_dir, image_name.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())

                    # YOLO (归一化的 cx, cy, w, h) -> COCO (绝对像素值 x_min, y_min, w, h)
                    w_pixel, h_pixel = w * width, h * height
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height

                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, w_pixel, h_pixel],
                        "area": w_pixel * h_pixel,
                        "iscrowd": 0
                    })
                    annotation_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"✅ 转换完成，已保存至 {output_json}")


if __name__ == '__main__':
    # 假设您的原始 YOLO 数据集路径如下，请根据实际情况修改
    m3fd_root = '/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD/'

    # 原版 DETR 期望的数据集结构
    detr_data_root = '/home/Cug-Rs02/20251011/IRSTD-YOLO/M3FD_COCO'
    os.makedirs(os.path.join(detr_data_root, 'annotations'), exist_ok=True)

    # 转换训练集
    convert_yolo_to_coco(
        yolo_images_dir=os.path.join(m3fd_root, 'images/train'),
        yolo_labels_dir=os.path.join(m3fd_root, 'labels/train'),
        output_json=os.path.join(detr_data_root, 'annotations/instances_train2017.json'),
        output_image_dir=os.path.join(detr_data_root, 'train2017')  # DETR 源码默认硬编码了这个文件夹名
    )

    # 转换验证集
    convert_yolo_to_coco(
        yolo_images_dir=os.path.join(m3fd_root, 'images/val'),
        yolo_labels_dir=os.path.join(m3fd_root, 'labels/val'),
        output_json=os.path.join(detr_data_root, 'annotations/instances_val2017.json'),
        output_image_dir=os.path.join(detr_data_root, 'val2017')
    )