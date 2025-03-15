from ultralytics import YOLO
import os
import cv2
import numpy as np

# 加载训练好的 YOLOv5/v9 模型
model = YOLO('runs/train/yolov9s_40epochs-maskdata2/weights/best.pt')  # 请替换为你训练后模型的路径

# 设置测试图像目录和真实标签目录
test_images_dir = 'maskdata/test/images'
ground_truth_folder = 'maskdata/test/labels'


# 计算分类准确率
def calculate_accuracy(predictions, ground_truth_folder):
    correct = 0
    total = 0
    for image_name, pred in predictions.items():
        # 获取真实标签文件路径
        true_label_file = os.path.join(ground_truth_folder, image_name.split('.')[0] + '.txt')

        if not os.path.exists(true_label_file):
            continue

        # 读取真实标签
        with open(true_label_file, 'r') as f_true:
            true_classes = [int(line.split()[0]) for line in f_true.readlines()]

        # 获取预测类别
        predicted_class = pred['class']  # 选择预测类别中概率最大的一项
        if predicted_class in true_classes:
            correct += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy


# 进行推理并获取预测结果
predictions = {}  # 保存每个图像的预测结果
for image_name in os.listdir(test_images_dir):
    if image_name.endswith(('.jpg', '.png', 'jpeg')):  # 检查图像文件格式
        image_path = os.path.join(test_images_dir, image_name)

        # 读取图像
        img = cv2.imread(image_path)

        # 进行推理
        results = model(img)

        # 获取类别概率最大的预测
        # results.xywh[0][:, -1] 表示所有预测的类别索引，results.conf[0] 表示所有目标的置信度
        pred_class = results[0].boxes.cls.cpu().numpy()  # 获取预测的类别
        pred_conf = results[0].boxes.conf.cpu().numpy()  # 获取每个目标的置信度

        if len(pred_class) > 0:  # 如果检测到目标
            # 选择置信度最高的目标
            max_conf_idx = np.argmax(pred_conf)
            predicted_class = int(pred_class[max_conf_idx])  # 获取最大置信度对应的类别

            # 保存预测结果
            predictions[image_name] = {'class': predicted_class, 'confidence': pred_conf[max_conf_idx]}

# 计算准确率
accuracy = calculate_accuracy(predictions, ground_truth_folder)
print(f'分类准确率: {accuracy:.2f}%')
