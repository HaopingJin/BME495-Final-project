import os

# 文件夹路径
train_dir = 'maskdata/train/images'
test_dir = 'maskdata/test/images'
val_dir = 'maskdata/val/images'

# 用于存储YOLO格式标签的文件夹路径（与 images 同级）
train_label_dir = 'maskdata/train/labels'
test_label_dir = 'maskdata/test//labels'
val_label_dir = 'maskdata/val/labels'

# 确保每个数据集的标签文件夹存在
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 目标标签生成函数
def generate_label(image_name):
    if 'virus' in image_name:
        return 1
    elif 'bacteria' in image_name:
        return 2
    else:
        return 0

# 生成YOLO格式的标签
def generate_yolo_label(image_name, label):
    # 获取图像的宽度和高度 (假设你有图像尺寸信息，如果没有，可以暂时用图像大小替代)
    # 这里假设图片是一个固定尺寸（例如 1024x1024），你也可以通过读取图像获取尺寸。
    width = 1024
    height = 1024

    # 这里假设框的坐标为图像的整个区域
    # x_center, y_center, width, height
    x_center = 0.5
    y_center = 0.5
    obj_width = 1.0
    obj_height = 1.0

    # 返回YOLO格式标签
    return f"{label} {x_center} {y_center} {obj_width} {obj_height}"

# 遍历并生成标签文件
def generate_labels_for_folder(folder_path, label_dir):
    for image_name in os.listdir(folder_path):
        if image_name.endswith(('.jpg', '.png', 'jpeg')):  # 检查文件类型
            # 生成标签
            label = generate_label(image_name)

            # 生成YOLO格式标签
            yolo_label = generate_yolo_label(image_name, label)

            # 创建对应的标签文件，文件名与图像名相同，但扩展名为 .txt
            label_file_path = os.path.join(label_dir, image_name.split('.')[0] + '.txt')
            with open(label_file_path, 'w') as f:
                f.write(yolo_label)

# 生成各数据集的标签文件
generate_labels_for_folder(train_dir, train_label_dir)
generate_labels_for_folder(test_dir, test_label_dir)
generate_labels_for_folder(val_dir, val_label_dir)

print("YOLO标签文件已生成！")
