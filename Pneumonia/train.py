from ultralytics import YOLO


def train(model_name, data_yaml, epochs, batch, project):
    name = model_name.split('.')[0] + '_' + str(epochs) + 'epochs' +'-maskdata'  # 模型名称
    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, batch=batch, project=project, name=name)  # 训练参数，如数据集配置文件，训练轮次等


if __name__ == '__main__':
    train('yolov9s.yaml', 'data_mask.yaml', 40, 32, 'runs/train')
