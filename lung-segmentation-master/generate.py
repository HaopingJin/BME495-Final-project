from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from src.data import blend
from src.models import UNet, PretrainedUNet

# 文件路径设置
origin_folder = Path("data/val/images")  # 图像文件夹路径
masks_folder = Path("data/val/masks")  # 掩膜保存文件夹路径
models_folder = Path("models")  # 模型文件夹路径
model_name = "unet-6v.pt"  # 模型文件名

# 创建保存掩膜的文件夹（如果不存在）
masks_folder.mkdir(parents=True, exist_ok=True)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
unet = PretrainedUNet(
    in_channels=1,
    out_channels=2,
    batch_norm=True,
    upscale_mode="bilinear"
)
unet.load_state_dict(torch.load(models_folder / model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval()

# 批量处理文件夹中的所有图像
for origin_filename in origin_folder.glob("*.jpeg"):
    # 读取原始图像
    origin = Image.open(origin_filename).convert("P")

    # 调整图像尺寸
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5  # 归一化

    # 进行推理
    with torch.no_grad():
        origin = torch.stack([origin])  # 扩展维度以适应批处理
        origin = origin.to(device)
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)  # 获取每个像素的类别

        origin = origin[0].to("cpu")
        out = out[0].to("cpu")

    # 将输出掩膜转换为黑白图像
    out = (out > 0.5).float() * 255  # 转换为 0 或 255
    out = out.byte()  # 转为 uint8 类型
    out_image = torchvision.transforms.functional.to_pil_image(out)  # 转换为 PIL 图像
    out_image = out_image.convert("1")  # 转为 1 通道黑白图像

    # 保存生成的掩膜
    mask_filename = masks_folder / (origin_filename.stem + "_mask.png")
    out_image.save(mask_filename)
    print(f"Saved mask for {origin_filename.name} to {mask_filename}")

print("All masks have been generated and saved.")
