
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SimpleCNN  


TEST_IMAGE_DIR = r"D:\kmnist-project\test pic"          # 原始图片目录
PROCESSED_IMAGE_DIR = r"D:\kmnist-project\outputs\processed_pic"  # 预处理后图像保存目录
MODEL_PATH = r"D:\kmnist-project\outputs\kmnist_cnn.pth"  # 模型路

os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

# 预处理
def preprocess_and_save(image_path, save_dir):

    try:
        # 灰度
        image = Image.open(image_path).convert('L')
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),      # 调整尺寸
            transforms.ToTensor(),            # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
        ])
        image_tensor = transform(image)
        
        # 保存预处理后的图像
        processed_image = transforms.functional.to_pil_image(image_tensor.squeeze())
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        processed_image.save(save_path)
        print(f"预处理图像已保存至: {save_path}")
        
        return image_tensor.unsqueeze(0)  # 添加批次维度 [1, 1, 28, 28]
    except Exception as e:
        print(f"预处理失败: {image_path}，错误: {e}")
        return None

def predict_char(image_tensor, model):

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

def main():
    
    model = SimpleCNN()
    if not os.path.exists(MODEL_PATH):
        print(f"模型这个玩意儿 {MODEL_PATH} 有点问题")
        return
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("模型加载成功。")
    
    # 类别与平假名对应表
    class_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
    
    # 遍历测试图片
    for filename in os.listdir(TEST_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(TEST_IMAGE_DIR, filename)
            # 预处理图像
            input_tensor = preprocess_and_save(image_path, PROCESSED_IMAGE_DIR)
            if input_tensor is None:
                continue
            # 预测
            pred_class = predict_char(input_tensor, model)
            print(f"图片 {filename} 预测结果 → 类别: {pred_class}, 平假名: {class_names[pred_class]}")

if __name__ == "__main__":
    main()