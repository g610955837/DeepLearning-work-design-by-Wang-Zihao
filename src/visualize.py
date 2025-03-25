# src/visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 强制切换工作目录到项目根目录 ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
os.chdir(project_root)
print("当前工作目录:", os.getcwd())

def load_model():
    """加载训练好的模型"""
    model = SimpleCNN()
    model_path = os.path.join(project_root, 'outputs', 'kmnist_cnn.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到，请先运行 train.py！")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_training_metrics():
    """生成训练曲线图（原有功能）"""
    metrics_path = os.path.join(project_root, 'outputs', 'training_metrics.pth')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("请先运行 train.py 生成训练指标数据！")
    metrics = torch.load(metrics_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accs'], label='Training Accuracy')
    plt.plot(metrics['test_accs'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(project_root, 'outputs', 'training_metrics.png'))
    print("训练曲线已保存至 outputs/training_metrics.png")

def plot_confusion_matrix():
    """新增：生成混淆矩阵"""
    # 加载模型和测试集
    model = load_model()
    test_data = datasets.KMNIST(
        root=os.path.join(project_root, 'data'),
        train=False,
        download=True,
        transform=transforms.ToTensor()  # 确保与训练时相同的预处理
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # 收集预测结果和真实标签
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(project_root, 'outputs', 'confusion_matrix.png'))
    plt.close()
    print("混淆矩阵已保存至 outputs/confusion_matrix.png")

if __name__ == "__main__":
    plot_training_metrics()  # 生成训练曲线
    plot_confusion_matrix()   # 新增：生成混淆矩阵