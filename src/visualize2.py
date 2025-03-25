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
    """生成训练曲线图"""
    metrics_path = os.path.join(project_root, 'outputs', 'training_metrics.pth')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("请先运行 train.py 生成训练指标数据！")
    metrics = torch.load(metrics_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accs'], label='Training Accuracy', color='green')
    plt.plot(metrics['test_accs'], label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(project_root, 'outputs', 'training_metrics.png'))
    plt.close()
    print("训练曲线已保存至 outputs/training_metrics.png")

def plot_confusion_matrix():
    """生成混淆矩阵"""
    model = load_model()
    test_data = datasets.KMNIST(
        root=os.path.join(project_root, 'data'),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(project_root, 'outputs', 'confusion_matrix.png'))
    plt.close()
    print("混淆矩阵已保存至 outputs/confusion_matrix.png")

def plot_lr_comparison():
    """对比不同学习率"""
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    lr_results = {'loss': {}, 'test_acc': {}}
    
    for lr in lr_list:
        metrics_path = os.path.join(project_root, 'outputs', f'training_metrics_lr_{lr}.pth')
        if not os.path.exists(metrics_path):
            print(f"警告: 学习率 {lr} 的实验数据未找到，跳过")
            continue
        metrics = torch.load(metrics_path)
        lr_results['loss'][lr] = metrics['train_losses']
        lr_results['test_acc'][lr] = metrics['test_accs']
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for lr in lr_list:
        if lr in lr_results['loss']:
            plt.plot(lr_results['loss'][lr], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('不同学习率的训练损失对比')
    
    plt.subplot(1, 2, 2)
    for lr in lr_list:
        if lr in lr_results['test_acc']:
            plt.plot(lr_results['test_acc'][lr], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('不同学习率的测试准确率对比')
    plt.savefig(os.path.join(project_root, 'outputs', 'lr_comparison.png'))
    plt.close()
    print("学习率对比图已保存至 outputs/lr_comparison.png")

def plot_test_samples(num_samples=100):
    """展示测试集前100样本的预测结果"""
    model = load_model()
    test_data = datasets.KMNIST(
        root=os.path.join(project_root, 'data'),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_samples = test_data.data[:num_samples].unsqueeze(1).float() / 255.0  # 转换为张量并归一化
    true_labels = test_data.targets[:num_samples].numpy()
    
    with torch.no_grad():
        outputs = model(test_samples)
        pred_labels = torch.argmax(outputs, dim=1).numpy()
    
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        plt.subplot(10, 10, i+1)
        plt.imshow(test_samples[i].squeeze(), cmap='gray')
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        plt.title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", color=color, fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'outputs', 'test_predictions.png'))
    plt.close()
    print("测试集前100样本预测结果已保存至 outputs/test_predictions.png")

if __name__ == "__main__":
    plot_training_metrics()    # 训练曲线
    plot_confusion_matrix()    # 混淆矩阵
    plot_lr_comparison()       # 学习率对比
    plot_test_samples()        # 测试样本预测