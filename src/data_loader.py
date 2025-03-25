# src/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_kmnist(batch_size=32):
    # 打印路径信息（调试用）
    print("当前工作目录:", os.getcwd())
    print("数据存储路径:", os.path.abspath('data'))
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomRotation(10)
    ])
    
    # 加载数据集（路径调整为项目内的data目录）
    train_data = datasets.KMNIST(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_data = datasets.KMNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_kmnist()
    print("数据加载成功！")