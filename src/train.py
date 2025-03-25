# src/train.py
import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from data_loader import load_kmnist
from model import SimpleCNN

# --- 强制切换工作目录到项目根目录 ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(script_path))
os.chdir(project_root)
print("当前工作目录:", os.getcwd())

def train_model(epochs=10, lr=0.001, batch_size=32):
    # 加载数据
    train_loader, test_loader = load_kmnist(batch_size)
    
    # 初始化模型、损失函数、优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_accs = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练集指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 测试集验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = test_correct / test_total
        test_accs.append(test_acc)
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型和训练指标
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(output_dir, 'kmnist_cnn.pth')
    torch.save(model.state_dict(), model_path)
    
    # 保存训练指标
    metrics_path = os.path.join(output_dir, f'training_metrics_lr_{lr}_bs_{batch_size}.pth')
    torch.save({
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }, metrics_path)
    
    return train_losses, train_accs, test_accs

if __name__ == "__main__":
    # 对比不同学习率
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    for lr in learning_rates:
        print(f"\n=== 学习率实验: lr={lr} ===")
        train_model(epochs=10, lr=lr, batch_size=32)
    
    # 对比不同批量大小
    batch_sizes = [8, 16, 32, 64, 128]
    for bs in batch_sizes:
        print(f"\n=== 批量大小实验: batch_size={bs} ===")
        train_model(epochs=10, lr=0.001, batch_size=bs)