# src/model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输入通道1（灰度图），输出通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 计算尺寸：28x28 → 池化后14x14 → 卷积后12x12 → 池化后6x6 → 卷积后4x4 → 池化后2x2
        self.fc2 = nn.Linear(128, 10)           # 输出10类（0-9）
        # 正则化
        self.dropout = nn.Dropout(0.5)          # 防止过拟合

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 输出尺寸：32通道 x 14x14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # 输出尺寸：64通道 x 5x5
        # 展平
        x = x.view(x.size(0), -1)                   # 转换为 [batch_size, 64*5*5]
        # 全连接
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x