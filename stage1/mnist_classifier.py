import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# 1. 数据准备
# ============================================================

# 定义变换：把图片转成张量，并归一化
transform = transforms.Compose([
    transforms.ToTensor(),                    # 转成 [0, 1] 的张量
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
])

# 下载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data',          # 数据保存路径
    train=True,             # 训练集
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)       # 提示：和 train_dataset 类似，但 train=False

# DataLoader：批处理
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)         # 提示：test 不需要 shuffle

# ============================================================
# 2. 定义模型
# ============================================================

model = nn.Sequential(
    nn.Flatten(),           # 把 (1, 28, 28) 展平成 (784,)
    nn.Linear(784, 128),    # 输入 784，隐藏层 128
    nn.ReLU(),
    nn.Linear(128, 10)      # 输出 10 类
)

# ============================================================
# 3. 损失函数和优化器
# ============================================================

criterion = nn.CrossEntropyLoss()             # 提示：分类用 CrossEntropyLoss
optimizer = optim.SGD(model.parameters(), lr=0.01)            # 提示：optim.SGD 或 optim.Adam

# ============================================================
# 4. 训练
# ============================================================

epochs = 5

for epoch in range(epochs):
    model.train()           # 切换到训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # 每个 epoch 结束后，在测试集上评估
    model.eval()            # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():   # 评估时不需要梯度
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)     # 取最大概率的类别
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch}: 测试集准确率 = {accuracy:.4f}")