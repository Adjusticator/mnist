import os
import sys
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Use a non-GUI backend by default so script works in terminals and CI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def build_model() -> nn.Module:
    # Simple feed-forward network
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model


def train(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float = 1e-3):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%", flush=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.2f}%", flush=True)
    return acc


def visualize_predictions(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str = 'mnist_predictions.png', n: int = 6):
    model.to(device)
    model.eval()
    dataiter = iter(loader)
    images, labels = next(dataiter)
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f"P:{predicted[i].cpu().item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved predictions to {out_path}", flush=True)
    # On Windows, try to open the image automatically
    try:
        if sys.platform == 'win32':
            os.startfile(out_path)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description='Simple MNIST example (simplified)')
    p.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    p.add_argument('--batch-size', type=int, default=64, help='batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    p.add_argument('--save', type=str, default='mnist_predictions.png', help='path to save visualization')
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}", flush=True)

    trainloader, testloader = get_data(batch_size=args.batch_size)
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}", flush=True)

    model = build_model()

    train(model, trainloader, device, epochs=args.epochs, lr=args.lr)
    evaluate(model, testloader, device)
    visualize_predictions(model, testloader, device, out_path=args.save)


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
print("1. 导入 torch 库成功", flush=True)
sys.stdout.flush()

import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端，避免窗口问题
import matplotlib.pyplot as plt
print("2. 导入 matplotlib 成功", flush=True)
sys.stdout.flush()

# 数据预处理：将图像转换为Tensor，并进行归一化处理
print("3. 准备数据预处理...", flush=True)
sys.stdout.flush()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 图像归一化到[-1, 1]之间
])

# 下载训练和测试数据集
print("4. 正在下载 MNIST 数据集...", flush=True)
sys.stdout.flush()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(f"5. 训练集大小: {len(trainset)}, 测试集大小: {len(testset)}", flush=True)
sys.stdout.flush()

# 使用DataLoader进行数据批处理
print("6. 创建 DataLoader...", flush=True)
sys.stdout.flush()
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
print(f"7. 训练批次数: {len(trainloader)}, 测试批次数: {len(testloader)}", flush=True)
sys.stdout.flush()

# 构建神经网络模型
print("8. 构建模型...", flush=True)
sys.stdout.flush()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层：输入784维，输出256维
        self.fc1 = nn.Linear(28*28, 256)
        # 定义第二层：256维，输出128维
        self.fc2 = nn.Linear(256, 128)
        # 定义输出层：128维，输出10维（对应10个数字类别）
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 展平输入图片（28x28 -> 784）
        x = x.view(-1, 28*28)
        # 前向传播：使用ReLU激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层，不需要激活函数
        return x

# 创建模型实例
print("9. 创建模型实例...", flush=True)
sys.stdout.flush()
model = Net()

# 检查CUDA是否可用，使用GPU训练
print("10. 检查 CUDA...", flush=True)
sys.stdout.flush()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    使用设备: {device}", flush=True)
sys.stdout.flush()
model.to(device)

# 设置训练参数
print("11. 设置训练参数...", flush=True)
sys.stdout.flush()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001

# 训练过程
print("12. 开始训练...", flush=True)
sys.stdout.flush()
num_epochs = 1  # 只训练1个epoch以快速测试

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清空优化器的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印每个epoch的训练信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# 在测试集上进行评估
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 在评估时不计算梯度
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 可视化部分测试图像和预测结果
print("\n生成可视化图像...")
dataiter = iter(testloader)
images, labels = next(dataiter)

# 获取模型预测
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 绘制前6个图像并保存
fig, axes = plt.subplots(1, 6, figsize=(10, 5))
for i in range(6):
    ax = axes[i]
    ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
    ax.set_title(f'Pred: {predicted[i].cpu().item()}')
    ax.axis('off')

# 保存图像并显示
plt.savefig('mnist_predictions.png', dpi=100, bbox_inches='tight')
print("预测图像已保存为 mnist_predictions.png", flush=True)
sys.stdout.flush()
import os
import subprocess
# 在 Windows 上自动打开图像
if sys.platform == 'win32':
    os.startfile('mnist_predictions.png')
plt.close()
