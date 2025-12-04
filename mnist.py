import os#打开生成的图像文件
import sys#用于检测操作系统
import argparse#用于解析命令行操作
from typing import Tuple#用于注释函数返回类型


import torch#导入库
import torch.nn as nn#神经网络模块库
import torch.optim as optim#优化器库
import torchvision#计算机视觉工具包
import torchvision.transforms as transforms#图像预处理工具
from torch.utils.data import DataLoader
#从pytorch中导入dataloader类为后续模型训练提供数据

#确保它能在终端中运行
import matplotlib
matplotlib.use('Agg')
#用于可视化
import matplotlib.pyplot as plt

#数据获取
def get_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
#定义函数，默认每次迭代样本数量为64，返回两个元组
    transform = transforms.Compose([
        transforms.ToTensor(),
#将图像转换为pytorch张量，像素值从[0,255]缩放到[0,1]
        transforms.Normalize((0.5,), (0.5,))
#范围转化，将[0,1]转化为[-1,1]
    ])

#下载MNIST数据集
#数据储存路径/区分测试集和训练集/如果有本地没有则自动下载/应用前面定义的数据变换
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#创建数据加载器
#训练集打乱数据顺序便于提高训练效果
#测试集保持数据顺序便于评估
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

##模型构建函数
def build_model() -> nn.Module:
    model = nn.Sequential(
        nn.Flatten(),#将二维转化成一维
        nn.Linear(28 * 28, 256),#输入到隐藏层1
        nn.ReLU(),#激活函数，引入非线性
        nn.Linear(256, 128),#隐藏层1到隐藏层2
        nn.ReLU(),
        nn.Linear(128, 10)#输出层，识别0-9，输出为10个
    )
    return model

#训练函数
#模型/数据加载器/设备/训练轮数/学习率
def train(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float = 1e-3):
    model.to(device)#将模型移到显卡上
    model.train()#开启训练模式
    criterion = nn.CrossEntropyLoss()#损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)#优化器，adam算法，自适应学习率


#epoch为模型完整的学习一遍整个训练数据集
#遍历每一次训练
    for epoch in range(epochs):
        #初始化每个epoch的统计变量
        running_loss = 0.0#累加器
        correct = 0#计数器，模型预测正确的个数
        total = 0#模型总共预测的个数
        #遍历数据批次，将数据移动到指定设备
        #每次循环取出图片和真实标签
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)#将数据转移至显卡中
            optimizer.zero_grad()#清零梯度，避免反向传播时由累加产生错误
            outputs = model(inputs)#向前传播
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#反向传播，计算梯度，减小损失（求导）
            optimizer.step()#根据梯度优化模型参数

            running_loss += loss.item()#获取损失值
            _, predicted = torch.max(outputs, 1)#取出最大值及他的索引，即预测结果
            total += labels.size(0)#用布尔数比较预测和真实样本是否相同
            correct += (predicted == labels).sum().item()#计算预测正确的个数

        avg_loss = running_loss / len(loader)#计算平均损失值
        acc = 100.0 * correct / total if total > 0 else 0.0#计算百分比形式正确率
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%", flush=True)#将结果打印出来

#评估函数
#接受训练好的模型/数据加载器/计算设备，返回浮点数型准确率数值
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.to(device)#移动到显卡上
    model.eval()#关闭dropout层和batchnorm层，不在丢弃神经元，更新参数
    correct = 0#记录预测正确个数
    total = 0#记录样本总数
    with torch.no_grad():#禁用梯度计算，节省空间
        for inputs, labels in loader:#遍历测试集
            inputs, labels = inputs.to(device), labels.to(device)#将数据移动到显卡上
            outputs = model(inputs)#前向传播
            _, predicted = torch.max(outputs, 1)#取得预测结果和类型标签
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.2f}%", flush=True)
    return acc

#可视化函数
#代展示模型/测试集/计算设备
def visualize_predictions(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str = 'mnist_predictions.png', n: int = 6):
    model.to(device)
    model.eval()
    dataiter = iter(loader)#将测试集转换为迭代器
    images, labels = next(dataiter)#从迭代器中取出一批数据
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)#用最新的权重、偏值再次预测结果

#创建可视化图并附有预测结果
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f"P:{predicted[i].cpu().item()}")
        ax.axis('off')

#保存并尝试打开图片
    plt.tight_layout()#自动调整子图间距
    plt.savefig(out_path, dpi=120)
    print(f"Saved predictions to {out_path}", flush=True)#输出提示
    try:
        if sys.platform == 'win32':#自动保存打开图片
            os.startfile(out_path)
    except Exception:
        pass

#参数解析
def parse_args():
    p = argparse.ArgumentParser(description='Simple MNIST example (simplified)')
    p.add_argument('--epochs', type=int, default=1, help='number of training epochs')#训练轮次为一轮
    p.add_argument('--batch-size', type=int, default=64, help='batch size')#批次为64
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')#学习率默认
    p.add_argument('--no-cuda', action='store_true', help='disable CUDA')#用户指定这个数则禁用cuda
    p.add_argument('--save', type=str, default='mnist_predictions.png', help='path to save visualization')#可视化图片保存路径
    return p.parse_args()

#主函数
def main():
    #查看该环境下是否有可用的GPU 
    args = parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}", flush=True)

    trainloader, testloader = get_data(batch_size=args.batch_size)
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}", flush=True)

    model = build_model()

    train(model, trainloader, device, epochs=args.epochs, lr=args.lr)#训练模型
    evaluate(model, testloader, device)#评估模型
    visualize_predictions(model, testloader, device, out_path=args.save)#可视化模型

#程序入口，启动整个项目流程
if __name__ == '__main__':
    main()

