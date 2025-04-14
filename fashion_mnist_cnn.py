# 导入所需要的包
import torch                       # PyTorch主包，提供张量操作、计算图等功能
import torch.nn as nn              # 定义神经网络模块
import torch.optim as optim        # 定义优化器
from torchvision import datasets, transforms  # 处理数据集和图像转换
from torch.utils.data import DataLoader, random_split   # 用于高效加载数据和管理数据集划分
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端，适合保存图片，不显示窗口
import matplotlib.pyplot as plt    # 用于绘制图表
import numpy as np                 # 用于数值计算
from sklearn.metrics import confusion_matrix  # 用于计算混淆矩阵
import pandas as pd                # 用于绘制表格

# ---------------------------------
# 1. 设置设备：优先使用CUDA，否则使用CPU
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备：", device)

# -------------------------------------
# 2. 数据预处理及数据加载Fashion MNIST数据集
# -------------------------------------
# 定义数据预处理：图像转换为Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 Fashion MNIST 数据集，使用本地文件
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 将训练数据集拆分为训练集和验证集，以便在训练过程中监控模型性能
train_size = int(0.8 * len(train_dataset))  # 计算训练子集大小，占原始训练集的 80%
val_size = len(train_dataset) - train_size  # 计算验证集大小，占剩余的 20%
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])  # 随机分割数据集为训练集和验证集

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------------
# 3. 定义神经网络模型（卷积神经网络）
# ----------------------------
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 4. 定义损失函数与优化器
# ----------------------------
# 初始化模型、损失函数和优化器
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 5. 模型训练与评估
# ----------------------------
# 初始化列表，用于记录损失和准确率
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0  # 在循环开始时初始化
    total_train = 0  # 在循环开始时初始化
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算验证准确率
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

# ----------------------------
# 6. 绘制训练和验证损失曲线
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')  # 保存为文件
plt.close()  # 关闭当前图形，避免重叠

# ----------------------------
# 7. 绘制训练和验证准确率曲线
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')  # 保存为文件
plt.close()

# ----------------------------
# 8. 测试集评估
# ----------------------------
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())  # 保存预测结果
        all_labels.extend(labels.cpu().numpy())    # 保存真实标签

test_accuracy = correct / total
print(f"测试准确率: {100 * test_accuracy:.2f}%")

# ----------------------------
# 9. 计算并绘制类别级准确率表
# ----------------------------
# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
# 计算每个类别的准确率
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Fashion MNIST类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印类别级准确率
print("\n类别级准确率表:")
print("Class\t\tAccuracy (%)")
for i, acc in enumerate(class_accuracy):
    print(f"{class_names[i]:<15}{acc * 100:.2f}")

# 使用Matplotlib绘制表格
df = pd.DataFrame({
    'Class': class_names,
    'Accuracy (%)': [acc * 100 for acc in class_accuracy]
})
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
plt.title('Class-wise Accuracy on Test Set')
plt.savefig('class_accuracy_table.png')  # 保存为文件
plt.close()

print("图表已保存为 'loss_curve.png', 'accuracy_curve.png', 'class_accuracy_table.png'")