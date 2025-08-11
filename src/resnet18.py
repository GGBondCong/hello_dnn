import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os
from PIL import Image
from invert import InvertIfBright

# ---------- 强制多线程 ----------
torch.set_num_threads(8)          # 按我 CPU 的物理核心数来调
# --------------------------------
device = torch.device("cpu")      # 强制 CPU

# -------------------- 超参数 --------------------
BATCH_SIZE = 64
EPOCHS = 5 
LR = 1e-3                         #学习率
SAVE_PATH = "model_Mnist.pth"
PREDICT_FOLDER = os.path.join('.', 'assets', 'imgs', 'my_digits')


# 数据：直接 64×64 灰度
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # 关键：大幅缩小输入
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_set = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
test_set  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------- 轻量模型：ResNet18 ----------
# model = models.resnet18(weights=None)
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 改 1 通道
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
# ----------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train(epoch):
    model.train()
    epoch_loss = 0.0        # 用来累加整个 epoch 的 loss
    correct = 0             # 用来累加整个 epoch 的正确样本数
    total = 0               # 用来累加整个 epoch 的总样本数

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            print(f"\rEpoch {epoch+1} | batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")

    # 整个 epoch 结束后一次性打印平均 loss 和准确率
    avg_loss = epoch_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"\n[{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")

def test(epoch):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    print(f"[{epoch+1}/{EPOCHS}] Test Acc: {acc:.2f}%")

if not os.path.exists(SAVE_PATH):
    print("开始训练，使用 ResNet18 + 64×64 输入，多线程加速...")
    for i in range(5):
        train(i)
        test(i)

    torch.save(model.state_dict(), SAVE_PATH)
    print("训练完成，权重已保存为", SAVE_PATH)


# ---------- 预测 ----------
transform_pred = transforms.Compose([
    transforms.Resize((64, 64)),
    InvertIfBright(),
    transforms.Grayscale(num_output_channels=1),
    transforms.GaussianBlur(3),
    transforms.RandomAdjustSharpness(sharpness_factor=3.0, p=0.9),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

from torchvision.utils import save_image

def predict_folder():
    if not os.path.isdir(PREDICT_FOLDER):
        os.makedirs(PREDICT_FOLDER)
        print(f"已创建识别文件夹：{PREDICT_FOLDER}")
        return
    imgs = [f for f in os.listdir(PREDICT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        print("文件夹里没有图片！")
        return
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    for name in sorted(imgs):
        path = os.path.join(PREDICT_FOLDER, name)
        try:
            img = Image.open(path)
            tensor = transform_pred(img).unsqueeze(0)
            # save_image(tensor, f"./debug_imgss/debug_{name}")
            pred = model(tensor).argmax(dim=1).item()
            print(f"{name} -> 预测数字: {pred}")
        except Exception as e:
            print(f"{name} 读取失败: {e}")

predict_folder()
