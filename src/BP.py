# bp_mnist_offline.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

# -------------------- 超参数 --------------------
INPUT_SIZE   = 28 * 28   # 784
H1_SIZE      = 128       # 第 1 隐藏层
H2_SIZE      = 64        # 第 2 隐藏层
OUTPUT_SIZE  = 10        # 输出类别
LR           = 0.05      # 学习率
EPOCHS       = 30
BATCH_SIZE   = 128
MODEL_PATH   = 'bp_weights.npz'          # 权重保存路径
IMG_DIR      = os.path.join(os.path.expanduser('~'),
                            'Desktop', 'test_imgs')  # 桌面待识别图片目录
ALPHA        = 0.01      # Elu alpha

# ------------------------------------------------
# 1. 数据读取：torchvision 自动下载 MNIST
# ------------------------------------------------
def load_mnist():
    trans = transforms.Compose([
        transforms.ToTensor(),                # 0~1
        transforms.Normalize((0.5,), (0.5,))  # -1~1
    ])
    train_ds = datasets.MNIST('./mnist_data',
                              train=True,  download=True, transform=trans)
    test_ds  = datasets.MNIST('./mnist_data',
                              train=False, download=True, transform=trans)

    # 转 numpy
    X_train = train_ds.data.numpy().reshape(-1, INPUT_SIZE).astype(np.float32) / 255.0
    X_test  = test_ds.data.numpy().reshape(-1, INPUT_SIZE).astype(np.float32) / 255.0
    # 再次归一到 [-1,1]
    X_train = (X_train - 0.5) / 0.5
    X_test  = (X_test  - 0.5) / 0.5

    y_train = train_ds.targets.numpy()
    y_test  = test_ds.targets.numpy()
    return (X_train, y_train), (X_test, y_test)
    

# ------------------------------------------------
# 2. 3 层 BP 网络
# ------------------------------------------------
class BPNet:
    def __init__(self):
        # Xavier 初始化
        self.W1 = np.random.randn(INPUT_SIZE, H1_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros((1, H1_SIZE))
        self.W2 = np.random.randn(H1_SIZE, H2_SIZE) * np.sqrt(2.0 / H1_SIZE)
        self.b2 = np.zeros((1, H2_SIZE))
        self.W3 = np.random.randn(H2_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / H2_SIZE)
        self.b3 = np.zeros((1, OUTPUT_SIZE))

    # 激活 / 导数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def swish(self, x):
        return x * self.sigmoid(x)

    def d_swish(self, x):
        s = self.sigmoid(x)
        return s + x * s * (1 - s)   # 链式求导公式
    def relu(self, x):
        return np.maximum(0, x)
    def d_relu(self, x):
        return (x > 0).astype(float)
    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    # 前向
    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        # self.a1 = self.relu(self.z1)
        self.a1 = self.swish(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        # self.a2 = self.relu(self.z2)
        self.a2 = self.swish(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    # 反向
    def backward(self, y_true):
        m = y_true.shape[0]
        dz3 = self.a3 - y_true
        dW3 = (self.a2.T @ dz3) / m 
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = dz3 @ self.W3.T
        # dz2 = da2 * self.d_relu(self.z2)
        dz2 = da2 * self.d_swish(self.z2)
        dW2 = (self.a1.T @ dz2) / m 
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        # dz1 = da1 * self.d_relu(self.z1)
        dz1 = da1 * self.d_swish(self.z1)
        dW1 = (self.X.T @ dz1) / m 
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新
        self.W3 -= LR * dW3; self.b3 -= LR * db3
        self.W2 -= LR * dW2; self.b2 -= LR * db2
        self.W1 -= LR * dW1; self.b1 -= LR * db1

    # 交叉熵损失
    def loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_like = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-12)
        return np.sum(log_like) / m

    # 预测类别
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    # 权重保存 / 加载
    def save(self, path=MODEL_PATH):
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3)
        print('权重已保存到', path)

    def load(self, path=MODEL_PATH):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        print('权重已从', path, '加载')

# ------------------------------------------------
# 3. 训练
# ------------------------------------------------
def train():
    (X_train, y_train), (X_test, y_test) = load_mnist()
    y_train_oh = np.eye(OUTPUT_SIZE)[y_train]

    net = BPNet()
    n = X_train.shape[0]

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train_oh[idx]

        for i in range(0, n, BATCH_SIZE):
            Xb = X_shuf[i:i+BATCH_SIZE]
            yb = y_shuf[i:i+BATCH_SIZE]
            net.forward(Xb)
            net.backward(yb)

        # 打印损失
        preds = net.forward(X_train)
        l = net.loss(preds, y_train_oh)
        print(f'Epoch {epoch+1}/{EPOCHS}  Loss: {l:.4f}')

    # 测试集准确率
    preds = net.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'测试集准确率: {acc:.2%}')

    net.save()
    return net

# ------------------------------------------------
# 4. 推理桌面图片（OpenCV 预处理版，其余代码不变）
# ------------------------------------------------
def predict_folder(net):

    if not os.path.exists(IMG_DIR):
        print(f'未找到目录: {IMG_DIR}'); return
    files = [f for f in os.listdir(IMG_DIR)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print('文件夹中没有图片'); return

    for fname in files:
        img_path = os.path.join(IMG_DIR, fname)

        # ---------- 正确的预处理 ----------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'无法读取 {fname}，跳过'); continue

        # 1) 保证“黑底白字”：MNIST 训练集如此
        if np.mean(img) > 128:          # 如果是“白底黑字”
            img = 255 - img             # 反色 → 黑底白字

        # 2) 自适应阈值（白色数字，黑色背景）
        _, bw = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3) 膨胀：把笔画变粗
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bw = cv2.dilate(bw, kernel, iterations=3)

        # 4) 中值滤波去掉孤立噪点
        bw = cv2.medianBlur(bw, 3)

        # 5) 缩放到 28×28
        bw = cv2.resize(bw, (28, 28), interpolation=cv2.INTER_AREA)

        # 6) 额外锐化
        blur   = cv2.GaussianBlur(bw, (3, 3), 0)                # 轻微模糊
        detail = bw.astype(np.float32) - blur.astype(np.float32)  # 细节层
        alpha  = 10                                              # 锐化强度
        sharpened = bw + alpha * detail
        bw = np.clip(sharpened, 0, 255).astype(np.uint8)

# ---------- 预处理结束 ----------

        # 归一化到 [-1,1]
        img_np = ((bw.astype(np.float32) / 255.0) - 0.5) / 0.5
        img_np = img_np.reshape(1, -1)

        digit = net.predict(img_np)[0]
        print(f'{fname}  ->  识别结果: {digit}')

        # 保存 debug 图
        debug = ((img_np.reshape(28, 28) + 1) * 127.5).astype(np.uint8)
        cv2.imwrite(os.path.join(IMG_DIR, 'debug_' + fname), debug)

# ------------------------------------------------
# 5. 主入口
# ------------------------------------------------
if __name__ == '__main__':
    net = BPNet()
    if os.path.exists(MODEL_PATH):
        net.load()
    else:
        print('第一次运行，开始训练...')
        net = train()

    predict_folder(net)