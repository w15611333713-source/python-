import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import scipy.io  # 导入scipy库，用于加载.mat文件


# --- 1. CWRU数据集的加载与预处理 (已更新) ---
def load_cwru_data(data_dir, window_size=2048, stride=512):
    """
    加载CWRU轴承数据，将其分割成窗口（样本），并分配标签。
    """
    all_data = []
    all_labels = []

    # 注意：这里的 'B007', 'IR007' 等仅用于打印信息，实际识别靠下面的文件名判断
    label_map = {
        'Normal': 0, 'B007': 1, 'IR007': 2, 'OR007': 3
    }

    print("正在加载和预处理数据...")
    if not os.path.isdir(data_dir):
        print(f"错误：找不到文件夹 '{data_dir}'。请检查路径是否正确。")
        return None, None

    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            fault_type = 'Unknown'
            label = -1

            # =======================================================
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            #  修改部分：根据数字文件名来识别故障类型
            # =======================================================
            if '98' in filename or '97' in filename:  # 97和98都是正常文件
                fault_type, label = 'Normal', 0
            elif '119' in filename or '120' in filename or '121' in filename:  # 滚动体故障
                fault_type, label = 'B007', 1
            elif '106' in filename or '105' in filename or '107' in filename:  # 内圈故障
                fault_type, label = 'IR007', 2
            elif '131' in filename or '130' in filename or '132' in filename:  # 外圈故障
                fault_type, label = 'OR007', 3
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            # =======================================================

            if label == -1:
                print(f"跳过未知文件: {filename}")
                continue

            print(f"正在处理文件: {filename} -> 故障类型: {fault_type}")

            filepath = os.path.join(data_dir, filename)
            mat_data = scipy.io.loadmat(filepath)

            de_key = [key for key in mat_data.keys() if 'DE_time' in key][0]
            signal = mat_data[de_key].flatten()

            num_samples = (len(signal) - window_size) // stride + 1
            for i in range(num_samples):
                window = signal[i * stride: i * stride + window_size]
                window = (window - np.mean(window)) / np.std(window)
                all_data.append(window)
                all_labels.append(label)

    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    X = np.expand_dims(X, axis=1)

    return X, y


# --- 主程序开始执行 ---

# 你的文件夹路径 (请确保这里是你自己的正确路径)
DATA_DIRECTORY = r'C:\Users\27666\Desktop\CWRU_data'

WINDOW_SIZE = 2048
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# 加载真实数据
X_raw, y_raw = load_cwru_data(DATA_DIRECTORY, window_size=WINDOW_SIZE)

# 只有成功加载数据后才继续执行
if X_raw is not None and len(X_raw) > 0:
    print(f"\n数据加载完成。总样本数: {len(X_raw)}, 单个样本的形状: {X_raw[0].shape}")

    # 后续代码与之前完全相同
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw)
    X_train_tensor = torch.tensor(X_train);
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test);
    y_test_tensor = torch.tensor(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    class CNN1D(nn.Module):
        def __init__(self, num_classes=4):
            super(CNN1D, self).__init__()
            self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 64, 16, 24), nn.ReLU(), nn.MaxPool1d(2, 2))
            self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool1d(2, 2))
            self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool1d(2, 2))
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(nn.Linear(1024, 100), nn.ReLU(), nn.Linear(100, num_classes))

        def forward(self, x):
            x = self.conv1(x);
            x = self.conv2(x);
            x = self.conv3(x)
            return self.fc(self.flatten(x))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    print("\n模型结构:");
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n开始训练模型...")
    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f'轮次 {epoch + 1}/{EPOCHS}')
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("训练完成。")

    print("\n正在测试集上评估模型...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f'测试集准确率: {accuracy * 100:.2f}%')

    class_names = ['正常', '滚动体故障', '内圈故障', '外圈故障']
    cm = confusion_matrix(all_labels, all_preds)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("在CWRU测试集上的混淆矩阵")
    plt.show()
else:
    print("\n数据加载失败，程序已停止。请检查文件和路径。")