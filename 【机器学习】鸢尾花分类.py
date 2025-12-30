import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 加载数据集
iris = load_iris()                                                              # 使用 load_iris 函数加载鸢尾花数据集，返回一个 Bunch 对象
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)                 # 将数据集的特征数据转换为 pandas DataFrame，并指定列名为特征名称
data['target'] = iris.target                                                    # 在 DataFrame 中添加一列 'target'，对应每个样本的目标标签
data['target_names'] = data['target'].apply(lambda x: iris.target_names[x])     # 添加一列 'target_names'，通过映射 'target' 列的值来获取对应的目标名称

# 查看数据
# print(data.head())      # 查看数据集前五行数据
# print('-')
# print(data.info())      # 查看数据集的基本信息
# print('-')
# print(data.describe())  # 查看数据集的统计信息
# 绘制特征分布图
sns.pairplot(data, hue='target', markers=["o", "s", "D"])  # 使用 seaborn 库的 pairplot 函数绘制散点图矩阵，根据 'target' 列着色，并使用不同的标记
# plt.show()  # 显示图形

# 检查是否有缺失值
print(data.isnull().sum())  # 打印每列的缺失值数量，以检查数据集中是否存在缺失值

# 标准化特征值
scaler = StandardScaler()  # 创建一个 StandardScaler 对象
data[iris.feature_names] = scaler.fit_transform(data[iris.feature_names])  # 对特征列进行标准化处理，使其均值为 0，标准差为 1

# 查看标准化后的数据
print(data.head())  # 打印标准化后的数据集的前五行，查看标准化效果

# 分割数据集
X = data[iris.feature_names]  # 特征数据，包含所有特征列
y = data['target']  # 目标数据，包含目标标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)  # 将数据集按 70% 训练集和 30% 测试集进行分割，设置随机种子为 42 以确保可重复性

# 查看分割后的数据集大小
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")  # 打印训练集和测试集的大小

# 初始化 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)  # 创建一个 K 近邻分类器对象，设置近邻数为 3

# 训练模型
knn.fit(X_train, y_train)     # 使用训练数据训练 K 近邻分类器模型



# 预测测试集
y_pred = knn.predict(X_test)  # 使用训练好的模型对测试数据进行预测，得到预测标签


# 定义参数范围
param_grid = {'n_neighbors': range(1, 20)}  # 创建一个参数字典，设置 'n_neighbors' 参数的取值范围为 1 到 19

# 网格搜索
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)  # 创建一个 GridSearchCV 对象，使用 KNeighborsClassifier 和定义的参数范围，设置交叉验证折数为 5
grid_search.fit(X_train, y_train)                                     # 使用训练数据进行网格搜索，以找到最佳参数组合

# 最优参数
print(f"Best parameters: {grid_search.best_params_}")  # 打印通过网格搜索找到的最优参数
print('-')

# 使用最优参数训练模型
knn_best = grid_search.best_estimator_  # 获取使用最优参数训练的最佳模型
y_pred_best = knn_best.predict(X_test)  # 使用最佳模型对测试数据进行预测

# 评估模型
print(f"Accuracy（best）: {accuracy_score(y_test, y_pred_best)}")
print("Classification Report（best）:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix（best）:\n", confusion_matrix(y_test, y_pred_best))




# 新样本数据
new_samples = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]

# 将新样本数据转换为 DataFrame，并指定特征名称
new_samples_df = pd.DataFrame(new_samples, columns=iris.feature_names)



# 标准化新样本
new_samples_scaled = pd.DataFrame(scaler.transform(new_samples_df),columns=iris.feature_names) # 使用之前的标准化器 scaler 对新样本进行标准化处理，使其与训练数据具有相同的尺度

# 预测新样本的类别
predictions = knn_best.predict(new_samples_scaled)  # 使用最佳模型 knn_best 对标准化后的新样本进行类别预测
print(f"新样本的预测类别: {predictions}")  # 打印新样本的预测类别

import numpy as np

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)

# 按行归一化（每一行加起来是 1）
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("混淆矩阵（行归一化，百分比显示）")
plt.show()
