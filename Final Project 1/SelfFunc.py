import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

# 設定資料路徑
data_path = 'Data/ORL3232'

# 設定訓練和測試資料的編號
train_indices = [1, 3, 5, 7, 9]
test_indices = [2, 4, 6, 8, 10]

# 初始化儲存訓練和測試資料的列表
train_data = []
train_labels = []
test_data = []
test_labels = []

# 遍歷每個類別的資料夾
for i in range(1, 41):
    class_path = os.path.join(data_path, str(i))

    # 遍歷每個圖片檔案
    for j in range(1, 11):
        img_path = os.path.join(class_path, str(j) + '.bmp')

        # 讀取圖片並轉換為numpy陣列
        img = Image.open(img_path)
        img_array = np.array(img).flatten()  # 將圖片展開為一維陣列

        # 根據編號將圖片分配到訓練或測試資料
        if j in train_indices:
            train_data.append(img_array)
            train_labels.append(i - 1)  # 標籤從0開始
        elif j in test_indices:
            test_data.append(img_array)
            test_labels.append(i - 1)  # 標籤從0開始

# 將訓練和測試資料轉換為numpy陣列
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


# 手動實現PCA
class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # 計算均值
        self.mean = np.mean(X, axis=0)
        # 去均值
        X_centered = X - self.mean
        # 計算協方差矩陣
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # 計算特徵值和特徵向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # 降序排序特徵值和特徵向量
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_index]
        sorted_eigenvectors = eigenvectors[:, sorted_index]

        # 計算需要的主成分數量以保留指定比例的資訊
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumulative_variance_ratio = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
            self.n_components = np.searchsorted(cumulative_variance_ratio, self.n_components) + 1

        # 選擇前n個特徵向量
        if self.n_components is not None:
            sorted_eigenvectors = sorted_eigenvectors[:, :self.n_components]
        self.components = sorted_eigenvectors

    def transform(self, X):
        # 去均值
        X_centered = X - self.mean
        # 投影到主成分空間
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# 手動實現 MinMaxScaler
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        # 計算最小值和最大值
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        # 避免除以零的錯誤
        scale = self.max - self.min
        scale[scale == 0] = 1
        return (X - self.min) / scale

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


pca = PCA(n_components=0.95)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(test_data)

# MaxMin歸一化
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data_pca)
test_data_normalized = scaler.transform(test_data_pca)

# 建立Entropy based BackPropagation神經網路
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                    alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                    max_iter=200, shuffle=True, random_state=42)

# 訓練模型
mlp.fit(train_data_normalized, train_labels)

# 模型評估
predictions = mlp.predict(test_data_normalized)
accuracy = accuracy_score(test_labels, predictions)
print(f"accuracy: {accuracy:.2%}")

with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True Label', 'Predicted Label'])
    writer.writerows(zip(test_labels, predictions))

# 繪製混淆矩陣
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(set(test_labels)))
plt.xticks(tick_marks, set(test_labels), rotation=45)
plt.yticks(tick_marks, set(test_labels))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
