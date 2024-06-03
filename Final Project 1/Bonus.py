import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix

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


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def train_with_lbfgs(X, y, hidden_size=150, max_iter=200):
    input_size = X.shape[1]
    output_size = len(np.unique(y))

    # 初始化權重和偏置
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)

    def forward(X, W1, b1, W2, b2):
        hidden = relu(X @ W1 + b1)
        output = softmax(hidden @ W2 + b2)
        return output

    def objective(params, X, y):
        W1 = params[:input_size * hidden_size].reshape(input_size, hidden_size)
        b1 = params[input_size * hidden_size:input_size * hidden_size + hidden_size]
        W2 = params[input_size * hidden_size + hidden_size:-output_size].reshape(hidden_size, output_size)
        b2 = params[-output_size:]

        output = forward(X, W1, b1, W2, b2)
        loss = -np.sum(np.log(output[np.arange(len(y)), y]))
        return loss

    def gradient(params, X, y):
        W1 = params[:input_size * hidden_size].reshape(input_size, hidden_size)
        b1 = params[input_size * hidden_size:input_size * hidden_size + hidden_size]
        W2 = params[input_size * hidden_size + hidden_size:-output_size].reshape(hidden_size, output_size)
        b2 = params[-output_size:]

        hidden = relu(X @ W1 + b1)
        output = softmax(hidden @ W2 + b2)

        grad_output = output.copy()
        grad_output[np.arange(len(y)), y] -= 1

        grad_W2 = hidden.T @ grad_output
        grad_b2 = np.sum(grad_output, axis=0)

        grad_hidden = grad_output @ W2.T
        grad_hidden[hidden <= 0] = 0

        grad_W1 = X.T @ grad_hidden
        grad_b1 = np.sum(grad_hidden, axis=0)

        grad = np.concatenate((grad_W1.ravel(), grad_b1, grad_W2.ravel(), grad_b2))
        return grad

    # 將權重和偏置展平為一個參數向量
    params0 = np.concatenate((W1.ravel(), b1, W2.ravel(), b2))

    # 使用 L-BFGS 算法進行優化
    result = minimize(objective, params0, args=(X, y), method='L-BFGS-B',
                      jac=gradient, options={'maxiter': max_iter})

    # 從優化結果中提取權重和偏置
    W1 = result.x[:input_size * hidden_size].reshape(input_size, hidden_size)
    b1 = result.x[input_size * hidden_size:input_size * hidden_size + hidden_size]
    W2 = result.x[input_size * hidden_size + hidden_size:-output_size].reshape(hidden_size, output_size)
    b2 = result.x[-output_size:]

    return W1, b1, W2, b2


# 訓練模型
W1, b1, W2, b2 = train_with_lbfgs(train_data_normalized, train_labels)


# 進行預測
def predict(X, W1, b1, W2, b2):
    hidden = relu(X @ W1 + b1)
    output = softmax(hidden @ W2 + b2)
    return np.argmax(output, axis=1)


predictions = predict(test_data_normalized, W1, b1, W2, b2)
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
