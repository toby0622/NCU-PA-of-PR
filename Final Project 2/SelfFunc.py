import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix

# 設定資料集路徑
dataset_path = "Data/ORL3232"

# 設定訓練集和測試集圖片編號
train_images = [1, 3, 5, 7, 9]
test_images = [2, 4, 6, 8, 10]

# 讀取訓練集圖片和標籤
X_train = []
y_train = []
for i in range(1, 41):
    for j in train_images:
        img_path = os.path.join(dataset_path, str(i), f"{j}.bmp")
        img = io.imread(img_path)
        X_train.append(img.flatten())
        y_train.append(i - 1)

# 讀取測試集圖片和標籤
X_test = []
y_test = []
for i in range(1, 41):
    for j in test_images:
        img_path = os.path.join(dataset_path, str(i), f"{j}.bmp")
        img = io.imread(img_path)
        X_test.append(img.flatten())
        y_test.append(i - 1)

# 轉換為NumPy陣列
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


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


# 手動實現LDA
class LDA:
    def __init__(self):
        self.means = None
        self.scalings = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)

        # 計算類內散佈矩陣 (S_W)
        S_W = np.zeros((n_features, n_features))
        for label in class_labels:
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            S_W += np.dot((X_class - mean_class).T, (X_class - mean_class))

        # 計算類間散佈矩陣 (S_B)
        S_B = np.zeros((n_features, n_features))
        for label in class_labels:
            X_class = X[y == label]
            mean_class = np.mean(X_class, axis=0)
            n_class_samples = X_class.shape[0]
            mean_diff = (mean_class - mean_overall).reshape(n_features, 1)
            S_B += n_class_samples * np.dot(mean_diff, mean_diff.T)

        # 解決廣義特徵值問題 (S_W^-1 * S_B)
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        sorted_indices = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 確保取實部
        self.scalings = np.real(eigenvectors)
        self.means = mean_overall

    def transform(self, X):
        return np.dot(X - self.means, self.scalings)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class KNN:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
            nearest_neighbors_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors_labels = self.y_train[nearest_neighbors_indices]
            unique_labels, counts = np.unique(nearest_neighbors_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            y_pred.append(predicted_label)
        return np.array(y_pred)


pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lda = LDA()
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

knn = KNN(n_neighbors=1)
knn.fit(X_train_lda, y_train)
y_pred = knn.predict(X_test_lda)

# 評估分類性能
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.2%}")

# 將結果輸出為CSV表格
with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "True Label", "Predicted Label"])
    for i in range(len(y_test)):
        writer.writerow([f"image_{i + 1}", y_test[i], y_pred[i]])

# 繪製混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(set(y_test)))
plt.xticks(tick_marks, range(len(set(y_test))), rotation=45)
plt.yticks(tick_marks, range(len(set(y_test))))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
