import os
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv

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
        y_train.append(i-1)

# 讀取測試集圖片和標籤
X_test = []
y_test = []
for i in range(1, 41):
    for j in test_images:
        img_path = os.path.join(dataset_path, str(i), f"{j}.bmp")
        img = io.imread(img_path)
        X_test.append(img.flatten())
        y_test.append(i-1)

# 轉換為NumPy陣列
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# PCA降維
pca = PCA(n_components=0.95)  # 保留95%的變異量
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# LDA
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

# 最近鄰居分類器
knn = KNeighborsClassifier(n_neighbors=1)
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
        writer.writerow([f"image_{i+1}", y_test[i], y_pred[i]])