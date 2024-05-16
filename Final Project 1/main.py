import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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

# 進行PCA降維
pca = PCA(n_components=0.95)  # 選擇保留95%的資訊
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
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
print("Accuracy:", accuracy)

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