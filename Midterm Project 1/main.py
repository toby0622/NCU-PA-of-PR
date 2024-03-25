import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# input and target csv path
input_data_file = "iris_in.csv"
target_data_file = "iris_out.csv"

# pandas read file
x_full = pd.read_csv(input_data_file)
y_full = pd.read_csv(target_data_file)

# training data (using head 75)
x_train = x_full.head(75).values
y_train = y_full.head(75).values.reshape(-1, 1)

# testing data (using tail 75)
x_test = x_full.tail(75).values
y_test = y_full.tail(75).values.reshape(-1, 1)

# 定义 sigmoid 函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 定义线性传递函数及其导数
def linear_transfer(x):
    return x


def linear_transfer_derivative(x):
    return 1


# 确保这里是按照您的数据和网络结构来设置的
input_layer_size = 4  # 输入层神经元数量（特征数量）
hidden_layer_size = 4  # 隐藏层神经元数量，举例
output_layer_size = 1  # 输出层神经元数量

# 使用 He 初始化方法初始化权重
np.random.seed(112522083)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2. / input_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2. / hidden_layer_size)
bias_output = np.zeros((1, output_layer_size))

learning_rate = 0.01  # 学习率
epochs = 2000  # 迭代次数
rmse_history = []  # 用于记录每个epoch的rmse

for epoch in range(epochs):
    # 前向传播
    hidden_layer_input = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = linear_transfer(final_output_input)

    # 计算 RMSE
    error = y_train - final_output
    rmse = np.sqrt(np.mean(np.square(error)))
    rmse_history.append(rmse)

    # 反向传播
    d_predicted_output = error * linear_transfer_derivative(final_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # 更新权重和偏置
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += x_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 100 == 0:
        print(f'epoch: {epoch}, rmse: {rmse}')
    elif epoch + 1 == epochs:
        print(f'epoch: {epoch + 1}, rmse: {rmse}')

# 绘制 rmse 曲线
plt.figure(figsize=(10, 6))
plt.plot(rmse_history, label='Training RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE vs Epoch')
plt.legend()
plt.show()

# 使用训练好的模型对测试集进行预测
hidden_layer_input_test = np.dot(x_test, weights_input_hidden) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
output_layer_output_test = linear_transfer(output_layer_input_test)

with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(output_layer_output_test.transpose())

result = []
counter = 0

for element in output_layer_output_test:
    if 0.5 <= element < 1.5:
        result.append(1)
    elif 1.5 <= element < 2.5:
        result.append(2)
    elif 2.5 <= element < 3.5:
        result.append(3)
    else:
        result.append(0)

for i in range(len(y_test)):
    if result[i] == y_test[i]:
        counter += 1

accuracy = round((counter / len(y_test)) * 100)
print('Test Accuracy: ' + str(accuracy) + '%')
