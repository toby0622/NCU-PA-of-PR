import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import csv

# input and target csv path
input_data_file = "iris_in.csv"
target_data_file = "iris_out.csv"

# pandas read csv file
x_full = pd.read_csv(input_data_file)
y_full = pd.read_csv(target_data_file)

# one-hot-encoding
encoder = OneHotEncoder(sparse_output=False)
y_full_encoded = encoder.fit_transform(y_full.values.reshape(-1, 1))

# training data (using head 75)
x_train = x_full.head(75).values
y_train = y_full_encoded[:75, :]

# testing data (using tail 75)
x_test = x_full.tail(75).values
y_test = y_full_encoded[-75:, :]


# sigmoid function (0 - 1 mapping)
def sigmoid(x):
    return np.round(1 / (1 + np.exp(-x)), 4)


# sigmoid derivative
def sigmoid_derivative(x):
    return np.round(x * (1 - x), 4)


# softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.round(e_x / np.sum(e_x, axis=1, keepdims=True), 4)


# neural network settings
input_layer_size = 4  # input nodes
hidden_layer_size = 12  # hidden nodes
output_layer_size = 3  # output nodes

# specify random seed (for result reproduction)
np.random.seed(112522083)

# 'He' method initialization (weight initialization based on previous layer size)
weights_input_hidden = np.round(np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2. / input_layer_size),
                                4)
bias_hidden = np.round(np.zeros((1, hidden_layer_size)), 4)
weights_hidden_output = np.round(
    np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2. / hidden_layer_size), 4)
bias_output = np.round(np.zeros((1, output_layer_size)), 4)

# learning rate
learning_rate = 0.008

# epochs (training rounds)
epochs = 3600

# rmse for each epoch
rmse_history = []

for epoch in range(epochs):
    # forward propagation
    hidden_layer_input = np.round(np.dot(x_train, weights_input_hidden) + bias_hidden, 4)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output_input = np.round(np.dot(hidden_layer_output, weights_hidden_output) + bias_output, 4)
    final_output = softmax(final_output_input)

    error = np.round(y_train - final_output, 4)
    rmse = np.round(np.sqrt(np.mean(np.square(error))), 4)
    rmse_history.append(rmse)

    # backpropagation
    d_predicted_output = error
    d_softmax_output = d_predicted_output * final_output * (1 - final_output)
    error_hidden_layer = np.round(d_softmax_output.dot(weights_hidden_output.T), 4)
    d_hidden_layer = np.round(error_hidden_layer * sigmoid_derivative(hidden_layer_output), 4)

    # weight and bias update
    weights_hidden_output += np.round(hidden_layer_output.T.dot(d_softmax_output) * learning_rate, 4)
    bias_output += np.round(np.sum(d_softmax_output, axis=0, keepdims=True) * learning_rate, 4)
    weights_input_hidden += np.round(x_train.T.dot(d_hidden_layer) * learning_rate, 4)
    bias_hidden += np.round(np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate, 4)

    if epoch % 300 == 0:
        print(f'epoch: {epoch}, rmse: {rmse}')
    elif epoch + 1 == epochs:
        print(f'epoch: {epoch + 1}, rmse: {rmse}')

# testing data predictions
hidden_layer_input_test = np.round(np.dot(x_test, weights_input_hidden) + bias_hidden, 4)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
final_output_test = np.round(np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output, 4)
final_output_test = softmax(final_output_test)

# accuracy calculation
true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(final_output_test, axis=1)
accuracy = np.round(np.mean(predicted_labels == true_labels), 4)
accuracy_percentage = round(accuracy * 100)
print(f'Test Accuracy: {accuracy_percentage}%')

# prediction result export
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class1', 'Class2', 'Class3', 'PredictedLabel'])
    for i in range(len(final_output_test)):
        predicted_label = np.argmax(final_output_test[i])
        writer.writerow(np.round(np.append(final_output_test[i], predicted_label), 4))

# rmse vs epochs graph
plt.figure(figsize=(10, 6))
plt.plot(rmse_history, label='Training RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE vs Epoch')
plt.legend()
plt.show()
