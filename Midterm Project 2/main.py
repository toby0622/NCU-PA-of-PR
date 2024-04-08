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
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)


# neural network settings
input_layer_size = 4  # input nodes
hidden_layer_size = 4  # hidden nodes
output_layer_size = 3  # output nodes

# specify random seed (for result reproduction)
np.random.seed(411522)

# 'He' method initialization (weight initialization based on previous layer size)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2. / input_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2. / hidden_layer_size)
bias_output = np.zeros((1, output_layer_size))

# learning rate
learning_rate = 0.01
# epochs (training rounds)
epochs = 1600
# rmse for each epoch
rmse_history = []

for epoch in range(epochs):
    # forward propagation
    hidden_layer_input = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = final_output_input

    error = y_train - final_output
    rmse = np.sqrt(np.mean(np.square(error)))
    rmse_history.append(rmse)

    # backpropagation
    d_predicted_output = error
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # weight and bias update
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += x_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 100 == 0:
        print(f'epoch: {epoch}, rmse: {rmse}')
    elif epoch + 1 == epochs:
        print(f'epoch: {epoch + 1}, rmse: {rmse}')

# testing data predictions
hidden_layer_input_test = np.dot(x_test, weights_input_hidden) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
final_output_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output

# accuracy calculation
true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(final_output_test, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
accuracy_percentage = round(accuracy * 100)
print(f'Test Accuracy: {accuracy_percentage}%')

# prediction result export
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class1', 'Class2', 'Class3', 'PredictedLabel'])
    for i in range(len(final_output_test)):
        predicted_label = np.argmax(final_output_test[i])
        writer.writerow(np.append(final_output_test[i], predicted_label))

# rmse vs epochs graph
plt.figure(figsize=(10, 6))
plt.plot(rmse_history, label='Training RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE vs Epoch')
plt.legend()
plt.show()
