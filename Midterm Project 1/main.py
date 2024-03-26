import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# input and target csv path
input_data_file = "iris_in.csv"
target_data_file = "iris_out.csv"

# pandas read csv file
x_full = pd.read_csv(input_data_file)
y_full = pd.read_csv(target_data_file)

# training data (using head 75)
x_train = x_full.head(75).values
y_train = y_full.head(75).values.reshape(-1, 1)

# testing data (using tail 75)
x_test = x_full.tail(75).values
y_test = y_full.tail(75).values.reshape(-1, 1)


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)


# linear transfer function (purelin simulation)
def linear_transfer(x):
    return x


# linear transfer derivative
def linear_transfer_derivative(x):
    return 1


# neural network settings
input_layer_size = 4  # input nodes
hidden_layer_size = 4  # hidden nodes
output_layer_size = 1  # output nodes

# He method initialization (weight initialization based on previous layer size)
np.random.seed(112522083)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2. / input_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2. / hidden_layer_size)
bias_output = np.zeros((1, output_layer_size))

# learning rate
learning_rate = 0.01
# epochs (training rounds)
epochs = 2000
# rmse for each epoch
rmse_history = []

for epoch in range(epochs):
    # forward propagation
    hidden_layer_input = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = linear_transfer(final_output_input)

    # rmse calculation
    error = y_train - final_output
    rmse = np.sqrt(np.mean(np.square(error)))
    rmse_history.append(rmse)

    # bakc propagation
    d_predicted_output = error * linear_transfer_derivative(final_output)
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

# rmse vs epochs graph
plt.figure(figsize=(10, 6))
plt.plot(rmse_history, label='Training RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE vs Epoch')
plt.legend()
plt.show()

# testing data predictions
hidden_layer_input_test = np.dot(x_test, weights_input_hidden) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output) + bias_output
output_layer_output_test = linear_transfer(output_layer_input_test)

# prediction result export
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(output_layer_output_test.transpose())

result = []
counter = 0

# classification method for 'output layer output test'
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

# accuracy calculation
accuracy = round((counter / len(y_test)) * 100)
print('Test Accuracy: ' + str(accuracy) + '%')
