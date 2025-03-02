#!/usr/bin/env python
# coding: utf-8

# In[1]:


def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = [0.05, 0.1] 
weights_input_hidden = [[0.15, 0.2], [0.25, 0.3]] 
bias_hidden = [0.35, 0.35]

weights_hidden_output = [[0.4, 0.45], [0.5, 0.55]]  
bias_output = [0.6, 0.6]

targets = [0.1, 0.99] 
learning_rate = 0.5  

net_hidden = [
    inputs[0] * weights_input_hidden[0][0] + inputs[1] * weights_input_hidden[0][1] + bias_hidden[0],
    inputs[0] * weights_input_hidden[1][0] + inputs[1] * weights_input_hidden[1][1] + bias_hidden[1]
]

out_hidden = [sigmoid(net_hidden[0]), sigmoid(net_hidden[1])]

net_output = [
    out_hidden[0] * weights_hidden_output[0][0] + out_hidden[1] * weights_hidden_output[0][1] + bias_output[0],
    out_hidden[0] * weights_hidden_output[1][0] + out_hidden[1] * weights_hidden_output[1][1] + bias_output[1]
]

out_output = [sigmoid(net_output[0]), sigmoid(net_output[1])]


error = 0.5 * ((targets[0] - out_output[0])**2 + (targets[1] - out_output[1])**2)

error_output = [(out_output[i] - targets[i]) * sigmoid_derivative(out_output[i]) for i in range(2)]


for i in range(2):
    for j in range(2):
        weights_hidden_output[i][j] -= learning_rate * error_output[i] * out_hidden[j]
    bias_output[i] -= learning_rate * error_output[i]

error_hidden = [
    (error_output[0] * weights_hidden_output[0][i] + error_output[1] * weights_hidden_output[1][i]) * sigmoid_derivative(out_hidden[i])
    for i in range(2)
]

for i in range(2):
    for j in range(2):
        weights_input_hidden[i][j] -= learning_rate * error_hidden[i] * inputs[j]
    bias_hidden[i] -= learning_rate * error_hidden[i]

# طباعة النتائج
print("Updated Weights Input-Hidden:", weights_input_hidden)
print("Updated Weights Hidden-Output:", weights_hidden_output)
print("Updated Bias Hidden:", bias_hidden)
print("Updated Bias Output:", bias_output)


# In[ ]:




