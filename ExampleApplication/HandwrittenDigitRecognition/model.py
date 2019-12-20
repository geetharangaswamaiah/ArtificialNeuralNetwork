import pandas as pd
import scipy.io
from ArtificialNeuralNetwork.utilities import *

# Read data
datafile = 'ex4data1.mat'
mat = scipy.io.loadmat( datafile )
X = mat['X']
y = mat['y']

# Define Network Architecture
sizeof_input_layer = 400
sizeof_hidden_layer = 25
sizeof_output_layer = 10
regularization_param = 0

# Set Network Parameters
setParameters(sizeof_input_layer, sizeof_hidden_layer, sizeof_output_layer, regularization_param)

# Train the model
model = train(X, y)

# Accuracy of the model
accuracy = getModelAccuracy(model, X, y)

# Predict data
X_test = X[2000]
y_test = y[2000]
prediction = predict(model, X_test)
print("Data ->" ,y_test)
print("Prediction ->" ,prediction)


