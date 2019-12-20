# Import libraries
import numpy as np
import pandas as pd
import scipy.io
import random 
import scipy.optimize
import itertools

# Sigmoid Function
def sigmoid(z):
	g = 1.0 / (1.0 + np.exp(-z))
	return g

# Sigmoid Gradient Function
def sigmoidGradient(z):
	g_grad = sigmoid(z) * (1 - sigmoid(z))
	return g_grad

# Unroll theta
def unrollTheta(theta_matrix_list):
  theta_flattened_list = [theta.flatten() for theta in theta_matrix_list]
  theta_flattened_list = list(itertools.chain.from_iterable(theta_flattened_list))
  theta_list = np.array(theta_flattened_list).reshape((len(theta_flattened_list), 1))
  return theta_list 

# Reshape theta
def reshapeTheta(theta_list):
  theta1 = theta_list[:(input_layer_size + 1) * hidden_layer_size].reshape((hidden_layer_size, input_layer_size + 1))
  theta2 = theta_list[(input_layer_size + 1) * hidden_layer_size:].reshape((num_labels, hidden_layer_size + 1))
  theta_matrix = [theta1, theta2]
  return theta_matrix

# Unroll X
def unrollX(X):
  x_list = np.array(X.flatten()).reshape((m * (input_layer_size + 1), 1))
  return x_list

# Reshape X
def reshapeX(x_list):
  x_matrix = np.array(x_list).reshape((m, input_layer_size + 1))
  return x_matrix

# Randomly Initialize Weights
def randomlyInitializeWeights(l_in, l_out):
  epsilon = 0.12
  weights = np.random.rand(l_out, (1 + l_in)) * 2 * epsilon - epsilon
  return weights

#Forward Propagation
def forwardPropagation(x_one, theta):
  zs = []
  for i in range(len(theta)):
    z = theta[i].dot(x_one).reshape((theta[i].shape[0], 1))
    a = sigmoid(z)
    zs.append((z, a))
    if i == (len(theta)-1):
      return zs
    a = np.insert(a, 0, 1)
    x_one = a

# Cost Function
def costFunction(theta_list, x_list, y, lmbda):
  theta_array = reshapeTheta(theta_list)
  x_array = reshapeX(x_list)
  J = 0
  for i in range(m):
    x_one = x_array[i]
    h_one = forwardPropagation(x_one, theta_array)[-1][1]
    y_bool = np.zeros((10, 1))
    y_bool[y[i] - 1] = 1
    J = J + (-y_bool.T.dot(np.log(h_one)) - (1 - y_bool.T).dot(np.log(1 - h_one)))
  J = J / m
  # Regularization
  reg_term = 0
  for theta in theta_array:
    reg_term = reg_term + np.sum(theta * theta)
  reg_term = reg_term * (lmbda / (2 * m))
  J = float(J) + reg_term
  return J

# Back Propagation
def backPropagation(theta_list, x_list, y, lmbda):
  theta_array = reshapeTheta(theta_list)
  x_array = reshapeX(x_list)
  grad1 = np.zeros((np.shape(theta_array[0])))
  grad2 = np.zeros((np.shape(theta_array[1])))
  for i in range(m):
    x_one = x_array[i]
    a1 = x_one.reshape((input_layer_size + 1, 1))
    f_prop = forwardPropagation(x_one, theta_array)
    z2 = f_prop[0][0]
    a2 = f_prop[0][1]
    a2 = np.insert(a2, 0, 1, axis=0)
    z3 = f_prop[1][0]
    a3 = f_prop[1][1]
    y_bool = np.zeros((10, 1))
    y_bool[y[i] - 1] = 1
    delta3 = a3 - y_bool
    delta2 = theta_array[1].T[1:,:].dot(delta3) * sigmoidGradient(z2)
    grad2 = grad2 + delta3.dot(a2.T)
    grad1 = grad1 + delta2.dot(a1.T)
  grad1 = grad1 / m
  grad2 = grad2 / m
  # Regularization
  grad1[:,1:] = grad1[:,1:] + (float(lmbda) / m) * theta_array[0][:,1:]
  grad2[:,1:] = grad2[:,1:] + (float(lmbda) / m) * theta_array[1][:,1:]
  grad = unrollTheta([grad1, grad2]).flatten()
  return grad

# Set Parameters
def setParameters(inpt, hidden, output, reg_param):
  print("             --------------------------------------------------")
  print("             |                                                |")
  print("             |            Artificial Neural Network           |")
  print("             |                                                |")
  print("             --------------------------------------------------")
  print()
  global input_layer_size
  global hidden_layer_size
  global num_labels
  global lmbda
  input_layer_size = inpt
  hidden_layer_size = hidden
  num_labels = output
  lmbda = reg_param
  print("Model parameters set")

# Train model
def train(X_data, y):
  print("Training model...")
  # Add intercept term
  X_data = np.insert(X_data, 0, 1, axis=1)
  global X
  X = X_data
  global m
  m = np.shape(X)[0]
  initial_theta_1 = randomlyInitializeWeights(input_layer_size, hidden_layer_size).flatten()
  initial_theta_2 = randomlyInitializeWeights(hidden_layer_size, num_labels).flatten()
  randomThetas_unrolled = unrollTheta([initial_theta_1, initial_theta_2])
  result = scipy.optimize.fmin_cg(costFunction, x0=randomThetas_unrolled, fprime=backPropagation, args=(unrollX(X), y, lmbda), maxiter=50,disp=True,full_output=True)
  return reshapeTheta(result[0])

# Accuracy
def getModelAccuracy(theta_updated, X, y):
  X = np.insert(X, 0, 1, axis=1)
  num_correct = 0
  for i in range(m):
    x_one = X[i]
    h = forwardPropagation(x_one, theta_updated)[-1][1]
    y_bool = np.zeros((10, 1))
    y_bool[y[i] - 1] = 1
    max_prob = np.argmax(h)
    p_bool = np.zeros((10, 1))
    p_bool[max_prob] = 1
    if(p_bool == y_bool).all:
      num_correct = num_correct + 1
  accuracy = (num_correct / m) * 100
  return accuracy

# Prediction
def predict(theta_updated, X_test):
  X_test = np.insert(X_test, 0, 1)
  h = forwardPropagation(X_test, theta_updated)[-1][1]
  max_prob = np.argmax(h)
  prediction = max_prob + 1
  return prediction

