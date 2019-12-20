## Python Library for Artificial Neural Network

* Artificial Neural Network is a series of Learning Algorithms to recognize underlying relationships in a dataset through a process that is based on human brain operations.

* Install the library using the command pip3 install git+https://github.com/rgeetha2010/ArtificialNeuralNetwork.git

* An example application is provided for reference.

* setParameters(sizeof_input_layer, sizeof_hidden_layer, sizeof_output_layer, reg_param)
  - sizeof_input_layer: Number of units in the input layer
  - sizeof_hidden_layer: Number of units in the hidden layer
  - sizeof_output_layer: Number of units in the output layer (Number of labels/classes)
  - reg_param: Regularization Parameter - Control on the fitting parameters

* model = train(X, y)
  - X: Example features of training set
  - y: Corresponding labels of training set

* accuracy = getModelAccuracy(model, X_test, y_test):
  - model: Trained model
  - X_test: Example features of test set
  - y_test: Corresponding labels of test set

* prediction = predict(model, X_test)
  - model: Trained model
  - X_test: Example features of test set
