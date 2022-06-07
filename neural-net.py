"""
PES1201800395
PES1201801549
PES1201801618
Code for designing a 2-layer neural network (LINEAR->SIGMOID->LINEAR->SIGMOID) from scratch
"""

# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)

# Helper functions for designing the neural network
def sigmoid(Z):
    """
    Implements the sigmoid activation function
    """
    A = 1 / (1 + np.exp(-Z))

    return A

def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing the parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01                           # random initialisation
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the neural network's forward propagation.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2                           # pre-activation variable
    A2 = sigmoid(Z2)                                   # activation values

    parameters["Z1"] = Z1
    parameters["A1"] = A1
    parameters["Z2"] = Z2
    parameters["A2"] = A2

    return parameters

def cost(Y, parameters):
    """
    Implements the cross-entropy cost function.
    """
    m = Y.shape[1]                           # number of examples
    AL = parameters["A2"]                    # activation values

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)                  # this turns [[num]] into num
    assert(cost.shape == ())

    return cost

def back_propagation(X, Y, parameters):
    """
    Implements the backward propagation for the neural network
    """
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    Z1 = parameters["Z1"]
    W2 = parameters["W2"]
    m = X.shape[1]                           # number of examples

    # back propagation gradients
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = (np.sum(dZ2, axis=1, keepdims=True)) / m
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1-A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = (np.sum(dZ1, axis=1, keepdims=True)) / m

    parameters["dZ2"] = dZ2
    parameters["dW2"] = dW2
    parameters["db2"] = db2
    parameters["dZ1"] = dZ1
    parameters["dW1"] = dW1
    parameters["db1"] = db1

    return parameters

def adam_optimizer(parameters, n_x, n_h, n_y, iteration):
    """
    Implements the Adam optimization algorithm
    """
    beta1 = 0.9                          # first moment hyperparameter
    beta2 = 0.999                        # second moment hyperparameter

    dW1 = parameters["dW1"]
    db1 = parameters["db1"]
    dW2 = parameters["dW2"]
    db2 = parameters["db2"]

    VdW1 = np.zeros(shape=(n_h, n_x))    # momentum variable
    VdW2 = np.zeros(shape=(n_y, n_h))
    Vdb1 = np.zeros(shape=(n_h, 1))
    Vdb2 = np.zeros(shape=(n_y, 1))
    SdW1 = np.zeros(shape=(n_h, n_x))    # RMS prop variable
    SdW2 = np.zeros(shape=(n_y, n_h))
    Sdb1 = np.zeros(shape=(n_h, 1))
    Sdb2 = np.zeros(shape=(n_y, 1))

    VdW1 = np.multiply(beta1, VdW1) + np.multiply((1 - beta1), dW1)
    VdW2 = np.multiply(beta1, VdW2) + np.multiply((1 - beta1), dW2)
    Vdb1 = np.multiply(beta1, Vdb1) + np.multiply((1 - beta1), db1)
    Vdb2 = np.multiply(beta1, Vdb2) + np.multiply((1 - beta1), db2)
    SdW1 = np.multiply(beta2, SdW1) + np.multiply((1 - beta2), np.power(dW1, 2))
    SdW2 = np.multiply(beta2, SdW2) + np.multiply((1 - beta2), np.power(dW2, 2))
    Sdb1 = np.multiply(beta2, Sdb1) + np.multiply((1 - beta2), np.power(db1, 2))
    Sdb2 = np.multiply(beta2, Sdb2) + np.multiply((1 - beta2), np.power(db2, 2))

    # bias correction
    VdW1_corrected = VdW1 / (1 - np.power(beta1, iteration))
    VdW2_corrected = VdW2 / (1 - np.power(beta1, iteration))
    Vdb1_corrected = Vdb1 / (1 - np.power(beta1, iteration))
    Vdb2_corrected = Vdb2 / (1 - np.power(beta1, iteration))
    SdW1_corrected = SdW1 / (1 - np.power(beta2, iteration))
    SdW2_corrected = SdW2 / (1 - np.power(beta2, iteration))
    Sdb1_corrected = Sdb1 / (1 - np.power(beta2, iteration))
    Sdb2_corrected = Sdb2 / (1 - np.power(beta2, iteration))

    parameters["VdW1_c"] = VdW1_corrected
    parameters["VdW2_c"] = VdW2_corrected
    parameters["Vdb1_c"] = Vdb1_corrected
    parameters["Vdb2_c"] = Vdb2_corrected
    parameters["SdW1_c"] = SdW1_corrected
    parameters["SdW2_c"] = SdW2_corrected
    parameters["Sdb1_c"] = Sdb1_corrected
    parameters["Sdb2_c"] = Sdb2_corrected

    return parameters

def update_parameters(parameters, learning_rate):
    """
    Update parameters using Adam optimization algorithm
    """
    epsilon = 10e-05                        # used to prevent situations of division by 0

    for l in range(2):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (parameters["VdW" + str(l + 1) + "_c"] / np.sqrt(parameters["SdW" + str(l + 1) + "_c"] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (parameters["Vdb" + str(l + 1) + "_c"] / np.sqrt(parameters["Sdb" + str(l + 1) + "_c"] + epsilon))

    return parameters

def two_layer_model(X, Y, layers_dims, learning_rate=0.001, num_iterations=100000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->SIGMOID->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    costs = []                              # to keep track of the cost
    m = X.shape[1]                          # number of examples
    (n_x, n_h, n_y) = layers_dims

    # putting all functions together
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        parameters = forward_propagation(X, parameters)
        costt = cost(Y, parameters)
        parameters = back_propagation(X, Y, parameters)
        parameters = adam_optimizer(parameters, n_x, n_h, n_y, i+1)
        parameters = update_parameters(parameters, learning_rate)
        if print_cost and i % 3000 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(costt)))
        if print_cost and i % 3000 == 0:
            costs.append(costt)

    # visualising cost - this comes in handy for implementing "early stopping regularization" in the case of overfitting
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 3 thousands)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

def predict(X, y, parameters, train=False):
    """
    Predicts the results of the 2-layer neural network.
    """
    m = X.shape[1]
    p = np.zeros((1,m))

    parameters = forward_propagation(X, parameters)
    probas = parameters["A2"]                          # activation values
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:                          # threshold value
            p[0,i] = 1
        else:
            p[0,i] = 0
    if train == True:
        print("\nTrain Accuracy: "  + str(np.sum((p == y)/m)))
    else:
        print("\nTest Accuracy: " + str(np.sum((p == y)/m)))

    return p

class NN:
    '''
    X and Y are dataframes
    '''
    def fit(self, trainX, trainY, layers_dims):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        self.trainX = trainX
        self.trainY = trainY
        self.layers_dims = layers_dims

        # calling function two_layer_model() in fit
        parameters = two_layer_model(trainX, trainY, layers_dims = layers_dims, num_iterations = 30000, print_cost=True)

        return parameters

    def predict(self, X, y, parameters, train=False):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values
        pred is a list of the predicted value for df X
        """
        self.X = X
        self.y = y
        self.parameters = parameters
        self.train = train

        # calling function predict() in predict
        pred = predict(X, y, parameters, train=train)

        return pred

    def CM(self, y_test, y_test_obs):
        '''
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model
        '''
        self.y_test = y_test
        self.y_test_obs = y_test_obs

        for i in range(len(y_test_obs)):
            if(y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0
        cm = [[0,0],[0,0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp + 1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn + 1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp + 1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn + 1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)
        print("\nConfusion Matrix: ")
        print(cm)
        print("\n")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"F1 SCORE: {f1}")

# Main function
def main():
    # reading the dataset
    df = pd.read_csv("cleanedLBW.csv")

    # dropping redundant columns
    df.drop(df.columns[[0, 1]], axis = 1, inplace=True)

    # splitting the dataset into train and test sets
    X = df.iloc[:,:-1]
    y = df.Result

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.50, random_state=42)  # train-test split ratio is choosen to be 0.5 for implementation purposes

    # converting to numpy array
    trainX = np.array(train_x.T)
    testX = np.array(test_x.T)
    trainY = np.array(train_y.values.reshape((1, train_y.shape[0])))
    testY = np.array(test_y.values.reshape((1, test_y.shape[0])))

    n_x = trainX.shape[0]
    # n_h = int(input("Enter number of hidden units for 2 layer model: "))
    n_h = 17
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    parameters = {}
    obj = NN()                                  # initialising class object

    parameters = obj.fit(trainX, trainY, layers_dims)
    pred_train = obj.predict(trainX, trainY, parameters, train=True)
    pred_test = obj.predict(testX, testY, parameters)
    obj.CM(testY[0], pred_test[0])

if __name__ == '__main__': main()

####COMPLETED AT 12:58:00 17-11-2000####
####COPYRIGHT FOURSTRIPES CO.####
