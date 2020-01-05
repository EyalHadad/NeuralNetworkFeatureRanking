import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random
from mnist import MNIST
import os
from operator import itemgetter


def initialize_parameters(layer_dims):
    # np.random.seed(3)
    default_params = {}
    normalize_param = 0.05
    layer_dims_length = len(layer_dims)  # number of layers in the network
    i = 1
    while i < layer_dims_length:
        w_name = 'W' + str(i)
        b_name = 'b' + str(i)
        w_value = random.randn(layer_dims[i], layer_dims[i - 1]) * normalize_param
        b_value = np.zeros((layer_dims[i], 1))
        default_params.update({w_name: w_value})
        default_params.update({b_name: b_value})
        i = i + 1

    return default_params


def linear_forward(A, W, b):
    Z = np.dot(W.T, A) + b
    local_cache = [A, W, b]
    return Z, local_cache


def sigmoid(z):
    A = 1 / (1 + np.exp(-z))
    return A, z.copy()


###need to return activation cache

def relu(z):
    A = np.maximum(z, 0)
    return A, z.copy()


###need to return activation cache

def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W.T, B)
    if activation == "relu":
        A, activation_cache = relu(Z)
    else:
        A, activation_cache = sigmoid(Z)

    linear_cache.append(activation)
    return A, (linear_cache, activation_cache)


def forward_propagation(X, parameters):
    globalCache = []

    A1, tmpLocalCache = linear_activation_forward(X, parameters['W1'], parameters['b1'], "relu")
    globalCache.append(tmpLocalCache)
    A2, tmpLocalCache = linear_activation_forward(A1, parameters['W2'], parameters['b2'], "relu")
    globalCache.append(tmpLocalCache)
    A3, tmpLocalCache = linear_activation_forward(A2, parameters['W3'], parameters['b3'], "relu")
    globalCache.append(tmpLocalCache)
    AL, tmpLocalCache = linear_activation_forward(A3, parameters['W4'], parameters['b4'], "sigmoid")
    globalCache.append(tmpLocalCache)
    return AL, globalCache


def compute_cost(last_activation, real_labels):
    len_al = len(last_activation)
    i = 0
    cost = 0
    while i < len_al:
        cost += (real_labels[i] * math.log10(last_activation[i])) + ((1 - real_labels[i]) * (1 - last_activation[i]))
        i = i + 1
    final_cost = -(cost / len_al)
    return final_cost


def relu_backward(dA, activation_cache):
    z = activation_cache
    dZ = np.array(dA, copy=True)
    # dZ=dZ*(dZ>0)

    z = np.asarray(z)[0]
    dZ[z <= 0] = 0
    return dZ


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    Z = np.asarray(Z)[0]
    s, f = relu(Z)
    dZ = dA * s * (1 - s)

    return dZ


def linear_backward(dZ, cache):
    # a, a2, a3 = cache
    a = cache[0]
    a2 = cache[1]
    a3 = cache[2]
    W = np.asarray(a2)
    b = np.asarray(a3)
    A_prev = np.asarray(a)
    m = A_prev.shape[1]
    dZ = np.asarray([dZ])[0]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, localCache, activation):
    linear_cache = localCache[0]
    activation_cache = localCache[1:4]

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def back_propagation(al, Y, caches):
    numOfPre = len(al[0])
    al = al[0]
    # Y = Y.reshape(al.shape)
    numOfLayers = len(caches)
    gradients = {}

    dAL = - (np.divide(Y, al) - np.divide(1 - Y, 1 - al))

    thisCache = caches[numOfLayers - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, thisCache, activation=thisCache[0][3])
    # gradients.append([dA_prev, dW, db])
    gradients["dA%s" % numOfLayers] = dA_prev
    gradients["dW%s" % numOfLayers] = dW
    gradients["db%s" % numOfLayers] = db

    for l in reversed(range(numOfLayers - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        thisCache = caches[l]
        dAL = gradients["dA%s" % (l + 2)]

        dA_prev, dW, db = linear_activation_backward(dAL, thisCache, activation=thisCache[0][3])
        gradients["dA%s" % (l + 1)] = dA_prev
        gradients["dW%s" % (l + 1)] = dW
        gradients["db%s" % (l + 1)] = db

        # gradients.append([dA_prev, dW, db])

    return gradients


def update_parameters(parameters, grads, learning_rate):
    parameters_length = len(parameters) // 2
    for l in range(1, parameters_length):
        parameters["W%s" % l] -= learning_rate * grads["dW%s" % l]
        parameters["b%s" % l] -= learning_rate * grads["db%s" % l]
    return parameters


def train_network(x_data, y_data, layers_dims, l_rate, n_iterations):
    parameters = initialize_parameters(layers_dims)
    costs = []
    for iter in range(0, n_iterations - 1):
        last_activation, caches = forward_propagation(x_data, parameters)
        cost = compute_cost(last_activation[0], y_data)
        grads = back_propagation(last_activation, y_data, caches)
        parameters = update_parameters(parameters, grads, l_rate)
        if iter % 100 == 0:
            costs.append(cost)
    print("the costs:")
    print(costs)
    return parameters, costs


def Predict_tmp(X, Y, parameters):
    m = len(X)
    n = len(parameters) / 2  # number of layers in the neural network

    # Forward propagation
    probas, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions

    if probas[0] > 0.5:
        p = 1
    else:
        p = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    acc = str(np.sum((p == Y)))
    print("Accuracy: " + acc)

    return acc


def Predict(X, Y, parameters):
    preds, caches = forward_propagation(X, parameters)
    preds = np.round(preds[0])
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y_actual = Y

    for i in range(len(preds)):
        if y_actual[i] == preds[i] == 1:
            TP += 1
        if preds[i] == 1 and y_actual[i] != preds[i]:
            FP += 1
        if y_actual[i] == preds[i] == 0:
            TN += 1
        if preds[i] == 0 and y_actual[i] != preds[i]:
            FN += 1

    return (TP + TN) / (TP + TN + FP + FN)


def getTrainTestData(mndata, first_digit, second_digit):
    train_images, train_images_labels = mndata.load_training()
    train_labels, train_data = get_relevant_data(train_images, train_images_labels, first_digit, second_digit)

    test_images, test_images_labels = mndata.load_testing()
    test_labels, test_data = get_relevant_data(test_images, test_images_labels, first_digit, second_digit)

    return train_labels, train_data, test_labels, test_data


def get_relevant_data(images, labels, first_digit, second_digit):
    arr = np.array(labels)
    index1 = np.where(arr == first_digit)[0]
    index2 = np.where(arr == second_digit)[0]
    both_indexes = np.append(index1, index2)
    getter = itemgetter(*both_indexes)
    both_images = getter(images)
    new_labels = arr[both_indexes]
    new_labels[new_labels == 3] = 0
    new_labels[new_labels == 8] = 1
    return new_labels, np.asarray(both_images).T


if __name__ == '__main__':
    # Get the data
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    mndata = MNIST(os.path.join(dname, 'samples'))
    # filter the relevant data by 2 specific digits
    train_labels, train_data, test_labels, test_data = getTrainTestData(mndata, first_digit=3, second_digit=8)

    network_layers = [784, 20, 7, 5, 1]
    finalParam, costs_1 = train_network(train_data, train_labels, network_layers, l_rate=0.001, n_iterations=3000)

    accuracy38_train = Predict(train_data, train_labels, finalParam)
    accuracy38_test = Predict(test_data, test_labels, finalParam)
    print(accuracy38_train)
    print(accuracy38_test)


