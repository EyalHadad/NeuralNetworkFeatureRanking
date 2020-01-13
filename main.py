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
    a = 1 / (1 + np.exp(-z))
    return a, z.copy()


###need to return activation cache

def relu(z):
    a = np.maximum(z, 0)
    return a, z.copy()


###need to return activation cache

def linear_activation_forward(a_prev, w, b, activation):
    z, linear_cache = linear_forward(a_prev, w.T, b)
    if activation == "relu":
        a, activation_cache = relu(z)
    else:
        a, activation_cache = sigmoid(z)

    linear_cache.append(activation)
    return a, (linear_cache, activation_cache)


def forward_propagation(x, parameters):
    global_cache = []

    a1, local_cache = linear_activation_forward(x, parameters['W1'], parameters['b1'], "relu")
    global_cache.append(local_cache)
    a2, local_cache = linear_activation_forward(a1, parameters['W2'], parameters['b2'], "relu")
    global_cache.append(local_cache)
    a3, local_cache = linear_activation_forward(a2, parameters['W3'], parameters['b3'], "relu")
    global_cache.append(local_cache)
    a_l, local_cache = linear_activation_forward(a3, parameters['W4'], parameters['b4'], "sigmoid")
    global_cache.append(local_cache)
    return a_l, global_cache


def compute_cost(last_activation, real_labels):
    len_al = len(last_activation)
    i = 0
    cost = 0
    while i < len_al:
        cost += (real_labels[i] * math.log10(last_activation[i])) + ((1 - real_labels[i]) * (1 - last_activation[i]))
        i = i + 1
    final_cost = -(cost / len_al)
    return final_cost


def relu_derivative(d_a, activation_cache):
    z = activation_cache[0]
    d_z = np.array(d_a, copy=True)
    d_z[z <= 0] = 0
    return d_z


def sigmoid_derivative(d_a, activation_cache):
    Z = activation_cache
    Z = np.asarray(Z)[0]
    s, f = relu(Z)
    dZ = d_a * s * (1 - s)

    return dZ


# def sigmoid_derivative(d_a, activation_cache):
#
# z = activation_cache[0]
# s, f = sigmoid(z)
# d_z = d_a * s * (1 - s)
# return d_z


def linear_backward(d_z, cache):
    a = cache[0]
    a2 = cache[1]
    a3 = cache[2]
    w = np.asarray(a2)
    b = np.asarray(a3)
    a_prev = np.asarray(a)
    m = a_prev.shape[1]
    d_z = np.asarray([d_z])[0]

    d_w = 1. / m * np.dot(d_z, a_prev.T)
    d_b = 1. / m * np.sum(d_z, axis=1, keepdims=True)
    d_a_prev = np.dot(w, d_z)

    return d_a_prev, d_w, d_b


def linear_activation_backward(d_a, local_cache, activation):
    linear_cache = local_cache[0]
    activation_cache = local_cache[1:4]

    if activation == "relu":
        d_z = relu_derivative(d_a, activation_cache)
    else:
        d_z = sigmoid_derivative(d_a, activation_cache)

    d_a_prev, d_w, db = linear_backward(d_z, linear_cache)
    return d_a_prev, d_w, db


def back_propagation(al, y, caches):
    # numOfPre = len(al[0])
    al = al[0]
    # Y = Y.reshape(al.shape)
    num_of_layers = len(caches)
    gradients = {}

    d_al = - (np.divide(y, al) - np.divide(1 - y, 1 - al))

    current_cache = caches[num_of_layers - 1]
    d_a_prev, d_w, db = linear_activation_backward(d_al, current_cache, activation=current_cache[0][3])
    # gradients.append([dA_prev, dW, db])
    gradients["dA%s" % num_of_layers] = d_a_prev
    gradients["dW%s" % num_of_layers] = d_w
    gradients["db%s" % num_of_layers] = db

    for l in reversed(range(num_of_layers - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        d_al = gradients["dA%s" % (l + 2)]
        d_a_prev, d_w, db = linear_activation_backward(d_al, current_cache, activation=current_cache[0][3])
        gradients["dA%s" % (l + 1)] = d_a_prev
        gradients["dW%s" % (l + 1)] = d_w
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
    costs,ranks = [], []
    print("the costs:")
    for n_iter in range(0, n_iterations - 1):
        last_activation, caches = forward_propagation(x_data, parameters)
        cost = compute_cost(last_activation[0], y_data)
        grads = back_propagation(last_activation, y_data, caches)
        parameters = update_parameters(parameters, grads, l_rate)
        rank = rank1_formula(parameters)

        if n_iter % 100 == 0:
            print(cost)
            costs.append(cost)
            ranks.append(rank)

    return parameters, costs, ranks


def predict(x, y, parameters):
    predicted, caches = forward_propagation(x, parameters)
    predicted = np.round(predicted[0])
    t_p, f_p, t_n, f_n = 0, 0, 0, 0
    y_actual = y

    for i in range(len(predicted)):
        if y_actual[i] == predicted[i] == 1:
            t_p += 1
        if predicted[i] == 1 and y_actual[i] != predicted[i]:
            f_p += 1
        if y_actual[i] == predicted[i] == 0:
            t_n += 1
        if predicted[i] == 0 and y_actual[i] != predicted[i]:
            f_n += 1

    return (t_p + t_n) / (t_p + t_n + f_p + f_n)


def split_data(f_mndata, first_digit, second_digit):
    train_images, train_images_labels = f_mndata.load_training()
    f_train_labels, f_train_data = get_relevant_data(train_images, train_images_labels, first_digit, second_digit)

    test_images, test_images_labels = f_mndata.load_testing()
    f_test_labels, f_test_data = get_relevant_data(test_images, test_images_labels, first_digit, second_digit)

    return f_train_labels, f_train_data, f_test_labels, f_test_data


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


def rank1_formula(final_params):
    relevant_keys = ['W1', 'W2', 'W3', 'W4']
    rank_dict = {'W1': [], 'W2': [], 'W3': [], 'W4': []}
    for key in relevant_keys:
        a_len = final_params[key].T.shape[0]
        for ind in range(a_len):
            new_key = key + str(ind)
            new_val = np.absolute(final_params[key].T[ind]).mean()
            rank_dict[key].append((new_key, new_val))

    return rank_dict


def factor_ranks(costs_new, ranks):
    relevant_keys = ['W1', 'W2', 'W3', 'W4']
    for i in range(len(ranks)):
        for key in relevant_keys:
            ranks[i][key] = [y * costs_new[i] for x, y in ranks[i]['W1']]
    return ranks


def calculate_final_rank(costs, ranks):
    i=6
    print(i)
    costs_new = costs_to_prob_lists(costs)
    factored_ranks = factor_ranks(costs_new, ranks)
    print(i)

    pass


def costs_to_prob_lists(costs):
    costs_abs = [abs(number) for number in costs]
    prob_factor = 1 / sum(costs_abs)
    return [prob_factor * p for p in costs_abs]


if __name__ == '__main__':
    # Get the data
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    mndata = MNIST(os.path.join(dname, 'samples'))
    # filter the relevant data by 2 specific digits
    train_labels, train_data, test_labels, test_data = split_data(mndata, first_digit=3, second_digit=8)

    network_layers = [784, 20, 7, 5, 1]
    finalParam, costs_1, ranks_1 = train_network(train_data, train_labels, network_layers, l_rate=0.001, n_iterations=300)
    final_rank = calculate_final_rank(costs_1, ranks_1)

    accuracy_train = predict(train_data, train_labels, finalParam)
    accuracy_test = predict(test_data, test_labels, finalParam)
    print(accuracy_train)
    print(accuracy_test)
