import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


## Helper
# create array with a number
def label_train_y (label_train):
    arr = np.zeros(10)
    arr[label_train] = 1
    return arr

def create_batch_array(data_size, batch_size):
    if (data_size / batch_size).is_integer():
        return np.random.permutation(data_size)
    else:
        a = np.arange(data_size)
        num = ((int(data_size / batch_size) + 1) * batch_size) - data_size
        additional_array = np.random.choice(data_size, num)
        return np.append(np.random.permutation(data_size), additional_array)

def get_mini_batch(im_train, label_train, batch_size):
    im_size, data_size = im_train.shape
    permuted_pos  = create_batch_array(data_size, batch_size)

    mini_batch_x = []
    mini_batch_y = []

    in_batch_x = []
    in_batch_y = []
    for i, permute in enumerate (permuted_pos):
        if i % batch_size == 0:
            in_batch_x = []
            in_batch_y = []
            if i != 0:
                mini_batch_x.append(in_batch_x)
                mini_batch_y.append(in_batch_y)
        in_batch_x.append(im_train[:,permute])
#         print(label_train[:,permute])
        in_batch_y.append(label_train_y(label_train[:,permute]))
    mini_batch_x.append(in_batch_x)
    mini_batch_y.append(in_batch_y)


    mini_batch_x = np.asarray(mini_batch_x)
    mini_batch_y = np.asarray(mini_batch_y)

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # x = m X 1
    # w = n X m
    # b = n X 1
    # wx+b = n X m * m X 1 + n X 1 = n X 1
    Wx = np.matmul(w, x)
    y  = Wx + b
    return y

def fc_backward(dl_dy, x, w, b, y):
    # dl_dx = dl_dy * dy_dx
    # dl_dw = dl_dy * dy_dw
    # dl_db = dl_dy * dy_db
    dl_dx = dl_dy @ w                 # 1 x m
    dl_dw = dl_dy * x                 # 1 x (n X m)
    dl_db = dl_dy * 1                 # 1 x n
    return dl_dx, dl_dw, dl_db

def loss_euclidean(y_tilde, y):
    # y_tilde -->
    l     = np.sum((y - y_tilde) ** 2)
    dl_dy = -2 * (y - y_tilde)

    return l, dl_dy



def loss_cross_entropy_softmax(x, y):
    '''
    x -- m X 1 --> no reshape needed
    y -- m     (0,1)
    '''
    y_tilde = np.exp(x) / np.sum(np.exp(x))
    l       = np.sum(y * np.log(y_tilde))

    dl_dy = y_tilde - y
    # d_dx f = f(1-f)


    return l, dl_dy

def relu(x):
    # TO DO
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    print('\n')
    print("===================================================")
    num_of_batches, batch_size, _ = mini_batch_y.shape
    learning_rate = 0.1
    decay_rate    = 0.5       # (0, 1]
    w = np.random.random_sample((10, 196))
    w = np.array(w, dtype=np.float32)
    b = np.random.random_sample((10, 1))
    b = np.array(b, dtype=np.float32)

    k = 0                    # batch position
    nIters = 10000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw = 0
        dL_db = 0

        # for each image xi in kth mini-batch ...
        for i in range (batch_size):
            x = mini_batch_x[k,i,:].reshape(-1,1)                  # x       shape = m X 1
            y_tilde  = fc(x, w, b).reshape(-1)                     # y_tilde shape = n
            y        = mini_batch_y[k,i,:]                         # y       shape = n
            l, dl_dy = loss_euclidean(y_tilde, y)
            dl_dy = np.transpose(dl_dy.reshape(-1,1))              # dl_dy   shape = 1 x n
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)   # dl_dx   shape = 1 x m
                                                                   # dl_dw   shape = m x n -- 1 x (n x m)
                                                                   # dl_db   shape = 1 x n
            dL_dw += dl_dw
            dL_db += dl_db
        k += 1
        if k == num_of_batches:
            k = 0
        w = w - (learning_rate / batch_size * dL_dw.transpose())
        b = b - (learning_rate / batch_size * dL_db.transpose())

        print("SLP_LINEAR == Loading : {:.2f}%".format(iIter/nIters * 100), end='\r')
    print("SLP_LINEAR == Loading : 100.00%", end='\n')
    return w, b


def train_slp(mini_batch_x, mini_batch_y):
    print('\n')
    print("===================================================")
    num_of_batches, batch_size, _ = mini_batch_y.shape
    learning_rate = 0.1
    decay_rate    = 0.5       # (0, 1]
    w = np.random.random_sample((10, 196))
    w = np.array(w, dtype=np.float32)
    b = np.random.random_sample((10, 1))
    b = np.array(b, dtype=np.float32)

    k = 0                    # batch position
    nIters = 10000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw = 0
        dL_db = 0

        # for each image xi in kth mini-batch ...
        for i in range (batch_size):
            x = mini_batch_x[k,i,:].reshape(-1,1)                  # x       shape = m X 1
            y_tilde  = fc(x, w, b).reshape(-1)                     # y_tilde shape = n
            y        = mini_batch_y[k,i,:]                         # y       shape = n
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            dl_dy = np.transpose(dl_dy.reshape(-1,1))              # dl_dy   shape = 1 x n
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)   # dl_dx   shape = 1 x m
                                                                   # dl_dw   shape = m x n -- 1 x (n x m)
                                                                   # dl_db   shape = 1 x n
            dL_dw += dl_dw
            dL_db += dl_db
        k += 1
        if k == num_of_batches:
            k = 0
        w = w - (learning_rate / batch_size * dL_dw.transpose())
        b = b - (learning_rate / batch_size * dL_db.transpose())

        print("    SLP    == Loading : {:.2f}%".format(iIter/nIters * 100), end='\r')
    print("    SLP    == Loading : 100.00%", end='\n')
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
#    main.main_mlp()
#    main.main_cnn()
