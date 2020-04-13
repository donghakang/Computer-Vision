import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

## Helper

# change y value (number between 0, 9) to array (1x10) that is all 0 except yth element being 1.
def label_train_y (label_train):
    arr = np.zeros(10)
    arr[label_train] = 1
    return arr

# mixes all the data of hand-written data. This will be used to generate batch of data.
def create_batch_array(data_size, batch_size):
    if (data_size / batch_size).is_integer():
        return np.random.permutation(data_size)
    else:
        a = np.arange(data_size)
        num = ((int(data_size / batch_size) + 1) * batch_size) - data_size
        additional_array = np.random.choice(data_size, num)
        return np.append(np.random.permutation(data_size), additional_array)

# generates batch of train data.
# Generates batch to implement Stochastic gradient descennt based neural network.
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
        in_batch_y.append(label_train_y(label_train[:,permute]))
    mini_batch_x.append(in_batch_x)
    mini_batch_y.append(in_batch_y)

    mini_batch_x = np.asarray(mini_batch_x)
    mini_batch_y = np.asarray(mini_batch_y)

    return mini_batch_x, mini_batch_y

# rearranges image array into columns
def im2col(A, size):
    H, W = A.shape
    steps_h = H - size + 1
    steps_w = W - size + 1
    im = []
    for h in range (steps_h):
        for w in range (steps_w):
            im.append(np.reshape(A[h:h+size, w:w+size], -1))
    im = np.transpose(np.asarray(im))

    return im















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

    l     = np.sum((y - y_tilde) ** 2)
    dl_dy = -2 * (y - y_tilde)
    dl_dy = np.transpose(dl_dy.reshape(-1,1))

    return l, dl_dy



def loss_cross_entropy_softmax(x, y):
    # x -- m X 1 --> no reshape needed
    # y -- m     (0,1)

    y_tilde = np.exp(x) / np.sum(np.exp((x)))
    y_tilde = np.array(y_tilde, dtype = np.float64)
    l       = np.sum(y * np.log(y_tilde))

    dl_dy = y_tilde - y
    dl_dy = np.transpose(dl_dy.reshape(-1,1))

    return l, dl_dy

def relu(x):
    epsilon = 0.001
    # leaky ReLu
    y = np.where(x < epsilon * x, epsilon * x, x)
    return y


def relu_backward(dl_dy, x, y):
    # dl_dy -- 1 x z
    # y     -- z
    # dl_dx -- 1 x z = dl_dy * dy_dx
    epsilon = 0.001
    dy_dx = np.where(y < 0, 0, 1)
    dl_dx = dl_dy * dy_dx
    return dl_dx



def conv(x, w_conv, b_conv):
    #     x   -- H X W X C1        --> 14 x 14 x 1
    #     w_c -- h X w X C1 X C2   --> 3 x 3 x 1 x 3
    #     b_c -- C2 X 1            --> 3 x 1
    X_H, X_W, C1 = x.shape
    c_h, c_w, C1, C2 = w_conv.shape

    y = np.zeros((X_H, X_W, C2))

    for c1 in range (C1):
        for c2 in range (C2):
            X = np.pad(x[:,:,c1], ((1,1),(1,1)))
            X = im2col(X, c_h)                     # horizontal first
            X = np.transpose(X)                    # 196 x 9

            w_ = w_conv[:,:,c1,c2]
            w_ = np.reshape(w_, (-1, 1))

            wx = np.dot(X, w_)                     # 196 x 1
            wx = wx.reshape(X_H, X_W)

            y[:,:,c2] += wx + b_conv[c2, 0]

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # dl_dy.shape  ==  y.shape  == (H , W, C2)
    c_h, c_w, C1, C2 = w_conv.shape
    H, W, _      = dl_dy.shape

    dl_dw = np.copy(w_conv)
    dl_dw.fill(0)          # set default array
    dl_db = np.copy(b_conv)
    dl_db.fill(0)

    # bias
    for b in range(C2):
        dy_db = dl_dy[:,:,b]
        dl_db[b,0] = np.sum(dy_db)

    # weight
    for d in range (C2):
        dL_dy = dl_dy[:,:,d].reshape(-1)
        for c in range (C1):
            X = np.pad(x[:,:,c], ((1,1), (1,1)))
            X = im2col(X, c_h)
            dy_dw = np.transpose(X)
            dl_dw[:,:,c,d] = np.dot(dL_dy, dy_dw).reshape(c_h, c_w)

    return dl_dw, dl_db

def pool2x2(x):
    # x  - 14 x 14 x C
    # y  - 7 x 7 x C
    #
    # max_pooling
    H, W, C = x.shape
    y = np.empty((int(H/2), int(W/2), C))

    step = 2 # since H --> H/2
    for c in range (C):
        for h in range(int(H/2)):
            for w in range(int(W/2)):
                y[h,w,c] = np.max(x[step*h:step*h+step , step*w:step*w+step, c])

    return y



def pool2x2_backward(dl_dy, x, y):
    # x  - 14 x 14 x C
    # y  - 7 x 7 x C
    # dl_dy - 7 x 7 x C
    # dl_dx - 14 x 14 x C
    POOL_SIZE = 2

    dl_dx = np.copy(x)
    dl_dx.fill(0)          # set default array

    y_h, y_w, C = y.shape
    for c in range (C):
        for h in range (y_h):
            for w in range (y_w):
                target_x = x[POOL_SIZE*h:POOL_SIZE*h+POOL_SIZE, POOL_SIZE*w:POOL_SIZE*w+POOL_SIZE, c]
                max_val = y[h,w,c]              # pool2x2 value
                max_pos = np.argwhere(target_x == max_val)
                if max_pos.shape[0] > 1:        # There are same number maximum values in the target area.
                    max_pos = max_pos[0]        # save only one of the data.
                else:
                    max_pos = max_pos.reshape(-1)
                max_pos[0] += POOL_SIZE * h
                max_pos[1] += POOL_SIZE * w

                dl_dx[max_pos[0], max_pos[1], c] = dl_dy[h, w, c]
    return dl_dx




def flattening(x):
    # x -- 14 x 14 x C      --> column major
    H, W, C = x.shape
    y = np.array([])
    for c in range (C):
        x_ = x[:,:,c]
        y = np.append(y, x_.reshape(-1, order='F'))

    return y.reshape(-1,1)



def flattening_backward(dl_dy, x, y):
    h, w, c = x.shape

    dl_dy_split = np.split(dl_dy, c)    # split dl_dy by c.
    dl_dx = np.empty((h, w, c))
    for count, dl_dx_n in enumerate(dl_dy_split):
        dl_dx[:,:,count] = dl_dx_n.reshape((h, w), order='F')
    return dl_dx




def train_slp_linear(mini_batch_x, mini_batch_y):
    print('\n')
    print("===================================================")
    num_of_batches, batch_size, _ = mini_batch_y.shape
    learning_rate = 0.01
    decay_rate    = 0.9       # (0, 1]

    w = np.random.randn(10,196)*np.sqrt(2/196)              # He-et-al Initialization
    b = np.random.randn(10,1)*np.sqrt(2/1)

    k = 0                                                   # batch position
    nIters = 16000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw = 0
        dL_db = 0
        for i in range (batch_size):
            x = mini_batch_x[k,i,:].reshape(-1,1)                  # x       shape = m X 1
            y_tilde  = fc(x, w, b).reshape(-1)                     # y_tilde shape = n
            y        = mini_batch_y[k,i,:]                         # y       shape = n

            l, dl_dy = loss_euclidean(y_tilde, y)                  # dl_dy   shape = 1 x n
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
    learning_rate = 0.2
    decay_rate    = 0.9       # (0, 1]

    w = np.random.randn(10,196)*np.sqrt(2/196)
    b = np.random.randn(10,1)*np.sqrt(2/1)


    k = 0                     # batch position
    nIters = 16000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw = 0
        dL_db = 0

        for i in range (batch_size):
            x = mini_batch_x[k,i,:].reshape(-1,1)                  # x       shape = m X 1
            y_tilde  = fc(x, w, b).reshape(-1)                     # y_tilde shape = n
            y        = mini_batch_y[k,i,:]                         # y       shape = n
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)      # dl_dy   shape = 1 x n
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
    print('\n')
    print("===================================================")
    num_of_batches, batch_size, _ = mini_batch_y.shape
    learning_rate = 0.2
    decay_rate    = 0.9       # (0, 1]

    w1 = np.random.randn(30,196)*np.sqrt(2/196)
    b1 = np.random.randn(30,1)*np.sqrt(2/1)
    w2 = np.random.randn(10,30)*np.sqrt(2/30)
    b2 = np.random.randn(10,1)*np.sqrt(2/1)


    k = 0                    # batch position
    nIters = 16000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw1 = 0
        dL_db1 = 0
        dL_dw2 = 0
        dL_db2 = 0

        for i in range (batch_size):
            x1 = mini_batch_x[k,i,:].reshape(-1,1)                  # x1 : 196 x 1

            y1 = fc(x1, w1, b1)                                     # y1 : 30 x 1
            y1_relu = relu(y1)
            y2 = fc(y1_relu, w2, b2)                                # y2 : 10 x 1
            y2_relu = relu(y2)

            y_tilde  = y2_relu.reshape(-1)                          # y_tilde : 10
            y        = mini_batch_y[k,i,:]                          # y       : 10
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)       # dl_dy   shape = 1 x 10

            dl_dx2_relu = relu_backward(dl_dy, y2.reshape(-1), y)
            dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dx2_relu, y1, w2, b2, y2_relu)
            dl_dx1_relu = relu_backward(dl_dx2, y1.reshape(-1), y1_relu.reshape(-1))
            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dx1_relu, x1, w1, b1, y1_relu)

            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2

        k += 1
        if k == num_of_batches:
            k = 0
        w1 = w1 - (learning_rate / batch_size * dL_dw1.transpose())
        b1 = b1 - (learning_rate / batch_size * dL_db1.transpose())
        w2 = w2 - (learning_rate / batch_size * dL_dw2.transpose())
        b2 = b2 - (learning_rate / batch_size * dL_db2.transpose())

        print("    MLP    == Loading : {:.2f}%".format(iIter/nIters * 100), end='\r')
    print("    MLP    == Loading : 100.00%", end='\n')
    return w1, b1, w2, b2



def train_cnn(mini_batch_x, mini_batch_y):
    print('\n')
    print("===================================================")
    num_of_batches, batch_size, _ = mini_batch_y.shape
    # assume that image size is 14x14
    im_size_h, im_size_w = (14, 14)
    learning_rate = 0.2
    decay_rate    = 0.9       # (0, 1]

    w_conv = np.random.randn(3,3,1,3)*np.sqrt(2/196)
    b_conv = np.random.randn(3,1)*np.sqrt(2/196)
    w_fc   = np.random.randn(10, 147)*np.sqrt(2/196)
    b_fc   = np.random.randn(10,1)*np.sqrt(2/196)


    k = 0                    # batch position
    nIters = 16000
    for iIter in range (nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dL_dw_conv = 0
        dL_db_conv = 0
        dL_dw_fc = 0
        dL_db_fc = 0

        for i in range (batch_size):
            x = mini_batch_x[k,i,:]
            x = x.reshape((im_size_h, im_size_w, 1), order='F') # 14x14x1
            pred1 = conv(x, w_conv, b_conv)        # pred1 = 14x14x3
            pred2 = relu(pred1)                    # pred2 = 14x14x3
            pred3 = pool2x2(pred2)                 # pred3 = 7 x7 x3
            pred4 = flattening(pred3)              # pred4 = 147 x 1
            pred5 = fc(pred4, w_fc, b_fc)          # pred5 = 10 x 1
            pred6 = relu(pred5)                    # pred6 = 10 x 1

            y_tilde  = pred5.reshape(-1)                            # y_tilde : 10,
            y        = mini_batch_y[k,i,:]                          # y       : 10,
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)       # dl_dy   shape = 1 x 10

            dl_dx1 = relu_backward(dl_dy, pred5, y)  # dl_dx = 1 x 10
            dl_dx2, dl_dw_fc, dl_db_fc = fc_backward(dl_dx1, pred4, w_fc, b_fc, pred5)
                                                    # dl_dx2   = 1 x 147
                                                    # dl_dw_fc = 147 x 10
                                                    # dl_db_fc = 1 x 10
            dl_dx3 = flattening_backward(dl_dx2.reshape(-1,1), pred3, pred4) # dl_dx3 = 7 x 7 x 3
            dl_dx4 = pool2x2_backward(dl_dx3, pred2, pred3)    # dl_dx4 = 14 x 14 x 3
            dl_dx5 = relu_backward(dl_dx4, pred1, pred2)       # dl_dx5 = 14 x 14 x 3
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx5, x, w_conv, b_conv, pred1)


            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc

        k += 1
        if k == num_of_batches:
            k = 0
        w_conv = w_conv - (learning_rate / batch_size * dL_dw_conv)
        b_conv = b_conv - (learning_rate / batch_size * dL_db_conv)
        w_fc   = w_fc - (learning_rate / batch_size * dL_dw_fc.transpose())
        b_fc   = b_fc - (learning_rate / batch_size * dL_db_fc.transpose())

        print("    CNN    == Loading : {:.2f}%".format(iIter/nIters * 100), end='\r')
    print("    CNN    == Loading : 100.00%", end='\n')
    return w_conv, b_conv, w_fc, b_fc



if __name__ == '__main__':
    c1, a1 = main.main_slp_linear()
    c2, a2 = main.main_slp()
    c3, a3 = main.main_mlp()
    c4, a4 = main.main_cnn()

    main.visualize_confusion_matrix_all(c1, a1, c2, a2, c3, a3, c4, a4)
