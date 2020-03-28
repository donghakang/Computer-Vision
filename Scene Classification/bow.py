import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath



################################################################
#################### GLOBAL VARIABLE  ##########################
###############################z#################################
DATA_PATH = "./scene_classification_data"


TINY_IMG_SIZE = (16, 16)

KNN_K = 10

# BOW
SVM_C = 0.58

# VOCAB
KMEAN_N   = 5
KMEAN_MAX = 300

DIC_SIZE  = 200

BOW_STRIDE = 25
BOW_SIZE   = 25
################################################################
########################### HELPER  ############################
################################################################
# confusion matrix returns len(label_classes) x len(label_classes)
# that calculates the similarity between ground truth label and prediction
def confusion_matrix(prediction, ground_truth_label, label_classes):
    count = len(label_classes)
    confusion_matrix = np.zeros((count, count))

    if len(prediction) != len(ground_truth_label):
        print('Prediction != Ground_Truth_Label')
        exit(0)

    steps = 0
    while steps < len(prediction):
        i = ground_truth_label[steps]
        j = prediction[steps]
        confusion_matrix[i][j] += 1
        steps +=1

    return confusion_matrix

# accuracy measure prints the accuracy level.
def accuracy_measure(confusion_matrix):
    l = np.shape(confusion_matrix)[0]
    accuracy = 0
    for i in range (l):
        accuracy += confusion_matrix[i][i]

    return accuracy / np.sum(confusion_matrix)


# change number to label class name
def translate_dictionary(label_classes):
    dictionary = {}
    for label in label_classes:
        dictionary[label] = label_classes.index(label)
    return dictionary


# normalize the matrix
def normalize_matrix(matrix):
    avg = np.mean(matrix)
    u_l = np.sqrt(np.sum(matrix**2))
    m   = np.divide(matrix - avg, u_l)
    return m

# mean is not zero.
def normalize_matrix_bow(matrix):
    u_l = np.sqrt(np.sum(matrix**2))
    m   = np.divide(matrix, u_l)
    return m


def count_bow(m, dic_size):
    r_m = np.zeros(dic_size)
    for i in m:
        r_m[i] += 1
    return r_m


################################################################
########################### TODO:   ############################
################################################################

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list



def get_tiny_image(img, output_size):
    img_h, img_w = np.shape(img)
    out_w, out_h = output_size
    feature      = np.zeros(out_w*out_h).reshape(out_h, out_w)

    step_h = int(img_h/out_h)+1
    step_w = int(img_w/out_w)+1
    for h in range (out_h):
        for w in range (out_w):
            temp_img = img[h*step_h:(h+1)*step_h, w*step_w:(w+1)*step_w]
            feature[h][w] = np.sum(temp_img)/(step_h*step_w)
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(img)
    # f.add_subplot(1,2, 2)
    # plt.imshow(feature.astype(int))
    # plt.show(block=True)
    feature = normalize_matrix(feature)
    feature = np.reshape(feature, out_w*out_h)

    return feature

# KNN
def predict_knn(feature_train, label_train, feature_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_train, label_train)
    label_test_pred = neigh.predict(feature_test)

    return label_test_pred

# SVN
def predict_svm(feature_train, label_train, feature_test):
    c = SVM_C
    clf = LinearSVC(C=c)
    clf.fit(feature_train, label_train)
    score = clf.decision_function(feature_test)     # checks the score.
    label_test_pred = np.argmax(score, axis=1)      # index of best match

    return label_test_pred



def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    print('\n\n');
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++  CLASSIFYING KNN TINY  +++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    list_len = len(img_train_list)
    tiny_image_size = TINY_IMG_SIZE
    k = 10

    # Create train features
    feature_train = []
    count = 0
    for img_file in img_train_list:
        print("TINY: FEATURE TRAIN -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, tiny_image_size)
        feature_train.append(tiny_img)
        count += 1
        if (count == list_len):
            print("TINY: FEATURE TRAIN -- {}/{} -- {:.2f}%".format(list_len, list_len, 100*list_len/list_len), end='\n')

    feature_train = np.asarray(feature_train)
    print("TINY: FEATURE TRAIN COMPLETE")
    print('-------------------------------------------------')
    # Create test features
    feature_test = []
    count = 0
    for img_file in img_test_list:
        print("TINY: FEATURE TRAIN -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, tiny_image_size)
        feature_test.append(tiny_img)
        count += 1
        if (count == list_len):
            print("TINY: FEATURE TRAIN -- {}/{} -- {:.2f}%".format(list_len, list_len, 100*list_len/list_len), end='\n')

    feature_test = np.asarray(feature_test)
    print("TINY: FEATURE TEST COMPLETE")
    print('-------------------------------------------------')
    label_list = translate_dictionary(label_classes)
    label_train = np.array([label_list[i] for i in label_train_list])
    label_test  = np.array([label_list[i] for i in label_test_list])

    predicted_list = predict_knn(feature_train, label_train, feature_test, k)

    confusion = confusion_matrix(predicted_list, label_test, label_classes)
    accuracy = accuracy_measure(confusion)
    print('** TINY/KNN ACCURACY : {:.2f}%'.format(accuracy*100.0),end='\n\n')
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy





def compute_dsift(img, stride, size):
    # stride : is like steps,
    # size   : size of the cropped image.
    # img[i*stride : i*stride+size, j*stride : j*stride+size]

    img = cv2.imread(img, 0)

    img_h, img_w = np.shape(img)
    step_h = int((img_h - size)/stride)+1
    step_w = int((img_w - size)/stride)+1

    dense_feature = np.empty((0,128))

    sift = cv2.xfeatures2d.SIFT_create()
    for h in range (step_h):
        for w in range (step_w):
            local_patch = img[h*stride:(h*stride)+size, w*stride:(w*stride)+size]
            kp = [cv2.KeyPoint(x = w * stride + int(size/2), \
                               y = h * stride + int(size/2), \
                           _size = size)]                  # local_patch
            _, des = sift.compute(img, kp)
            dense_feature = np.append(dense_feature, des, axis=0)

    return dense_feature


def build_visual_dictionary (dense_feature_list, dic_size):
    print('++++++++++++++++SAVING VOCAB+++++++++++++++++++++')
    kmeans = KMeans(n_clusters = dic_size, \
                    n_init     = KMEAN_N,\
                    max_iter   = KMEAN_MAX).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_   # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    np.savetxt(DATA_PATH + "/vocab.txt", vocab)    # save the vocab.txt
    # print(vocab)
    print('+++++++++++++++++++ DONE!! ++++++++++++++++++++++')
    return kmeans.cluster_centers_


def compute_bow(feature, vocab):
    dic_size, _ = np.shape(vocab)
    y = np.arange(dic_size)
    neigh = KNeighborsClassifier(1)
    neigh.fit(vocab, y)
    prediction = neigh.predict(feature)
    bow_feature = normalize_matrix_bow(count_bow(prediction,dic_size))

    return bow_feature



def load_classify_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++    CLASSIFYING BOW    ++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    stride             = BOW_STRIDE
    size               = BOW_SIZE
    dense_feature_list = np.empty((0,128))
    dic_size           = DIC_SIZE
    list_len           = len(img_train_list)

    ## VOCAB Setup
    if os.path.isfile(DATA_PATH + '/vocab.txt'):
        print ("vocab file exists ...")
        vocab = np.loadtxt(DATA_PATH + '/vocab.txt')
    else:
        print('vocab file does not exists ...')
        count = 0
        for i in img_train_list:
            print("VOCAB         -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
            dense_feature = compute_dsift(i, stride, size)
            dense_feature_list = np.vstack((dense_feature_list, dense_feature))
            count += 1
            if (count == list_len):
                print("VOCAB         -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
        vocab = build_visual_dictionary (dense_feature_list, dic_size)
        print("VOCAB DICTIONARY COMPLETE")


    # feature train
    feature_train = np.empty((0,dic_size))
    print('BOW : FEATURE TRAIN')
    if os.path.isfile(DATA_PATH + '/feature_train_bow.txt'):
        print('LOAD DATA / feature_train_bow.txt')
        feature_train = np.loadtxt(DATA_PATH + '/feature_train_bow.txt')
    else:
        count = 0
        for img_file in img_train_list:
            print("BOW : FEATURE TRAIN -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
            dense_feature = compute_dsift(img_file, stride, size)
            feature = compute_bow(dense_feature, vocab)
            feature_train = np.vstack((feature_train, feature))
            count += 1
            if (count == list_len):
                print("BOW : FEATURE TRAIN -- {}/{} -- {:.2f}%".format(list_len, list_len, 100*list_len/list_len), end='\n')
        np.savetxt(DATA_PATH + '/feature_train_bow.txt', feature_train)
        print("BOW : FEATURE TRAIN COMPLETE")
        print('-------------------------------------------------')

    # feature test
    feature_test = np.empty((0,dic_size))
    print('BOW : FEATURE TEST')
    if os.path.isfile(DATA_PATH + '/feature_test_bow.txt'):
        print('LOAD DATA / feature_test_bow.txt')
        feature_test = np.loadtxt(DATA_PATH + '/feature_test_bow.txt')
    else:
        count = 0
        for img_file in img_test_list:
            print("BOW : FEATURE TEST  -- {}/{} -- {:.2f}%".format(count, list_len, 100*count/list_len), end='\r')
            dense_feature = compute_dsift(img_file, stride, size)
            feature = compute_bow(dense_feature, vocab)
            feature_test = np.vstack((feature_test, feature))
            count += 1
            if (count == list_len):
                print("BOW : FEATURE TEST  -- {}/{} -- {:.2f}%".format(list_len, list_len, 100*list_len/list_len), end='\n')
        np.savetxt(DATA_PATH + '/feature_test_bow.txt', feature_test)
        print("BOW : FEATURE TEST COMPLETE")
        print('-------------------------------------------------')

    label_list = translate_dictionary(label_classes)
    label_train = np.array([label_list[i] for i in label_train_list])
    label_test  = np.array([label_list[i] for i in label_test_list])

    return feature_train, feature_test, label_train, label_test


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    k = KNN_K

    feature_train, feature_test, label_train, label_test = load_classify_bow(label_classes, \
                                                                             label_train_list, \
                                                                             img_train_list, \
                                                                             label_test_list, \
                                                                             img_test_list)

    label_test_pred = predict_knn(feature_train, label_train, feature_test, k)
    confusion = confusion_matrix(label_test_pred, label_test, label_classes)
    accuracy  = accuracy_measure(confusion)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    print('** BOW/KNN ACCURACY : {:.2f}%'.format(accuracy*100.0),end='\n\n')
    return confusion, accuracy


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train, feature_test, label_train, label_test = load_classify_bow(label_classes, \
                                                                             label_train_list, \
                                                                             img_train_list, \
                                                                             label_test_list, \
                                                                             img_test_list)
    # TODO: delete  M.
    label_test_pred = predict_svm(feature_train, label_train, feature_test)
    confusion = confusion_matrix(label_test_pred, label_test, label_classes)
    accuracy  = accuracy_measure(confusion)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    print('** BOW/SVM ACCURACY : {:.2f}%'.format(accuracy*100.0),end='\n\n')
    return confusion, accuracy





def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()

def visualize_confusion_matrix_all(c1, a1, c2, a2, c3, a3, label_classes):
        # f = plt.figure()
        # f.add_subplot(1,2, 1)
        # plt.imshow(img)
        # f.add_subplot(1,2, 2)
        # plt.imshow(feature.astype(int))
        # plt.show(block=True)
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.title("TINY accuracy = {:.3f}".format(a1))
    plt.imshow(c1)
    # ax, fig = plt.gca(), plt.gcf()
    # plt.xticks(np.arange(len(label_classes)), label_classes)
    # plt.yticks(np.arange(len(label_classes)), label_classes)
    # # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # # avoid top and bottom part of heatmap been cut
    # ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()


    fig.add_subplot(1, 3, 2)
    plt.title("KNN accuracy = {:.3f}".format(a2))
    plt.imshow(c2)
    # ax, fig = plt.gca(), plt.gcf()
    # plt.xticks(np.arange(len(label_classes)), label_classes)
    # plt.yticks(np.arange(len(label_classes)), label_classes)
    # # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # # avoid top and bottom part of heatmap been cut
    # ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()


    fig.add_subplot(1, 3, 3)
    plt.title("SVM accuracy = {:.3f}".format(a3))
    plt.imshow(c3)
    # ax, fig = plt.gca(), plt.gcf()
    # plt.xticks(np.arange(len(label_classes)), label_classes)
    # plt.yticks(np.arange(len(label_classes)), label_classes)
    # # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # # avoid top and bottom part of heatmap been cut
    # ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    # ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")

    c1, a1 = classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    c2, a2 = classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    c3, a3 = classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    visualize_confusion_matrix(c1, a1, label_classes)
    visualize_confusion_matrix(c2, a2, label_classes)
    visualize_confusion_matrix(c3, a3, label_classes)

    visualize_confusion_matrix_all(c1, a1, c2, a2, c3, a3, label_classes)
