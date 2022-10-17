import os
import gzip
import numpy as np
from sklearn.utils import shuffle


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_data_with_n_labels_for_each_class(x_train_full, y_train_full, nr_of_labels, num_classes):
    x_train_full, y_train_full = shuffle(x_train_full, y_train_full)

    x_train = []
    y_train = []

    min_queries = nr_of_labels * num_classes
    x_train.extend(x_train_full[0:min_queries])
    y_train.extend(y_train_full[0:min_queries])

    for index in range(min_queries, y_train_full.size):
        x_train.append(x_train_full[index])
        y_train.append(y_train_full[index])
        _, classes_counter = np.unique(np.array(y_train), return_counts=True)
        if np.amin(classes_counter) == nr_of_labels:
            break

    # TODO select random 500 from each class
    return np.array(x_train), np.array(y_train), x_train_full[len(y_train):], y_train_full[len(y_train):]
