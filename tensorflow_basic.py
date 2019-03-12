import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score


def read_dataset():
    # Reading the dataset using panda's dataframe
    df = pd.read_csv("input.csv")
    X = df[df.columns[0:4]].values
    y = df[df.columns[4]]

    # Encode the dependent variable
    Y = one_hot_encode(y)

    return X, Y


# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# for k-fold cross validation
def make_dataset(X_data, y_data, n_splits):

    def gen():
        for train_index, test_index in KFold(n_splits=n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            print('Train: %s | test: %s' % (train_index, test_index))
            yield X_train, y_train, X_test, y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))


# Read the dataset
X, Y = read_dataset()
# print(X, Y)
dataset = make_dataset(X, Y, 3)
#print(dataset)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(dataset))

