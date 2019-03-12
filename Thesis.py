import matplotlib.pyplot as plt
import sklearn
import sklearn as sk
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, KFold

# ongoing:
# metrics:
# 1. accuracy
# 2. f1-score
# 3. false-positive
# 4. false-negative
# 5. roc curve
# with datasets of different source
# change learning rate, epoch, k-fold, hidden_layer and no. of neurons in hidden layer

# parameters for DGA botnet detection
feature_no = 6
'''no_of_hidden_layers = int(input())
hidden_layer_neurons = []
for i in range(0, no_of_hidden_layers):
    hidden_layer_neurons.append(int(input()))'''


def Thesis(file_path, name):
    def read_dataset():
        # Reading the dataset using panda's dataframe
        df = pd.read_csv(file_path)
        X = df[df.columns[0:feature_no]].values
        y = df[df.columns[feature_no]]
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


    # Read the dataset
    X, Y = read_dataset()

    # Shuffle the dataset to mix up the rows.
    X, Y = shuffle(X, Y, random_state=20)

    # X, Y = X[0:10000], Y[0:10000]

    # Convert the dataset into train and test part
    final_train_x, final_test_x, final_train_y, final_test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

    # train_x, test_x, train_y, test_y = [], [], [], []
    # Inspect the shape of the training and testing.
    '''print(train_x.shape)
    print(train_y.shape)
    # print()
    print(test_x.shape)
    print(test_y.shape)'''

    # Define the important parameters and variable to work with the tensors
    learning_rate = 0.6
    training_epochs = 150
    # cost_history = np.empty(shape=[1], dtype=float)
    n_dim = X.shape[1]
    n_class = 2
    model_path = "/home/ashiq/PycharmProjects/TensorFlow_MLP_test/Graphs/graph"

    # Define the number of hidden layers and number of neurons for each layer
    n_hidden_1 = 50  # hidden_layer_neurons[0]
    n_hidden_2 = 40
    n_hidden_3 = 30
    n_hidden_4 = 20

    # x = value of feature vectors, y_ = vector of actual output values
    x = tf.placeholder(tf.float32, [None, n_dim])
    y_ = tf.placeholder(tf.float32, [None, n_class])

    # Define the model
    def multilayer_perceptron(x, weights, biases):

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden layer with sigmoid activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        # Hidden layer with sigmoid activation
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)

        # Hidden layer with RELU activation
        layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        layer_4 = tf.nn.sigmoid(layer_4)

        # Output layer with linear activation
        out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    # Define the weights and the biases for each layer

    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
    }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_class]))
    }

    # Initialize all the variables

    # Call your model defined
    y = multilayer_perceptron(x, weights, biases)

    # Define the cost function and optimizer
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0  # Choose an appropriate one.
    cost_function = tf.losses.mean_squared_error(labels=y_, predictions=y) + reg_constant * sum(reg_losses)
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def run_train(sess, train_x, train_y, test_x, test_y, flag, i):
        sess.run(init)
        # Calculate the cost and the accuracy for each epoch
        # print(train_x, test_x)
        mse_history = []
        accuracy_history = []
        f1_history = []
        # cost_history = []
        for epoch in range(training_epochs):
            sess.run(training_step, feed_dict={x: train_x, y_: train_y})
            cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
            # cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
            pred_y = sess.run(y, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)
            accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
            accuracy_history.append(accuracy)

            y_true = np.argmax(test_y, 1)  # 1 0 -> 0 , 0 1 -> 1
            y_pred = np.argmax(pred_y, 1)  # max min -> 0 , min max -> 1
            f1_score = sk.metrics.f1_score(y_true, y_pred)
            print('fold: ', i, "F1-score", f1_score)
            f1_history.append(f1_score)

            print('fold: ', i, 'epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        # Plot Accuracy Graph
        if flag == 1:
            plt.plot(accuracy_history, label='%s' % name)
            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy')
            plt.show()

            plt.plot(f1_history, label='%s' % name)
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.show()

            plt.plot(mse_history, label='%s' % name)
            plt.xlabel('Epoch')
            plt.ylabel('Training Error')
            plt.show()

    def cross_validate(session, split_size=5):
        results = []
        kf = KFold(n_splits=split_size)
        i = 0
        for train_index, test_index in kf.split(final_train_x):
            train_x1, test_x1 = X[train_index], X[test_index]
            train_y1, test_y1 = Y[train_index], Y[test_index]
            # print(train_index, test_index)
            # print("trx", train_x1)
            # print("try", train_y1)
            # print("tex", test_x1)
            # print("tey", test_y1)
            # if i == 0:
            #  flag = 1
            #else:
            #    flag = 0
            flag = 1
            run_train(session, train_x1, train_y1, test_x1, test_y1, flag, i)
            i += 1
            results.append(session.run(accuracy, feed_dict={x: test_x1, y_: test_y1}))
        return results

    def without_cross_validate(session):
        results = []

        train_x1, test_x1, train_y1, test_y1 = train_test_split(final_train_x, final_train_y, test_size=0.20, random_state=415)
        run_train(session, train_x1, train_y1, test_x1, test_y1, 1, 1)

        results.append(session.run(accuracy, feed_dict={x: test_x1, y_: test_y1}))
        return results

    with tf.Session() as session:
        # result = cross_validate(session)
        result = without_cross_validate(session)
        print("For %s" % name)
        print("Without Cross-validation result %s:" % result)
        # print("Cross-validation result %s:" % result)
        print("Test accuracy: %f" % session.run(accuracy, feed_dict={x: final_test_x, y_: final_test_y}))
        pred_y = session.run(y, feed_dict={x: final_test_x})
        mse = tf.reduce_mean(tf.square(pred_y - final_test_y))
        # print(pred_y)
        # print(final_test_y)
        actual_y = np.argmax(final_test_y, 1)  # 1 0 -> 0 , 0 1 -> 1
        predicted_y = np.argmax(pred_y, 1)  # max min -> 0 , min max -> 1
        f1_score = sk.metrics.f1_score(actual_y, predicted_y)
        precision = sk.metrics.precision_score(actual_y, predicted_y)
        recall = sk.metrics.recall_score(actual_y, predicted_y)
        print(predicted_y[0: 10])
        print("F1-Score: %.4f" % f1_score)
        print("Precision-Score: %.4f" % precision)
        print("Recall-Score: %.4f" % recall)
        print("MSE: %.4f" % session.run(mse))
        print()
        pred_y = np.array(pred_y)[:, 1]
        # print(pred_y)
        y_true = []
        for i in range(0, len(final_test_y)):
            if final_test_y[i][0] == 1:
                y_true.append(0)
            else:
                y_true.append(1)
        y_true = np.array(y_true)
        # print(y_true)
        fpr, tpr, thresholds = roc_curve(y_true, pred_y)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        # print(fpr, tpr, thresholds)
        return fpr, tpr, thresholds, roc_auc


# conflicker_fpr, conflicker_tpr, conflicker_threshholds, conflicker_roc_auc = Thesis("input_csv_files/other/conflicker.csv", 'conflicker')
# kraken_fpr, kraken_tpr, kraken_threshholds, kraken_roc_auc = Thesis("input_csv_files/other/kraken.csv", 'kraken')
# kwyjibo_fpr, kwyjibo_tpr, kwyjibo_threshholds, kwyjibo_roc_auc = Thesis("input_csv_files/other/kwyjibo.csv", 'kwyjibo')
# srizbi_fpr, srizbi_tpr, srizbi_threshholds, srizbi_roc_auc = Thesis("input_csv_files/other/srizbi.csv", 'srizbi')
# torpig_fpr, torpig_tpr, torpig_threshholds, torpig_roc_auc = Thesis("input_csv_files/other/torpig.csv", 'torpig')
# zeus_fpr, zeus_tpr, zeus_threshholds, zeus_roc_auc = Thesis("input_csv_files/other/zeus.csv", 'zeus')


# ML1_9_fpr, ML1_9_tpr, ML1_9_threshholds, ML1_9_roc_auc = Thesis("input_csv_files/hmm_dga/9ML1.csv", '9ML1')
# KL1_500_fpr, KL1_500_tpr, KL1_500_threshholds, KL1_500_roc_auc = Thesis("input_csv_files/hmm_dga/500KL1.csv", '500KL1')
# KL2_500_fpr, KL2_500_tpr, KL2_500_threshholds, KL2_500_roc_auc = Thesis("input_csv_files/hmm_dga/500KL2.csv", '500KL2')
# KL3_500_fpr, KL3_500_tpr, KL3_500_threshholds, KL3_500_roc_auc = Thesis("input_csv_files/hmm_dga/500KL3.csv", '500KL3')
# DNL1_fpr, DNL1_tpr, DNL1_threshholds, DNL1_roc_auc = Thesis("input_csv_files/hmm_dga/DNL1.csv", 'DNL1')
# DNL2_fpr, DNL2_tpr, DNL2_threshholds, DNL2_roc_auc = Thesis("input_csv_files/hmm_dga/DNL2.csv", 'DNL2')
# DNL3_fpr, DNL3_tpr, DNL3_threshholds, DNL3_roc_auc = Thesis("input_csv_files/hmm_dga/DNL3.csv", 'DNL3')
# DNL4_fpr, DNL4_tpr, DNL4_threshholds, DNL4_roc_auc = Thesis("input_csv_files/hmm_dga/DNL4.csv", 'DNL4')


pcfg_dict_fpr, pcfg_dict_tpr, pcfg_dict_threshholds, pcfg_dict_roc_auc = Thesis("input_csv_files/pcfg_dga/pcfg_dict.csv", 'PCFG_DICT')
pcfg_dict_num_fpr, pcfg_dict_num_tpr, pcfg_dict_num_threshholds, pcfg_dict_num_roc_auc = Thesis("input_csv_files/pcfg_dga/pcfg_dict_num.csv", 'PCFG_DICT_NUM')
pcfg_ipv4_fpr, pcfg_ipv4_tpr, pcfg_ipv4_threshholds, pcfg_ipv4_roc_auc = Thesis("input_csv_files/pcfg_dga/pcfg_ipv4.csv", 'PCFG_IPV4')
pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, pcfg_ipv4_num_threshholds, pcfg_ipv4_num_roc_auc = Thesis("input_csv_files/pcfg_dga/pcfg_ipv4_num.csv", 'PCFG_IPV4_NUM')


plt.title('Receiver Operating Characteristic for PCFG BOTNETs')
plt.plot(pcfg_dict_fpr, pcfg_dict_tpr, color='darkgrey', label='pcfg_dict, AUC = %0.2f' % pcfg_dict_roc_auc)
plt.plot(pcfg_dict_num_fpr, pcfg_dict_num_tpr, 'g', label='pcfg_dict_num, AUC = %0.2f' % pcfg_dict_num_roc_auc)
plt.plot(pcfg_ipv4_fpr, pcfg_ipv4_tpr, 'b', label='pcfg_ipv4, AUC = %0.2f' % pcfg_ipv4_roc_auc)
plt.plot(pcfg_ipv4_num_fpr, pcfg_ipv4_num_tpr, 'm', label='pcfg_ipv4_num, AUC = %0.2f' % pcfg_ipv4_num_roc_auc)


'''
plt.title('Receiver Operating Characteristic for HMM BOTNETs')
plt.plot(ML1_9_fpr, ML1_9_tpr, color='darkgrey', label='9ML1, AUC = %0.2f' % ML1_9_roc_auc)
plt.plot(KL1_500_fpr, KL1_500_tpr, 'g', label='500KL1, AUC = %0.2f' % KL1_500_roc_auc)
plt.plot(KL2_500_fpr, KL2_500_tpr, 'b', label='500KL2, AUC = %0.2f' % KL2_500_roc_auc)
plt.plot(KL3_500_fpr, KL3_500_tpr, 'm', label='500KL3, AUC = %0.2f' % KL3_500_roc_auc)
plt.plot(DNL1_fpr, DNL1_tpr, 'c', label='DNL1, AUC = %0.2f' % DNL1_roc_auc)
plt.plot(DNL2_fpr, DNL2_tpr, 'k', label='DNL2, AUC = %0.2f' % DNL2_roc_auc)
plt.plot(DNL3_fpr, DNL3_tpr, 'r', label='DNL3, AUC = %0.2f' % DNL3_roc_auc)
plt.plot(DNL4_fpr, DNL4_tpr, 'y', label='DNL4, AUC = %0.2f' % DNL4_roc_auc)
'''

'''
plt.title('Receiver Operating Characteristic for KNOWN BOTNETs')
# plt.plot(conflicker_fpr, conflicker_tpr, color='darkgrey', label='conflicker, AUC = %0.2f' % conflicker_roc_auc)
plt.plot(kraken_fpr, kraken_tpr, 'g', label='kraken, AUC = %0.2f' % kraken_roc_auc)
# plt.plot(kwyjibo_fpr, kwyjibo_tpr, 'b', label='kwyjibo, AUC = %0.2f' % kwyjibo_roc_auc)
plt.plot(srizbi_fpr, srizbi_tpr, 'm', label='srizbi, AUC = %0.2f' % srizbi_roc_auc)
plt.plot(torpig_fpr, torpig_tpr, 'c', label='torpig, AUC = %0.2f' % torpig_roc_auc)
plt.plot(zeus_fpr, zeus_tpr, 'k', label='zeus, AUC = %0.2f' % zeus_roc_auc)
'''


plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()