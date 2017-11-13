import tensorflow as tf
import numpy as np
import os


def BiRNN(x, weights, biases):
    '''

    :param x: the inputs
    :param weights: the weights
    :param biases: the biases
    :return: a bidirectional rnn
    '''
    print("begin to build the bi_rnn......")
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, shape=[-1, n_inputs])
    x = tf.split(x, n_steps)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    print("end to build the bi_rnn......")
    return tf.matmul(outputs[-1], weights) + biases


def next_batch(ts, tl):
    '''

    :param trs: the train set with the form of dictionary
    :param trl: the train label with the form of dictionary
    :return: the batch data with the length of batch size, train set and train label data after handles
    '''
    batch_set = []
    batch_label = []
    keys_list = []
    batch_step = 0
    for key in ts:
        batch_step += 1
        if batch_step <= batch_size:
            batch_set.append(ts[key])
            batch_label.append(tl[key])
            keys_list.append(key)
        else:
            break
    for k in keys_list:
        del ts[k]
        del tl[k]
    return np.array(batch_set), np.array(batch_label)


def truncated_array(sets):
    for key in sets:
        data = list(sets[key])
        length = 50 if len(data) >= 50 else len(data)
        sub_length = 50 - length
        if sub_length != 0:
            for _ in range(sub_length):
                data.append(np.array([0] * 300))
        else:
            data = data[0: 50]
        data = np.array(data)
        sets[key] = data
    return sets


# '''
# load the existed train set and train label
# '''
# print("load the existed train set and train label......")
# train_set = np.load('./npy_data/train_set.npy').tolist()
# if os.path.exists('./npy_data/trun_train_set_50.npy') == False:
#     np.save('./npy_data/trun_train_set_50.npy', truncated_array(train_set))
#
# '''
# load the existed test set and test label
# '''
# print("load the existed test set and test label......")
# test_set = np.load('./npy_data/test_set.npy').tolist()
# if os.path.exists('./npy_data/trun_test_set_50.npy') == False:
#     np.save('./npy_data/trun_test_set_50.npy', truncated_array(test_set))

'''
define the hyperparameters
'''
print("define the hyperparameters......")

learning_rate = 0.01
batch_size = 50
display_step = 5
train_iters = 50

n_inputs = 300
n_steps = 50
n_hidden_units = 512
n_classes = 8

'''
define the placeholder
'''
print("define the placeholder......")
x = tf.placeholder(shape=[None, n_steps, n_inputs], dtype=tf.float32)
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)

'''
define the weights and biases
'''
print("define the weights and biases......")
weights = tf.Variable(tf.truncated_normal(shape=[2 * n_hidden_units, n_classes]))
biases = tf.Variable(tf.truncated_normal(shape=[n_classes]))

'''
the objective function and accuracy of the prediction
'''
pred = BiRNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
the initialization operation
'''
init = tf.global_variables_initializer()

with tf.Session() as sess:
    '''
    begin to train the bi_rnn
    '''
    sess.run(init)
    print("begin to train the bi_rnn......")
    for iters in range(train_iters):
        step = 1
        trs = np.load(r'D:\PyCharm\my_data\npy_data\trun_train_set_50.npy').tolist()
        trl = np.load('./npy_data/train_label.npy').tolist()
        tes = np.load(r'D:\PyCharm\my_data\npy_data\trun_test_set_50.npy').tolist()
        tel = np.load('./npy_data/test_label.npy').tolist()
        total_acc = 0
        per_iters = int(len(trs) / (batch_size * display_step))
        print("开始第" + str(iters + 1) + "轮训练......")
        while True:
            batch_x, batch_y = next_batch(trs, trl)
            if len(batch_x) < batch_size or len(batch_x) == 0 or len(batch_y) == 0:
                print("the train data is empty now......step= " + str(step))
                break
            try:
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            except Exception as err:
                print(str(type(err)) + str(err))
            if step % display_step == 0:
                batch_x_test, batch_y_test = next_batch(tes, tel)
                if len(batch_x_test) < batch_size or len(batch_x_test) == 0 or len(batch_y_test) == 0:
                    print("the test data is empty now, generate the test data again......step= " + str(step))
                    tes = np.load(r'D:\PyCharm\my_data\npy_data\trun_test_set_50.npy').tolist()
                    tel = np.load('./npy_data/test_label.npy').tolist()
                    batch_x_test, batch_y_test = next_batch(tes, tel)
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test})
                total_acc += acc
                print("step= " + str(step) + ", the accuracy is " + str(acc))
            step = step + 1
        print("完成第" + str(iters + 1) + "轮训练......")
        print("the average accuracy is " + str(total_acc / per_iters) + "\n\n")
    print("Complete the train steps!")
