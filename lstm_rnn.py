import tensorflow as tf
import numpy as np


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

    return results


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


# hyperparameters
lr = 0.001
batch_size = 50
n_inputs = 300
n_steps = 50

train_iters = 50
display_step = 5

# neurons in hidden layer
n_hidden_units = 512
n_classes = 8

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

# Define biases
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
