import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell


label_index = {
    '0': "angry",
    '1': "anxious",
    '2': "disgust",
    '3': "happy",
    '4': "neutral",
    '5': "sad",
    '6': "surprise",
    '7': "worried"
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # multi rnn cell
    ##########################################
    cell = MultiRNNCell([BasicLSTMCell(n_hidden_units) for _ in range(layer_num)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)

    return results


def next_batch(ts):
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
            keys_list.append(key)
        else:
            break
    for k in keys_list:
        del ts[k]
    return np.array(batch_set), keys_list[0]


# hyperparameters
lr = 0.01
batch_size = 1
n_inputs = 300
n_steps = 50

train_iters = 10
display_step = 5

layer_num = 5

# neurons in hidden layer
n_hidden_units = 512
n_classes = 8

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

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

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    '''
    begin to train the bi_rnn
    '''
    sess.run(init)
    saver.restore(sess=sess, save_path='./rnn_ckpt_data/my_param.ckpt')

    results_dict = {}
    trs = np.load(r'D:\PyCharm\my_data\npy_data\data_for_rnn.npy').tolist()
    for key in trs:
        results_dict[key] = -1

    total_acc = 0
    per_iters = int(len(trs) / (batch_size * display_step))
    var = 1
    while True:
        try:
            batch_x, floder = next_batch(trs)
        except:
            break
        try:
            prediction = sess.run(pred, feed_dict={x: batch_x})
            results_dict[floder] = label_index[str(np.argmax(prediction[0]))]
        except Exception as err:
            print(str(type(err)) + str(err))
        print(var, results_dict[floder])
        var = var + 1
    np.save('./total_results/rnn_results.npy', results_dict)
    print("Completed!")
