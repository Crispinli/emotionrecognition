from __future__ import division
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time


class Emotion:
    def __init__(self, batch_size=100, min_after_dequeue=200, num_threads=2, train_times=10000,
                 model_path="./ckpt_data/my_param.ckpt"):
        # the absolute path and name of each tfrecord file
        self.tfrecords_filename_train = './data/Face_train.tfrecords'
        self.tfrecords_filename_test = './data/Face_test.tfrecords'
        # these properties will be used when we attempt to get the batch data
        self.batch_size = batch_size
        self.min_after_dequeue = min_after_dequeue
        self.num_threads = num_threads
        self.capacity = self.min_after_dequeue + 3 * self.batch_size
        # the number of iteration when train the CNN
        self.train_times = train_times  # use the training images for two times
        # the path which save the ckpt file
        self.model_path = model_path

    def read_tfrecord_to_data(self, filename, num_epochs=None):
        '''

        :param filename: an absolute path which contains the tfrecord file
        :return: an image and a corresponding label
        '''
        filename_queue = tf.train.string_input_producer(
            [filename],
            num_epochs=num_epochs
        )
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'img_floder': tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = self.image_pre_process(image)
        label = tf.decode_raw(features['label'], tf.float64)
        label = tf.reshape(label, shape=[8])
        # img_floder = features['img_floder']
        return image, label

    def image_pre_process(self, image):
        '''

        :param image: the image need to be preprocessed
        :return: a preprocessed image
        '''
        image = tf.reshape(image, [100, 100, 1])
        image = tf.image.convert_image_dtype(
            image=image,
            dtype=tf.float32
        )
        image = tf.image.random_flip_left_right(image=image)
        # image = self.distort_color(image=image, color_ordering=np.random.randint(2))
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_images(images=image, size=[96, 96], method=1)
        return image

    def distort_color(self, image, color_ordering=0):
        '''

        :param image: the image need to be adjusted
        :param color_ordering: the method of how to adjust the image
        :return: an image(a 3-D tensor)
        '''
        if color_ordering == 0:
            image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image=image, max_delta=0.2)
            image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image=image, max_delta=0.2)
        return tf.clip_by_value(t=image, clip_value_min=0.0, clip_value_max=1.0)

    def add_conv2D(self, input, out_size, kw, kh, sw=1, sh=1, padding='SAME', is_training=True):
        in_size = input.get_shape()[-1].value
        kernal_shape = [kw, kh, in_size, out_size]
        kernal = tf.Variable(tf.truncated_normal(shape=kernal_shape, mean=0.0, stddev=1.0, dtype=tf.float32))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, kernal, strides=[1, sw, sh, 1], padding=padding), bias)
        conv_bn = self.batch_norm(conv, is_training=is_training, is_conv_out=True)
        activation = tf.nn.relu(conv_bn)
        return activation

    def add_pooling(self, input, kw=2, kh=2, sw=2, sh=2, padding="SAME"):
        ksize = [1, kw, kh, 1]
        strides = [1, sw, sh, 1]
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

    def add_fc(self, input, out_size, is_training=True):
        length = len(input.get_shape())
        if length == 4:
            w1 = input.get_shape()[1].value
            w2 = input.get_shape()[2].value
            w3 = input.get_shape()[3].value
            w = w1 * w2 * w3
        elif length == 2:
            w = input.get_shape()[-1].value
        weights = tf.Variable(tf.truncated_normal(shape=[w, out_size], mean=0.0, stddev=1.0, dtype=tf.float32))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        input2D = tf.reshape(input, shape=[-1, w])
        fc = tf.nn.bias_add(tf.matmul(input2D, weights), bias=bias)
        fc_bn = self.batch_norm(fc, is_training=is_training, is_conv_out=False)
        # activation = tf.nn.relu(fc_bn)
        return fc_bn

    def batch_norm(self, inputs, is_training, is_conv_out=True, decay=0.997):
        '''

        :param inputs: the input tensor
        :param is_training: if the network is being trained
        :param is_conv_out: if the current layer is convolution layer
        :param decay: the decay factor
        :return: the normalized tensor
        '''
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        if is_training == True:
            if is_conv_out == True:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)

    def build_network(self):
        '''

        :return: the train_step interface and accuracy interface
        '''
        # ----------------------- build the configuration of the CNN ----------------------- #
        self.x = tf.placeholder(shape=[None, 96, 96, 1], dtype=tf.float32)
        # self.y_ = tf.placeholder(shape=[None, 8], dtype=tf.float32)
        self.is_training = tf.placeholder(shape=(), dtype=tf.bool)

        x_bn = self.batch_norm(self.x, self.is_training, is_conv_out=True)

        # the first convolution layer
        h_conv1_1 = self.add_conv2D(input=x_bn, out_size=32, kw=1, kh=1, is_training=self.is_training)

        h_conv1_2 = self.add_conv2D(input=h_conv1_1, out_size=32, kw=1, kh=3, is_training=self.is_training)

        h_conv1_3 = self.add_conv2D(input=h_conv1_2, out_size=32, kw=3, kh=1, is_training=self.is_training)

        h_pool1 = self.add_pooling(input=h_conv1_3, kw=2, kh=2, sw=2, sh=2)

        # the second convolution layer
        h_conv2 = self.add_conv2D(input=h_pool1, out_size=64, kw=3, kh=3, is_training=self.is_training)

        h_pool2 = self.add_pooling(h_conv2, kw=2, kh=2, sw=2, sh=2)

        # the third convolution layer
        h_conv3 = self.add_conv2D(input=h_pool2, out_size=128, kw=3, kh=3, is_training=self.is_training)

        h_pool3 = self.add_pooling(h_conv3, kw=2, kh=2, sw=2, sh=2)

        # the first full connected layer
        h_fc1 = tf.nn.relu(self.add_fc(input=h_pool3, out_size=300, is_training=self.is_training))

        # the second full connected layer
        h_fc2 = self.add_fc(input=h_fc1, out_size=8, is_training=self.is_training)

        # the loss function
        # diff = tf.nn.softmax_cross_entropy_with_logits(logits=h_fc2, labels=self.y_)
        #
        # cross_entropy_loss = tf.reduce_mean(diff)
        #
        # # the objective function
        # train_step = tf.train.AdagradOptimizer(1e-1).minimize(cross_entropy_loss)
        #
        # correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(self.y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return h_fc2


def run_test_step():
    '''

    :return: void
    '''
    results_of_cnn_50 = {}
    data_root = r"D:\mec_test_data\test_data_gray"
    npy_root = r"D:\mec_test_data" + os.sep + "image_counter_lower_50_npy.npy"
    if os.path.exists(npy_root) == True:
        image_counter_lowerer_50 = np.load(npy_root).tolist()
    for key in image_counter_lowerer_50:
        results_of_cnn_50[key] = []

    emotion = Emotion()
    h_fc2 = emotion.build_network()

    # ----------------------------- the initializer ----------------------------- #
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    )

    # ------------------ the saver used to save the parameters ------------------ #
    saver = tf.train.Saver()

    # ------------------------------ train the CNN ------------------------------ #
    with tf.Session() as sess:
        sess.run(init_op)
        print("load the ckpt data......")
        saver.restore(sess, emotion.model_path)

        current_var = 1
        for key in image_counter_lowerer_50:
            data = data_root + os.sep + key
            image_names = os.listdir(data)
            image_names.sort(key=lambda x: len(x))
            var = 0
            current_time = time.time()
            for image_name in image_names:
                if var < 50:
                    image = data + os.sep + image_name
                    img = np.array(Image.open(image))
                    img = np.reshape(a=img, newshape=[1, 96, 96, 1])
                    classification = sess.run(h_fc2, feed_dict={
                        emotion.x: img,
                        emotion.is_training: False
                    })
                    label = np.argmax(classification,axis=1)
                    results_of_cnn_50[key].append(label[0])
                    var = var + 1
                else:
                    break
            print(key, current_var, time.time() - current_time)
            current_var = current_var + 1
        np.save("./npy_data/results_of_cnn.npy", results_of_cnn_50)


# run_train_step()
run_test_step()
