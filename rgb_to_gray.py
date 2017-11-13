import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import time

root1 = r"D:\mec_test_data\test_data_gray"
root2 = r"C:\Users\lwp\Desktop\test_data\test"


def get_pix(file_name):
    '''
    :author: lwp
    :description: 若为无效图像，则会返回True，否则返回False
    :param file_name: 图像的绝对路径
    :return: boolean
    '''
    img = Image.open(file_name)
    pixdata = img.load()
    tsize = 0
    try:
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if pixdata[x, y][0] > 250 and pixdata[x, y][0] > 250 and pixdata[x, y][0] > 250:
                    tsize = tsize + 1
        allsize = img.size[0] * img.size[1]
        percentage = float(tsize) / float(allsize)
        if (percentage >= 0.9):
            return True
        else:
            return False
    except Exception as err:
        # print(str(type(err)) + str(err))
        return True

# with tf.Session() as sess:
region = [2, 2, 98, 98]
var = 1
filenames = os.listdir(root2)
for filename in filenames:
    floder2 = root2 + os.sep + filename
    floder1 = root1 + os.sep + filename
    if os.path.exists(floder1) == False:
        os.mkdir(floder1)
    current_time = time.time()
    for image in os.listdir(floder2):
        img_name2 = floder2 + os.sep + image
        img_name1 = floder1 + os.sep + image
        try:
            if get_pix(img_name2) == False:
                img = Image.open(img_name2).convert('L')
                img = img.crop(region)
                img = np.array(img)
                cv2.imwrite(img_name1, img)
        except:
            continue
    print(filename, var, time.time() - current_time)
    var = var + 1
