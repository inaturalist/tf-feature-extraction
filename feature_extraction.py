import os

from scipy.spatial.distance import euclidean

import numpy as np
import tensorflow as tf
from PIL import Image
from nets import inception_v3
from scipy.misc import imread, imresize

slim = tf.contrib.slim

# placeholder for the input image
input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])
# extract the named layer
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(
        input_tensor,
        num_classes=1001,
        is_training=False
    )
    prelogits_op = tf.squeeze(end_points['PreLogits'])

# saver to restore the pretrained imagenet model
saver = tf.train.Saver(max_to_keep=None)
# create the session
sess = tf.Session()
# restore the model
saver.restore(sess, "inception_v3.ckpt")

def extract_prelogits_fv(image_path):
    # open and preprocess the image
    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = np.array(image)
    image = (image - 128.) / 128.
    images = np.expand_dims(image, 0)
    # extract the prelogits
    prelogits = sess.run(
        prelogits_op,
        { input_tensor: images }
    )
    print(prelogits)
    return prelogits

def main():
    mine = "mine.jpg"
    my_vector = extract_prelogits_fv(mine)
    print(my_vector)

if __name__ == "__main__":
    main()
