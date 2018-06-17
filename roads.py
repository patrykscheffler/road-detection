from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from skimage import exposure
from cnn import *
from pylab import *

import numpy as np
import cv2


def roads(image):
    imsize = 49 
    input_shape = (imsize, imsize, 3)

    cnn = RoadsCnn(input_shape)
    cnn.create_roads_cnn() 
    img = image[..., ::-1]
    pad = imsize // 2
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)),
           'constant', constant_values=(0, 0))
    img = cv2.normalize(
        img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pre = cnn.predict(img, skip=3)

    return pre
