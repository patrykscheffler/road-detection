from skimage import exposure
from data import *
from cnn import *


def roads(image):
    imsize = 27
    input_shape = (imsize, imsize, 3)

    cnn = RoadsCnn(input_shape)
    pre = cnn.predict(image, skip=1)

    return pre
