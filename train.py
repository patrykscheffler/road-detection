from skimage import exposure
from data import *
from cnn import *


def main():
    imsize = 27
    input_shape = (imsize, imsize, 3)

    generateData(size=imsize, skip=10)

    cnn = RoadsCnn(input_shape)
    cnn.create_cnn(path_train_data=path_train + path_dataset,
                   path_test_data=path_test + path_dataset)

    cnn.train_model()

    return


if __name__ == '__main__':
    main()
