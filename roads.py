from skimage import exposure
from data import *
from cnn import *


def roads(image):
    imsize = 27
    input_shape = (imsize, imsize, 3)

    cnn = RoadsCnn(input_shape)
    pre = cnn.predict(image, skip=1)

    return pre


def main():
    imsize = 27
    input_shape = (imsize, imsize, 3)

    cnn = RoadsCnn(input_shape)
    cnn.create_roads_cnn()

    predict_generator(loadDataFromPathWithLabel('test_images/', norm=True), cnn)

    return


def predict_generator(gen, cnn, save_path='test_images_predict/'):
    for img, img_name in gen:
        print('Predict: ' + img_name)
        pre = cnn.predict(img, skip=3)
        # imsave(save_path + img_name, normalize(pre), cmap='gray')
        imsave(save_path + img_name, pre, cmap='gray')


if __name__ == '__main__':
    main()
