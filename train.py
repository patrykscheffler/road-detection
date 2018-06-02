from skimage import exposure
from data import *
from cnn import *


def main():
    imsize = 27
    input_shape = (imsize, imsize, 3)

    generateData(size=imsize, skip=5)

    # cnn = retinaCnn(input_shape)
    # cnn.create_cnn(path_train_data=path_drive_train + path_dataset,
    #                path_test_data=path_drive_test + path_dataset)

    # predict_generator(loadDataFromPathWithLabel(
    #     'test_images/', norm=True), cnn)
    return


def predict_generator(gen, cnn, save_path='test_images_predict/'):
    for img, img_name in gen:
        print('predict: ' + img_name)
        pre = cnn.predict(img, skip=1)
        imsave(save_path + img_name, normalize(pre), cmap='gray')


if __name__ == '__main__':
    main()
