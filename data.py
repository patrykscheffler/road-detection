from pylab import *
from skimage import data
from sklearn import preprocessing

import os

path_test = "data/test/"
path_train = "data/training/"

path_y = "target/"
path_x = "input/"

path_positive = "positive/"
path_negative = "negative/"


white = 1
black = 0


def generateData(size=49, skip=5, img_gen_count=(-1)):
    train_x = loadDataFromPath(path_train + path_x, norm=True)
    train_y = loadDataFromPath(path_train + path_y, norm=True)

    test_x = loadDataFromPath(path_test + path_x, norm=True)
    test_y = loadDataFromPath(path_test + path_y, norm=True)

    if checkPathExistsCreateIfNot(path_train + path_positive):
        return
    if checkPathExistsCreateIfNot(path_train + path_negative):
        return
    if checkPathExistsCreateIfNot(path_test + path_positive):
        return
    if checkPathExistsCreateIfNot(path_test + path_negative):
        return

    i = 0
    for x, y in zip(train_x, train_y):
        prepareTrainingSet(str(i), x, y, path_train + path_positive,
                           path_train + path_negative, skip=skip, size=size)
        i += 1
        if (img_gen_count != -1) and (i >= img_gen_count):
            break

    i = 0
    for x, y in zip(test_x, test_y):
        prepareTrainingSet(str(i), x, y, path_test + path_positive,
                           path_test + path_negative, skip=skip, size=size)
        i += 1
        if (img_gen_count != -1) and (i >= img_gen_count):
            break


def listDirWithSort(path):
    folder = os.listdir(path)
    folder.sort()
    return folder


def loadDataFromPath(path, norm=False):
    for file in listDirWithSort(path):
        print(file)
        img = data.imread(path + file).astype(np.float64)

        if norm:
            img = normalize(img)

        yield img


def loadDataFromPathWithLabel(path, norm=False):
    for file in listDirWithSort(path):
        print(file)
        img = data.imread(path + file).astype(np.float64)

        if norm:
            img = normalize(img)

        yield img, file


def checkPathExistsCreateIfNot(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        return False
    return True


def prepareTrainingSet(name, train_x, train_y, path_pos, path_neg, skip=5, size=27, skip_black=True):
    for x in range(0, len(train_x) - size, skip):
        print("x:", x)
        for y in range(0, len(train_x) - size, skip):

            img = train_x[x:x + size, y:y + size, :]
            if train_y[x + size // 2, y + size // 2] == white:
                imsave(path_pos + name + "_" + str(x) +
                       "_" + str(y) + "_pos", img)
            else:
                imsave(path_neg + name + "_" + str(x) +
                       "_" + str(y) + "_neg", img)


def normalize(image):
    image = image.copy()

    minval = np.amin(image)
    maxval = np.amax(image)
    image -= minval
    image /= (maxval - minval)

    return image
