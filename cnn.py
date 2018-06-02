from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os


class RoadsCnn:
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        self.save_file_name = 'cnn_weights.h5'

        self.batch_size = 32
        self.repeat_training = 0
        self.epochs = 2
        self.steps_per_epoch = 15000
        self.validation_steps = 800

        self._model = None
        self._train_gen = None
        self._test_gen = None

    def create_cnn(self, path_train_data, path_test_data):
        self.create_model()
        self.generate_train_and_test_keras_generator(
            path_train_data, path_test_data)

        if os.path.isfile(self.save_file_name):
            self._model.load_weights(self.save_file_name)

        for i in range(self.repeat_training):
            self.train_model()

    def generate_train_and_test_keras_generator(self, train_dir, test_dir):
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1. / 255
        )

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        self._train_gen = train_generator
        self._test_gen = test_generator

    def create_model(self):
        model = Sequential()
        model.add(Convolution2D(128, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())
        self._model = model

    def train_model(self):
        self._model.fit_generator(
            self._train_gen,
            steps_per_epoch=self.steps_per_epoch // self.batch_size,
            epochs=self.epochs,
            validation_data=self._test_gen,
            validation_steps=self.validation_steps // self.batch_size
        )

        print("Saving weights...")
        self._model.save_weights(self.save_file_name)

    def predict(self, image, skip=1):
        arr = []
        for x in self.img_pre_gen(image, skip):
            arr.append([i[0] for i in self._model.predict(x)])

        return arr

    def img_pre_gen(self, image, skip=1):
        for x in range(0, len(image) - self.input_shape[0], skip):
            print(x)
            arr = []
            for y in range(0, len(image[0]) - self.input_shape[1], skip):
                arr.append(
                    np.array(image[x:x + self.input_shape[0], y:y + self.input_shape[1], :]))
            yield np.array(arr)
