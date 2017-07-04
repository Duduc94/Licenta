"""
Author: Duduc Ionut
Date: 28.04.2017
"""

import os
import re
import cv2
import glob
import numpy as np


from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from FrameExtractor import FrameExtractor
from utils import clear_folder, ensure_dir
from keras import backend as K
K.set_image_dim_ordering('th')


class Vgg16:
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))

    def __init__(self):
        """
        Constructor: creates Vgg16 model and loads weights for later use
        """
        model = self.model = Sequential()
        model.add(Lambda(self.vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(3, activation='softmax'))

        fname = 'weights_3class_comprised.h5'
        model.load_weights(os.path.join(os.path.abspath('../weights'), fname))

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def get_data(self, path, target_size=(224, 224)):
        batches = self.get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
        return batches, np.concatenate([batches.next() for i in range(batches.samples)])

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical',
                    target_size=(224, 224)):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def vgg_preprocess(self, x):
        x = x - self.vgg_mean  # subtract mean
        return x[:, ::-1]  # reverse axis bgr->rgb

    def classify(self, path='C:\\Users\\Dudu\\Desktop\\Licenta\\VisualProcessing\\AudioVisualClip\\JK\\a7.avi',
                 temp_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')):
        emotion = ['negative', 'neutral', 'positive']
        ensure_dir(temp_path)
        images = FrameExtractor.process_file(path)
        for i in range(len(images)):
            cv2.imwrite(os.path.join(temp_path, 'unlabeled', "%d.jpg" % i), images[i])
        batches, test_data = self.get_data(temp_path)
        print('Starting prediction...')
        features = np.mean(self.model.predict(test_data, batch_size=1), axis=0)
        print('The video expresses %s feelings.' % emotion[np.argmax(features)])
        clear_folder(temp_path)
        return features

    def classify_all(self):
        source = os.path.abspath('../dataset/test')
        dict = {'a': 0, 'd': 0, 'f': 0, 'h': 2, 'n': 1, 'sa': 0, 'su': 2}
        video_no = 0
        vision_probabilities = []
        labels = []

        # parse entire directory and process all .avi files in it
        for file_path in glob.iglob(os.path.join(source, r'**/*.avi'), recursive=True):
            regex = r"([a-zA-Z]+)\d+"
            file_label = re.findall(regex, os.path.split(file_path)[-1])[0]
            for key in dict.keys():
                if key == file_label:
                    labels.append(dict[key])
            vision_probabilities.append(self.classify(file_path))
            video_no += 1

        vision_probabilities = np.array(vision_probabilities)
        labels = to_categorical(np.array(labels))

        np.save('vision_probabilities_test', vision_probabilities)
        np.save('labels_test', labels)

        print(vision_probabilities.shape)
        print(labels.shape)

vgg16 = Vgg16()
vgg16.classify_all()
# plot_model(vgg16.model, to_file='model.png')
# print(vgg16.model.summary())
