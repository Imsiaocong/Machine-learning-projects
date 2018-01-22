import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam

''' load and return the dataset '''
def L(filepath):
        dataframe = pd.read_csv(filepath)
        pixels = dataframe.values[:, 1:]
        labels = dataframe.values[:, 0]
        one_hot_encoded_labels = create_one_hot_encoded_labels(labels)

        return pixels, one_hot_encoded_labels


''' data normalization '''
def normalize_pixels(pixels):
        pixels = pixels.astype('float32')
        pixels = pixels.reshape(-1, 28, 28, 1)/255.0

        return pixels

def create_one_hot_encoded_labels(labels):
        unique_labels_count = np.unique(labels).shape[0]
        one_hot_encoded_labels = to_categorical(labels, 10)

        return one_hot_encoded_labels

def model():
        m = Sequential()

        m.add(Conv2D(filters=512, kernel_size=3, strides=1, batch_input_shape=(None, 28, 28, 1), padding='same', data_format='channels_first'))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first'))

        m.add(Conv2D(512, 3, strides=1, padding='same', data_format='channels_first'))
        m.add(Activation('relu'))
        m.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
        #Something wrong
        #m.add(Flatten())
        m.add(Dense(10, activation='softmax'))

        return m


if __name__ == '__main__':
        pixels_train, labels_train = L('fashionmnist/fashion-mnist_train.csv')
        pixels_train = normalize_pixels(pixels_train)

        pixels_test, labels_test = L('fashionmnist/fashion-mnist_test.csv')
        pixels_test = normalize_pixels(pixels_test)

        print(pixels_train.shape)
        print(labels_train.shape)

        model = model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=pixels_train, y=labels_train, batch_size=128, epochs=5, verbose=1)

        score = model.evaluate(pixels_test, labels_test, verbose=0)

        print('Test loss: {}'.format(score[0]))
        print('Test accuracy: {}'.format(score[1]))
        model.save('save_model.h5')
        plot_model(model, to_file='model.png')
