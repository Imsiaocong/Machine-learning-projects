import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
num_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28, 1)/255.
X_test = X_test.reshape(-1, 28, 28, 1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

m = Sequential()
m.add(Dense(16, activation='relu', batch_input_shape=(None, 28, 28, 1)))
m.add(Dense(16, activation='relu'))
m.add(Conv2D(32, 3, strides=1, padding='same', data_format='channels_first', activation='relu'))
m.add(Flatten())
m.add(Dense(10, activation='sigmoid'))
"""
m.add(Conv2D(
    batch_input_shape=(16 , 28, 28, 1),
    filters=16,
    kernel_size=3,
    strides=1,
    padding='same',      # Padding method
    data_format='channels_first',
))

m.add(Activation('relu'))
m.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

m.add(Conv2D(64, 3, strides=1, padding='same', data_format='channels_first'))
m.add(Activation('relu'))
m.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

m.add(Flatten())
m.add(Dense(512))
m.add(Activation('relu'))

m.add(Dense(10))
m.add(Activation('softmax'))
"""
print(y_train.shape)
'''
m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(y_train.shape)
print('Training ------------')
m.fit(X_train, y_train, epochs=1, batch_size=16)
print('done')
print('\nTesting ------------')
loss, accuracy = m.evaluate(X_test, y_test, 16)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
plot_model(m, to_file='model.png')
#m.save('my_model.h5')
del m
'''
