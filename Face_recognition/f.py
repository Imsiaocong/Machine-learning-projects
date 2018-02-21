from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import os
import keras
import numpy as np
string = 'IMG'
path = './rsc/'
path_test = './test/'
path_save = './resized/IMG.jpg'
filename = os.listdir(path)
data = np.empty((78, 64, 64, 3),dtype="float32")
label = np.empty((78,),dtype="uint8")
i = 0

for f in filename:
    if string in f:
        img = Image.open(path + f)
        img = img.resize((64,64), Image.ANTIALIAS)
        img_arr = np.asarray(img,dtype="float32")
        data[i, :, :, :] = img_arr
        label[i] = 1
        i += 1
    elif "false" in f:
        img = Image.open(path + f)
        img = img.resize((64,64), Image.ANTIALIAS)
        img_arr = np.asarray(img,dtype="float32")
        data[i, :, :, :] = img_arr
        label[i] = 0
        i += 1

print(data.shape)
print(label.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

trainlabel = keras.utils.to_categorical(label, 2)
print(label)
print(trainlabel)
model.fit(data, trainlabel, epochs=5, batch_size=32)

data_test = np.empty((1, 64, 64, 3),dtype="float32")
img = Image.open(path_test + 'IMG_1350.JPG')
img = img.resize((64,64), Image.ANTIALIAS)
img_arr = np.asarray(img,dtype="float32")
data_test[0, :, :, :] = img_arr
pred = model.predict(data_test)
print(pred)

