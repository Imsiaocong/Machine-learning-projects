import numpy as np
import os
import os

from skimage import color, io
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
from glob import glob
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical

file_dir = '/Users/diwang/Desktop/ML/Cats_vs_dogs/train/'
save_dir = '/Users/diwang/Desktop/ML/Machine-learning-projects/Cats_vs_dogs/resized/'
'''
def rescale_img(path):
        for file in os.listdir(path):
                img = Image.open(path + file)
                img = img.resize((150, 150), Image.ANTIALIAS)
                img.save(save_dir + file)
'''
'''
def open_images(file_path):
        cats = []
        dogs = []
        label_cats = []
        label_dogs = []

        for file in os.listdir(file_path):
                name = file.split('.')
                if name[0] == 'cat':
                        cats.append(file_dir + file)
                        label_cats.append(0)
                else:
                        dogs.append(file_dir + file)
                        label_dogs.append(1)

        image_list = np.hstack((cats, dogs))
        label_list = np.hstack((label_cats, label_dogs))
        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)

        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        label_list = [int(i) for i in label_list]

        return image_list, label_list
'''
def model():
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
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if __name__ == '__main__' :
###################################
### Import picture files
###################################

    files_path = file_dir

    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')

    cat_files = sorted(glob(cat_files_path))
    dog_files = sorted(glob(dog_files_path))

    n_files = len(cat_files) + len(dog_files)
    print(n_files)

    size_image = 64

    allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
    ally = np.zeros(n_files)
    count = 0
    for f in cat_files:
        try:
            img = io.imread(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            ally[count] = 0
            count += 1
        except:
            continue

    for f in dog_files:
        try:
            img = io.imread(f)
            new_img = imresize(img, (size_image, size_image, 3))
            allX[count] = np.array(new_img)
            ally[count] = 1
            count += 1
        except:
            continue
###################################
# Prepare train & test samples
###################################

# test-train split
    X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
    Y = to_categorical(Y, 2)
    Y_test = to_categorical(Y_test, 2)

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
    model.add(Activation('sigmoid'))
        
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(X, Y, batch_size=64, epochs=1, verbose=1)
