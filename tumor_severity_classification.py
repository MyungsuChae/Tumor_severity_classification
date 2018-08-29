########## Libraries
import os
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

########## GPU Setting
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,5,6"


########## Input image dimensions
img_rows, img_cols = 256, 256

########## Number of channels
img_channels = 1

########## Data path
path1 = "NEED_TO_CHANGE"  # path of folder of images
path2 = 'NEED_TO_CHANGE'  # path of folder to save resized images

listing = os.listdir(path1)
num_samples = size(listing)

########## Preprocessing
for file in sorted(listing):
    im = Image.open(path1 + '/' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here
    gray.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('input_data_resized' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

########## create matrix to store all flattened images
immatrix = array([array(Image.open('input_data_resized' + '/' + im2)).flatten()
                  for im2 in sorted(imlist)], 'f')

label = np.ones((num_samples,), dtype=int)

label[0:159] = 1 # grade 3
label[160:462] = 0 # grade 2

data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]

img = immatrix[167].reshape(img_rows, img_cols) # sample
# print(immatrix)

# plt.imshow(img)
# plt.imshow(img, cmap='gray')
# print(train_data[0].shape)
# print(train_data[1].shape)

########## batch_size to train
batch_size = 40
########## number of output classes
nb_classes = 2
########## number of epochs to train
nb_epoch = 100

########## number of convolutional filters to use
nb_filters = 16
########## size of pooling area for max pooling
nb_pool = 2
########## convolution kernel size
nb_conv = 4


########## data collection
(X, y) = (train_data[0],train_data[1])

########## split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

########## convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# i = 100 # iteration
# plt.imshow(X_train[i, 0], interpolation='nearest')
# print("label : ", Y_train[i,:])

########## modeling CNN

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
convout3 = Activation('relu')
model.add(convout3)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.1)) # dropout

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics = ['accuracy'])

########## model fitting

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2)

########## visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'],loc=4)
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

########## performance

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(model.summary())
