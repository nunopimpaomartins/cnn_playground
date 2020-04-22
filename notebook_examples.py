from keras import Input, layers
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import os
from tqdm import tqdm_notebook as tqdm

from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.utils import plot_some, plot_history
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data

limit_gpu_memory(fraction=1/2)

input_tensor = Input(shape=(128, 128, 1))
conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(input_tensor)
conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv1)
pool1 = layers.MaxPool2D((2, 2))(conv1)
conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv2)
pool2 = layers.MaxPool2D((2,2))(conv2)
conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')(pool2)
conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv3)
pool3 = layers.MaxPool2D((2,2))(conv3)
conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(pool3)
conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(conv4)
drop4 = layers.Dropout(0.5)(conv4)
pool4 = layers.MaxPool2D((2,2))(drop4)

conv5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(pool4)
conv5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(conv5)
drop5 = layers.Dropout(0.5)(conv5)

up6 = layers.UpSampling2D((2,2))(drop5)
up6 = layers.Conv2D(512, 3,padding='same', activation='relu')(up6)
#crop4 = layers.
merged6 = layers.concatenate([conv4, up6], axis=3)
conv6 = layers.Conv2D(512, 3,padding='same', activation='relu')(merged6)
conv6 = layers.Conv2D(512, 3,padding='same', activation='relu')(conv6)

up7 = layers.Conv2D(256, 2,padding='same', activation='relu')(layers.UpSampling2D((2,2))(conv6))
merged7 = layers.concatenate([conv3, up7], axis=3)
conv7 = layers.Conv2D(256, 3,padding='same', activation='relu')(merged7)
conv7 = layers.Conv2D(256, 3,padding='same', activation='relu')(conv7)

up8 = layers.Conv2D(128, 2,padding='same', activation='relu')(layers.UpSampling2D((2,2))(conv7))
merged8 = layers.concatenate([conv2, up8], axis=3)
conv8 = layers.Conv2D(128, 3,padding='same', activation='relu')(merged8)
conv8 = layers.Conv2D(128, 3,padding='same', activation='relu')(conv8)

up9 = layers.Conv2D(64, 2,padding='same', activation='relu')(layers.UpSampling2D((2,2))(conv8))
merged9 = layers.concatenate([conv1, up9], axis=3)
conv9 = layers.Conv2D(64, 3,padding='same', activation='relu')(merged9)
conv9 = layers.Conv2D(64, 3,padding='same', activation='relu')(conv9)
conv9 = layers.Conv2D(2, 3,padding='same', activation='relu')(conv9)
output_tensor = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')

model.summary()

base_dir = '/mnt/AE3205C73205958D/Data/3dliver_local/pc_adult/2d_slices/imagesXY/images_full/'
#train_dir = os.path.join(base_dir, 'trainingImages_512x512/')
train_dir = os.path.join(base_dir, 'trainImages')
#validation_dir = os.path.join(base_dir, 'trainingLabels_512x512/')
label_dir = os.path.join(base_dir, 'trainLabels')
#test_dir = os.path.join(base_dir, 'testImages_512x512')
#print(train_dir)

imgList = os.listdir(train_dir)
labelList = os.listdir(label_dir)

imgArray = []
for image in tqdm(imgList, 'Reading img'):
    imgArray.append(imread(os.path.join(train_dir, image)))

labelArray = []
for label in tqdm(labelList, 'Reading label'):
    labelArray.append(imread(os.path.join(label_dir, label)))

print(imgArray[0].shape)
print(labelArray[0].shape)

raw_data = RawData.from_arrays(
    imgArray,
    labelArray,
    axes='YX'
)

X, Y, XY_axes = create_patches(
    raw_data=raw_data,
    patch_size=(128, 128, 1),
    patch_axes='YXC',
    n_patches_per_image=25,
    save_file='/mnt/AE3205C73205958D/Data/3dliver_local/pc_adult/2d_slices/imagesXY/image_full/mydata_128x128patch.npz'
)

(X, Y), (X_val, Y_val), axes = load_training_data(
    '/mnt/AE3205C73205958D/Data/3dliver_local/pc_adult/2d_slices/imagesXY/image_full/mydata_128x128patch.npz',
    validation_split=0.1,
    verbose=True
)

n = 15
print(X[n].shape)
fig = plt.figure()
pl1_x = fig.add_subplot(2, 5, 1)
pl1_x.imshow(X[n][...,0])
pl1_y = fig.add_subplot(2, 5, 6)
pl1_y.imshow(Y[n][...,0])
pl2_x = fig.add_subplot(2, 5, 2)
pl2_x.imshow(X[n+1][...,0])
pl2_y = fig.add_subplot(2, 5, 7)
pl2_y.imshow(Y[n+1][...,0])
pl3_x = fig.add_subplot(2, 5, 3)
pl3_x.imshow(X[n+2][...,0])
pl3_y = fig.add_subplot(2, 5, 8)
pl3_y.imshow(Y[n+2][...,0])
pl4_x = fig.add_subplot(2, 5, 4)
pl4_x.imshow(X[n+3][...,0])
pl4_y = fig.add_subplot(2, 5, 9)
pl4_y.imshow(Y[n+3][...,0])
pl5_x = fig.add_subplot(2, 5, 5)
pl5_x.imshow(X[n+4][...,0])
pl5_y = fig.add_subplot(2, 5, 10)
pl5_y.imshow(Y[n+4][...,0])

print(np.max(X[n]))
print(np.max(Y[n]))

history = model.fit(
    X,
    Y,
    batch_size=16,
    epochs=3,
    #steps_per_epoch=50,
    validation_data=(X_val, Y_val),
    #validation_steps=10,
    #validation_split=0.1,
    shuffle=True,
    validation_freq=1
)

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss']);

model.save('myunet_crossentropy_loss.h5')