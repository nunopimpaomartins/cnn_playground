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
from nets import unet

limit_gpu_memory(fraction=1/2)

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

n = 10
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




model = unet(input_shape=(128, 128, 1), conv_kernel_size=3, dropout=0.5, filters=64, last_activation='sigmoid')
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
model.summary()

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
plot_history(history,['loss','val_loss'])

model.save('myunet_2.h5')
