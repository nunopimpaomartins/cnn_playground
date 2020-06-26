from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

encoder = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(128, 128, 3))

kernel_size = (3, 3)
last_activation = 'softmax'

up1 = layers.UpSampling2D((2, 2))(encoder.get_layer('block5_pool').output)
upblock1_conv1 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(up1)
upblock1_conv2 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(upblock1_conv1)
upblock1_conv3 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(upblock1_conv2)

up2 = layers.UpSampling2D((2, 2))(upblock1_conv3)
up_merge = layers.Concatenate()([encoder.get_layer('block4_conv3').output, up2])
upblock2_conv1 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(up_merge)
upblock2_conv2 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(upblock2_conv1)
upblock2_conv3 = layers.Conv2D(512, kernel_size, padding='same', activation='relu')(upblock2_conv2)

up3 = layers.UpSampling2D((2, 2))(upblock2_conv3)
up_merge2 = layers.Concatenate()([encoder.get_layer('block3_conv3').output, up3])
upblock3_conv1 = layers.Conv2D(256, kernel_size, padding='same', activation='relu')(up_merge2)
upblock3_conv2 = layers.Conv2D(256, kernel_size, padding='same', activation='relu')(upblock3_conv1)
upblock3_conv3 = layers.Conv2D(256, kernel_size, padding='same', activation='relu')(upblock3_conv2)

up4 = layers.UpSampling2D((2, 2))(upblock3_conv3)
up_merge3 = layers.Concatenate()([encoder.get_layer('block2_conv2').output, up4])
upblock4_conv1 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(up_merge3)
upblock4_conv2 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(upblock4_conv1)

up5 = layers.UpSampling2D((2, 2))(upblock4_conv2)
up_merge4 = layers.Concatenate()([encoder.get_layer('block1_conv2').output, up5])
upblock5_conv1 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(up_merge4)
upblock5_conv2 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(upblock5_conv1)

output_layer = layers.Conv2D(1, 1, padding='same', activation=last_activation)(upblock5_conv2)

model = models.Model(inputs=encoder.get_layer('input_7').output, outputs=output_layer)


# # Small encoder network
# #encoder
# input_shape = (128, 128, 3)
# input_tensor = layers.Input(input_shape)
# block1_conv1 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(input_tensor)
# block1_conv2 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(block1_conv1)
# pool1 = layers.MaxPooling2D((2, 2))(block1_conv2)
#
# block2_conv1 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(pool1)
# block2_conv2 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(block2_conv1)
# pool2 = layers.MaxPooling2D((2, 2))(block2_conv2)
#
# #decoder
# unpool2 = layers.UpSampling2D((2, 2))(pool2)
# up_merge = layers.Concatenate()([pool2, unpool2])
# upblock2_conv1 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(up_merge)
# upblock2_conv2 = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(upblock2_conv1)
#
# unpool1 = layers.UpSampling2D((2, 2))(upblock2_conv2)
# up_merge2 = layers.Concatenate()(pool1, unpool1)
# upblock1_conv1 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(up_merge2)
# upblock1_conv2 = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(upblock1_conv1)
# output_tensor = layers.Conv2D(1, 1, padding='valid', activation=last_activation)(upblock1_conv2)
#
# model = models.Model(inputs=input_tensor, outputs=output_tensor)
