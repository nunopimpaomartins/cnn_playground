from keras import Input, layers
from keras.models import Model


def unet_3d(input_shape=(4, 128, 128, 1), conv_kernel_size=(3, 3, 3), pooling_kernel=(2, 2, 2), filters=64, last_activation='linear'):
    """
    Small 3D Unet for channel prediction.
    :param input_shape: shape of the input images
    :param conv_kernel_size: kernel size for convolutions
    :param pooling_kernel: kernel size for the pooling operation
    :param filters: initial number of filters
    :param last_activation: last activation layer
    :return: returns model of network
    """
    input_tensor = Input(shape=input_shape)
    block1_conv1 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(input_tensor)
    block1_conv2 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(block1_conv1)
    pool1 = layers.MaxPool3D(pooling_kernel)(block1_conv2)

    block2_conv1 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(pool1)
    block2_conv2 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(block2_conv1)
    pool2 = layers.MaxPool3D(pooling_kernel)(block2_conv2)

    block3_conv1 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(pool2)
    block3_conv2 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(block3_conv1)

    up2 = layers.UpSampling3D(pooling_kernel)(block3_conv2)
    up2 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(up2)
    merge2 = layers.Concatenate(axis=-1)([block2_conv2, up2])
    block4_conv1 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(merge2)
    block4_conv2 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(block4_conv1)

    up1 = layers.UpSampling3D(pooling_kernel)(block4_conv2)
    up1 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(up1)
    merge1 = layers.Concatenate(axis=-1)([block1_conv2, up1])
    block5_conv1 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(merge1)
    block5_conv2 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(block5_conv1)

    output_tensor = layers.Conv3D(1, (1, 1, 1), padding='same', activation=last_activation)(block5_conv2)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model
