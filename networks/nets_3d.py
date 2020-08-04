from keras import Input, layers
from keras.models import Model


def unet_3d(input_shape, conv_kernel_size, filters, last_activation):
    """
    Small 3D Unet for channel prediction.
    :param input_shape:
    :param conv_kernel_size:
    :param filters:
    :param last_activation:
    :return:
    """
    input_tensor = Input(shape=input_shape)
    block1_conv1 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(input_tensor)
    block1_conv2 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(block1_conv1)
    pool1 = layers.MaxPool3D((2, 2, 2))(block1_conv2)

    block2_conv1 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(pool1)
    block2_conv2 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(block2_conv1)
    pool2 = layers.MaxPool3D((2, 2, 2))(block2_conv2)

    block3_conv1 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(pool2)
    block3_conv2 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(block3_conv1)
    pool3 = layers.MaxPool3D((2, 2, 2,))(block3_conv2)

    block4_conv1 = layers.Conv3D(filters * 8, conv_kernel_size, padding='same', activation='relu')(pool3)
    block4_conv2 = layers.Conv3D(filters * 8, conv_kernel_size, padding='same', activation='relu')(block4_conv1)

    up3 = layers.UpSampling3D((2, 2, 2))(block4_conv2)
    up3 = layers.Conv3D(filters * 4, 2, padding='same', activation='relu')(up3)
    merge3 = layers.Concatenate(axis=-1)([block3_conv2, up3])
    block5_conv1 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(merge3)
    block5_conv2 = layers.Conv3D(filters * 4, conv_kernel_size, padding='same', activation='relu')(block5_conv1)

    up2 = layers.UpSampling3D((2, 2, 2))(block5_conv2)
    up2 = layers.Conv3D(filters * 2, 2, padding='same', activation='relu')(up2)
    merge2 = layers.Concatenate(axis=-1)([block2_conv2, up2])
    block6_conv1 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(merge2)
    block6_conv2 = layers.Conv3D(filters * 2, conv_kernel_size, padding='same', activation='relu')(block6_conv1)

    up1 = layers.UpSampling3D((2, 2, 2))(block4_conv2)
    up1 = layers.Conv3D(filters, 2, padding='same', activation='relu')(up1)
    merge1 = layers.Concatenate(axis=-1)([block1_conv2, up1])
    block7_conv1 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(merge1)
    block7_conv2 = layers.Conv3D(filters, conv_kernel_size, padding='same', activation='relu')(block7_conv1)

    output_tensor = layers.Conv3D(1, (1, 1, 1), padding='same', activation=last_activation)(block7_conv2)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model
