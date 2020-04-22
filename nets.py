from keras import Input, layers
from keras.models import Model


def unet(input_shape, conv_kernel_size, dropout, filters, last_activation):
    '''
    This is the implementation of  a 2D UNet with keras. For practice purposes

    :param input_shape:  shape of the input image.
    :param conv_kernel_size: shape of the convolutiuon kernel to be used. Default should be 3
    :param dropout: dropout rate used in the last conv layer and before upsampling
    :param filters: initial number of filters in the net
    :param last_activation: last activation layer of the networkl. Current default is 'sigmoid'
    :return: returns a network as a model
    '''
    input_tensor = Input(shape=input_shape)
    conv1 = layers.Conv2D(filters, conv_kernel_size, padding='same', activation='relu')(input_tensor)
    conv1 = layers.Conv2D(filters, conv_kernel_size, padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPool2D((2, 2))(conv1)
    conv2 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPool2D((2, 2))(conv2)
    conv3 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D((2, 2))(conv3)
    conv4 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same', activation='relu')(pool3)
    conv4 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same', activation='relu')(conv4)
    drop4 = layers.Dropout(dropout)(conv4)
    pool4 = layers.MaxPool2D((2, 2))(drop4)

    conv5 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same', activation='relu')(pool4)
    conv5 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same', activation='relu')(conv5)
    drop5 = layers.Dropout(dropout)(conv5)

    up6 = layers.UpSampling2D((2, 2))(drop5)
    up6 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same', activation='relu')(up6)
    # crop4 = layers.
    merged6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same', activation='relu')(merged6)
    conv6 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same', activation='relu')(conv6)

    up7 = layers.Conv2D(filters * 4, 2, padding='same', activation='relu')(layers.UpSampling2D((2, 2))(conv6))
    merged7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same', activation='relu')(merged7)
    conv7 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same', activation='relu')(conv7)

    up8 = layers.Conv2D(filters * 2, 2, padding='same', activation='relu')(layers.UpSampling2D((2, 2))(conv7))
    merged8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same', activation='relu')(merged8)
    conv8 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same', activation='relu')(conv8)

    up9 = layers.Conv2D(filters, 2, padding='same', activation='relu')(layers.UpSampling2D((2, 2))(conv8))
    merged9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(filters, conv_kernel_size, padding='same', activation='relu')(merged9)
    conv9 = layers.Conv2D(filters, conv_kernel_size, padding='same', activation='relu')(conv9)
    conv9 = layers.Conv2D(2, conv_kernel_size, padding='same', activation='relu')(conv9)
    output_tensor = layers.Conv2D(1, 1, activation=last_activation)(conv9)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


def segnet(input_shape, conv_kernel_size, padding, dropout, filters, last_activation):
    '''
    Function retunns a model with the SegNet implementation in keras.
    paper reference:
    [1] V. Badrinarayanan et al IEEE Trans. Pattern Anal. Mach. Intell. 2017
    :param input_shape:
    :param conv_kernel_size:
    :param padding:
    :param dropout:
    :param filters:
    :param last_activation:
    :return:
    '''
    model = 'model to be added here'
    return model
