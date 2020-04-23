from keras import Input, layers
from keras.models import Model
from layers import MaxUnpooling2D, MaxPoolingWithArgmax2D

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
    :return: model with SegNet architecture, which is similar to UNet
    '''
    #encoder
    input_tensor = Input(input_shape)

    conv1_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(input_tensor)
    conv1_1 = layers.BatchNormalization()(conv1_1)
    conv1_1 = layers.Activation('relu')(conv1_1)
    conv1_2 = layers.Conv2D(filters, conv_kernel_size, padding='same')(conv1_1)
    conv1_2 = layers.BatchNormalization()(conv1_2)
    conv1_2 = layers.Activation('relu')(conv1_2)
    pool1, ind1 = MaxPoolingWithArgmax2D((2,2))(conv1_2)

    conv2_1 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(pool1)
    conv2_1 = layers.BatchNormalization()(conv2_1)
    conv2_1 = layers.Activation('relu')(conv2_1)
    conv2_2 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(conv2_1)
    conv2_2 = layers.BatchNormalization()(conv2_2)
    conv2_2 = layers.Activation('relu')(conv2_2)
    conv2_3 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(conv2_2)
    conv2_3 = layers.BatchNormalization()(conv2_3)
    conv2_3 = layers.Activation('relu')(conv2_3)
    pool2, ind2 = MaxPoolingWithArgmax2D((2,2))(conv2_3)

    conv3_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(pool2)
    conv3_1 = layers.BatchNormalization()(conv3_1)
    conv3_1 = layers.Activation('relu')(conv3_1)
    conv3_2 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv3_1)
    conv3_2 = layers.BatchNormalization()(conv3_2)
    conv3_2 = layers.Activation('relu')(conv3_2)
    conv3_3 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv3_2)
    conv3_3 = layers.BatchNormalization()(conv3_3)
    conv3_3 = layers.Activation('relu')(conv3_3)
    pool3, ind3 = MaxPoolingWithArgmax2D((2,2))(conv3_3)

    conv4_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(pool3)
    conv4_1 = layers.BatchNormalization()(conv4_1)
    conv4_1 = layers.Activation('relu')(conv4_1)
    conv4_2 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv4_1)
    conv4_2 = layers.BatchNormalization()(conv4_2)
    conv4_2 = layers.Activation('relu')(conv4_2)
    conv4_3 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv4_2)
    conv4_3 = layers.BatchNormalization()(conv4_3)
    conv4_3 = layers.Activation('relu')(conv4_3)
    pool4, ind4 = MaxPoolingWithArgmax2D((2,2))(conv4_3)

    conv5_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(pool4)
    conv5_1 = layers.BatchNormalization()(conv5_1)
    conv5_1 = layers.Activation('relu')(conv5_1)
    conv5_2 = layers.Conv2D(filters * 16, conv_kernel_size, padding="same")(conv5_1)
    conv5_2 = layers.BatchNormalization()(conv5_2)
    conv5_2 = layers.Activation('relu')(conv5_2)
    conv5_3 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(conv5_2)
    conv5_3 = layers.BatchNormalization()(conv5_3)
    conv5_3 = layers.Activation('relu')(conv5_3)
    pool5, ind5 = MaxPoolingWithArgmax2D((2,2))(conv5_3)

    #decoder
    up1 = MaxUnpooling2D((2,2))([ind5, pool5])
    conv6_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(up1)
    conv6_1 = layers.BatchNormalization()(conv6_1)
    conv6_1 = layers.Activation('relu')(conv6_1)
    conv6_2 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(conv6_1)
    conv6_2 = layers.BatchNormalization()(conv6_2)
    conv6_2 = layers.Activation('relu')(conv6_2)
    conv6_3 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(conv6_2)
    conv6_3 = layers.BatchNormalization()(conv6_3)
    conv6_3 = layers.Activation('relu')(conv6_3)

    up2 = MaxUnpooling2D((2,2))([ind4, conv6_3])
    conv7_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(up2)
    conv7_1 = layers.BatchNormalization()(conv7_1)
    conv7_1 = layers.Activation('relu')(conv7_1)
    conv7_2 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv7_1)
    conv7_2 = layers.BatchNormalization()(conv7_2)
    conv7_2 = layers.Activation('relu')(conv7_2)
    conv7_3 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv7_2)
    conv7_3 = layers.BatchNormalization()(conv7_3)
    conv7_3 = layers.Activation('relu')(conv7_3)

    model = 'model to be added here'
    return model
