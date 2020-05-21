from keras import Input, layers
from keras.models import Model
from .layers import MaxUnpooling2D, MaxPoolingWithArgmax2D


def unet(input_shape, conv_kernel_size, dropout, filters, last_activation):
    """
    This is the implementation of  a 2D UNet with keras. For practice purposes

    :param input_shape:  shape of the input image.
    :param conv_kernel_size: shape of the convolutiuon kernel to be used. Default should be 3
    :param dropout: dropout rate used in the last conv layer and before upsampling
    :param filters: initial number of filters in the net
    :param last_activation: last activation layer of the networkl. Current default is 'sigmoid'
    :return: returns a network as a model
    """
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


def segnet(input_shape, conv_kernel_size, dropout, filters, n_classes, last_activation):
    """
    Function retunns a model with the SegNet implementation in keras.
    paper reference:
    [1] V. Badrinarayanan et al IEEE Trans. Pattern Anal. Mach. Intell. 2017
    :param input_shape: image shape to be fed to the network
    :param conv_kernel_size: kernel size for the convolutional layers of the network
    :param dropout:
    :param filters:
    :param n_classes: number of labels to segment
    :param last_activation:
    :return: model with SegNet architecture, which is similar to UNet
    """
    # encoder
    input_tensor = Input(input_shape)

    conv1_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(input_tensor)
    conv1_2 = layers.BatchNormalization()(conv1_1)
    conv1_3 = layers.Activation('relu')(conv1_2)
    conv2_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(conv1_3)
    conv2_2 = layers.BatchNormalization()(conv2_1)
    conv2_3 = layers.Activation('relu')(conv2_2)

    pool1, ind1 = MaxPoolingWithArgmax2D((2, 2))(conv2_3)

    conv3_1 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(pool1)
    conv3_2 = layers.BatchNormalization()(conv3_1)
    conv3_3 = layers.Activation('relu')(conv3_2)
    conv4_1 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(conv3_3)
    conv4_2 = layers.BatchNormalization()(conv4_1)
    conv4_3 = layers.Activation('relu')(conv4_2)

    pool2, ind2 = MaxPoolingWithArgmax2D((2, 2))(conv4_3)

    conv5_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(pool2)
    conv5_2 = layers.BatchNormalization()(conv5_1)
    conv5_3 = layers.Activation('relu')(conv5_2)
    conv6_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv5_3)
    conv6_2 = layers.BatchNormalization()(conv6_1)
    conv6_3 = layers.Activation('relu')(conv6_2)
    conv7_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv6_3)
    conv7_2 = layers.BatchNormalization()(conv7_1)
    conv7_3 = layers.Activation('relu')(conv7_2)

    pool3, ind3 = MaxPoolingWithArgmax2D((2, 2))(conv7_3)

    conv8_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(pool3)
    conv8_2 = layers.BatchNormalization()(conv8_1)
    conv8_3 = layers.Activation('relu')(conv8_2)
    conv9_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv8_3)
    conv9_2 = layers.BatchNormalization()(conv9_1)
    conv9_3 = layers.Activation('relu')(conv9_2)
    conv10_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv9_3)
    conv10_2 = layers.BatchNormalization()(conv10_1)
    conv10_3 = layers.Activation('relu')(conv10_2)
    drop1 = layers.Dropout(dropout)(conv10_3)

    pool4, ind4 = MaxPoolingWithArgmax2D((2, 2))(drop1)

    conv11_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(pool4)
    conv11_2 = layers.BatchNormalization()(conv11_1)
    conv11_3 = layers.Activation('relu')(conv11_2)
    conv12_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding="same")(conv11_3)
    conv12_2 = layers.BatchNormalization()(conv12_1)
    conv12_3 = layers.Activation('relu')(conv12_2)
    conv13_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(conv12_3)
    conv13_2 = layers.BatchNormalization()(conv13_1)
    conv13_3 = layers.Activation('relu')(conv13_2)
    drop2 = layers.Dropout(dropout)(conv13_3)

    pool5, ind5 = MaxPoolingWithArgmax2D((2, 2))(drop2)

    # decoder
    upool5 = MaxUnpooling2D((2, 2))([pool5, ind5])

    conv14_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(upool5)
    conv14_2 = layers.BatchNormalization()(conv14_1)
    conv14_3 = layers.Activation('relu')(conv14_2)
    conv15_1 = layers.Conv2D(filters * 16, conv_kernel_size, padding='same')(conv14_3)
    conv15_2 = layers.BatchNormalization()(conv15_1)
    conv15_3 = layers.Activation('relu')(conv15_2)
    conv16_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv15_3)
    conv16_2 = layers.BatchNormalization()(conv16_1)
    conv16_3 = layers.Activation('relu')(conv16_2)

    upool4 = MaxUnpooling2D((2, 2))([conv16_3, ind4])

    conv17_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(upool4)
    conv17_2 = layers.BatchNormalization()(conv17_1)
    conv17_3 = layers.Activation('relu')(conv17_2)
    conv18_1 = layers.Conv2D(filters * 8, conv_kernel_size, padding='same')(conv17_3)
    conv18_2 = layers.BatchNormalization()(conv18_1)
    conv18_3 = layers.Activation('relu')(conv18_2)
    conv19_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv18_3)
    conv19_2 = layers.BatchNormalization()(conv19_1)
    conv19_3 = layers.Activation('relu')(conv19_2)

    upool3 = MaxUnpooling2D((2, 2))([conv19_3, ind3])

    conv20_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(upool3)
    conv20_2 = layers.BatchNormalization()(conv20_1)
    conv20_3 = layers.Activation('relu')(conv20_2)
    conv21_1 = layers.Conv2D(filters * 4, conv_kernel_size, padding='same')(conv20_3)
    conv21_2 = layers.BatchNormalization()(conv21_1)
    conv21_3 = layers.Activation('relu')(conv21_2)
    conv22_1 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(conv21_3)
    conv22_2 = layers.BatchNormalization()(conv22_1)
    conv22_3 = layers.Activation('relu')(conv22_2)

    upool2 = MaxUnpooling2D((2, 2))([conv22_3, ind2])

    conv23_1 = layers.Conv2D(filters * 2, conv_kernel_size, padding='same')(upool2)
    conv23_2 = layers.BatchNormalization()(conv23_1)
    conv23_3 = layers.Activation('relu')(conv23_2)
    conv24_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(conv23_3)
    conv24_2 = layers.BatchNormalization()(conv24_1)
    conv24_3 = layers.Activation('relu')(conv24_2)

    upool1 = MaxUnpooling2D((2, 2))([conv24_3, ind1])

    conv25_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(upool1)
    conv25_2 = layers.BatchNormalization()(conv25_1)
    conv25_3 = layers.Activation('relu')(conv25_2)
    conv26_1 = layers.Conv2D(filters, conv_kernel_size, padding='same')(conv25_3)
    conv26_2 = layers.BatchNormalization()(conv26_1)
    conv26_3 = layers.Activation('relu')(conv26_2)

    # conv28_1 = layers.Conv2D(32, conv_kernel_size, padding='same')(conv26_3)
    # conv28_2 = layers.BatchNormalization()(conv28_1)
    # conv28_3 = layers.Activation('relu')(conv28_2)

    conv27_1 = layers.Conv2D(n_classes, (1, 1), padding='valid')(conv26_3)
    output_tensor = layers.Activation(last_activation)(conv27_1)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model
