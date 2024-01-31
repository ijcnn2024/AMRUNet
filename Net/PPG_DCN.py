from keras.layers import Input, Activation
from keras.models import Model
import complexnn
# from ComplexNN_EDAN import complex_Conv3DTranspose
import tensorflow as tf

def PPGModel():
    input_shape = (129, 129, 2)
    input_layer = Input(input_shape, name='inputs')
    conv1 = complexnn.ComplexConv2D(filters=18, kernel_size=(9, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = complexnn.bn.ComplexBatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = complexnn.ComplexConv2D(filters=30, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv1)
    conv2 = complexnn.bn.ComplexBatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = complexnn.ComplexConv2D(filters=8, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv2)
    conv3 = complexnn.bn.ComplexBatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = complexnn.ComplexConv2D(filters=18, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv3)
    conv4 = complexnn.bn.ComplexBatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = complexnn.ComplexConv2D(filters=30, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv4)
    conv5 = complexnn.bn.ComplexBatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = complexnn.ComplexConv2D(filters=8, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv5)
    conv6 = complexnn.bn.ComplexBatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = complexnn.ComplexConv2D(filters=18, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv6)
    conv7 = complexnn.bn.ComplexBatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = complexnn.ComplexConv2D(filters=30, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv7)
    conv8 = complexnn.bn.ComplexBatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = complexnn.ComplexConv2D(filters=8, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv8)
    conv9 = complexnn.bn.ComplexBatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = complexnn.ComplexConv2D(filters=18, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv9)
    conv10 = complexnn.bn.ComplexBatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    conv11 = complexnn.ComplexConv2D(filters=30, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv10)
    conv11 = complexnn.bn.ComplexBatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)

    conv12 = complexnn.ComplexConv2D(filters=8, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv11)
    conv12 = complexnn.bn.ComplexBatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    conv13 = complexnn.ComplexConv2D(filters=18, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv12)
    conv13 = complexnn.bn.ComplexBatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)

    conv14 = complexnn.ComplexConv2D(filters=30, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv13)
    conv14 = complexnn.bn.ComplexBatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)

    conv15 = complexnn.ComplexConv2D(filters=8, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv14)
    conv15 = complexnn.bn.ComplexBatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    conv16 = complexnn.ComplexConv2D(filters=1, kernel_size=(129, 3), strides=(1, 1), padding='same')(conv15)
    # out = Activation()(conv16)
    model = Model(inputs=input_layer, outputs=conv16)
    return model

def PPGModel_plus():
    input_shape = (129, 129, 2)
    input_layer = Input(input_shape, name='inputs')
    conv1 = complexnn.ComplexConv2D(filters=36, kernel_size=(9, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = complexnn.bn.ComplexBatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = complexnn.ComplexConv2D(filters=60, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv1)
    conv2 = complexnn.bn.ComplexBatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = complexnn.ComplexConv2D(filters=16, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv2)
    conv3 = complexnn.bn.ComplexBatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = complexnn.ComplexConv2D(filters=36, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv3)
    conv4 = complexnn.bn.ComplexBatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = complexnn.ComplexConv2D(filters=60, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv4)
    conv5 = complexnn.bn.ComplexBatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = complexnn.ComplexConv2D(filters=16, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv5)
    conv6 = complexnn.bn.ComplexBatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = complexnn.ComplexConv2D(filters=36, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv6)
    conv7 = complexnn.bn.ComplexBatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = complexnn.ComplexConv2D(filters=60, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv7)
    conv8 = complexnn.bn.ComplexBatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv9 = complexnn.ComplexConv2D(filters=16, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv8)
    conv9 = complexnn.bn.ComplexBatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = complexnn.ComplexConv2D(filters=36, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv9)
    conv10 = complexnn.bn.ComplexBatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    conv11 = complexnn.ComplexConv2D(filters=60, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv10)
    conv11 = complexnn.bn.ComplexBatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)

    conv12 = complexnn.ComplexConv2D(filters=16, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv11)
    conv12 = complexnn.bn.ComplexBatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    conv13 = complexnn.ComplexConv2D(filters=36, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv12)
    conv13 = complexnn.bn.ComplexBatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)

    conv14 = complexnn.ComplexConv2D(filters=60, kernel_size=(5, 3), strides=(1, 1), padding='same')(conv13)
    conv14 = complexnn.bn.ComplexBatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)

    conv15 = complexnn.ComplexConv2D(filters=16, kernel_size=(9, 3), strides=(1, 1), padding='same')(conv14)
    conv15 = complexnn.bn.ComplexBatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    conv16 = complexnn.ComplexConv2D(filters=1, kernel_size=(129, 3), strides=(1, 1), padding='same')(conv15)
    # out = Activation()(conv16)
    model = Model(inputs=input_layer, outputs=conv16)
    return model

if __name__ == "__main__":
    model = PPGModel_plus()
    model.summary()