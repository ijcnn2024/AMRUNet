"""
    Models used in experiments
"""

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate,\
    BatchNormalization, Activation, add,multiply
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def AMRUNet(length, n_channel=1):
    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

        kernel = 3

        x = Conv1D(filters, kernel, padding=padding)(x)
        x = BatchNormalization()(x)

        if (activation == None):
            return x

        x = Activation(activation, name=name)(x)
        return x

    def MultiResBlock(U, inp, alpha=2.5):

        W = alpha * U

        shortcut = inp

        shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                             int(W * 0.5), 1, 1, activation=None, padding='same')

        conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = BatchNormalization()(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        return out

    def ResPath(filters, length, inp):

        shortcut = inp
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        for i in range(length - 1):
            shortcut = out
            shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                 activation=None, padding='same')

            out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

        return out

    def AttentionGate(F_g, F_l, F_int):

        g_conv = Conv1D(F_int, 1, activation='relu')(F_g)
        l_conv = Conv1D(F_int, 1, activation='relu')(F_l)

        gate = add([g_conv, l_conv])
        gate = Activation('sigmoid')(gate)

        gated_l = multiply([F_l, gate])

        return gated_l

    inputs = Input((length, n_channel))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
    up6 = AttentionGate(UpSampling1D(size=2)(mresblock5), mresblock4, 32 * 8)
    mresblock6 = MultiResBlock(32 * 8, up6)

    up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
    up7 = AttentionGate(UpSampling1D(size=2)(mresblock6), mresblock3, 32 * 4)
    mresblock7 = MultiResBlock(32 * 4, up7)

    up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
    up8 = AttentionGate(UpSampling1D(size=2)(mresblock7), mresblock2, 32 * 2)
    mresblock8 = MultiResBlock(32 * 2, up8)

    up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
    up9 = AttentionGate(UpSampling1D(size=2)(mresblock8), mresblock1, 32)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = Conv1D(1, 1)(mresblock9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model





if __name__ == '__main__':


    model = AMRUNet(length=1024, n_channel=1)
    model.summary()
