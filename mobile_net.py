from keras import backend
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation, \
    Input, GlobalAveragePooling2D, Reshape, Dropout
from keras.models import Model

def MobileNet(shape=[224, 224, 3], depth_multiplier=1, dropout_rate=1e-3, classes=1000):
    image_input = Input(shape=shape)

    t = Conv2D(32, kernel=(3, 3),
               padding='same',
               use_bias=False,
               strides=(2, 2),
               name='conv1')(image_input)
    t = BatchNormalization(name='conv1_bn')(t)
    t = Activation(backend.relu(t, max_value=6), name='conv1_relu')(t)

    curr_filters = 64

    for i in range(0, 14):
        if i == 0:
            t = _conv_block(t, curr_filters, depth_multiplier, block_id=i + 1)
        elif i == 1 or i == 2:
            if curr_filters != 128:
                t = _conv_block(t, curr_filters, depth_multiplier, strides=(2, 2), block_id=i + 1)
                curr_filters = 128
            t = _conv_block(t, curr_filters, depth_multiplier, block_id=i + 1)
        elif i == 3 or i == 4:
            if curr_filters != 256:
                t = _conv_block(t, curr_filters, depth_multiplier, strides=(2, 2), block_id=i + 1)
                curr_filters = 256
            t = _conv_block(t, curr_filters, depth_multiplier, block_id=i + 1)
        elif i >= 5 and i <= 10:
            if curr_filters != 512:
                t = _conv_block(t, curr_filters, depth_multiplier, strides=(2, 2), block_id=i + 1)
                curr_filters = 512
            t = _conv_block(t, curr_filters, depth_multiplier, block_id=i + 1)
        else:
            if curr_filters != 1024:
                t = _conv_block(t, curr_filters, depth_multiplier, strides=(2, 2), block_id=i + 1)
                curr_filters = 1024
            t = _conv_block(t, curr_filters, depth_multiplier, block_id=i + 1)

    t = GlobalAveragePooling2D()(t)
    t = Reshape((1, 1, 1024), name='reshape_1')(t)
    t = Dropout(dropout_rate, name='dropout')(t)

    t = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(t)
    t = Activation('softmax', name='act_softmax')(t)
    t = Reshape((classes, ), name='reshape_2')(t)

    inputs = image_input
    return Model(inputs, t, name='mobilenet_1_0_224_tf')


def _conv_block(inputs, filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    t = DepthwiseConv2D((3, 3),
                        strides=strides,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        use_bias=False,
                        name='dw_conv_%d' % block_id)(inputs)

    t = BatchNormalization(name='dw_conv_%d_bn' % block_id)(t)
    t = Activation(backend.relu(t, max_value=6), name='dw_conv_%d_relu' % block_id)(t)

    t = Conv2D(filters, (1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               name='pw_conv_%d' % block_id)(t)
    t = BatchNormalization(name='pw_conv_%d_bn' % block_id)(t)

    return Activation(backend.relu(t, max_value=6), name='pw_conv_%d_relu' % block_id)(t)
