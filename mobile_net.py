from keras import backend
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation

# def MobileNet(shape=[224, 224, 3], multiplier=1, dropout_rate=1e-3, classess=1000):

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

    return Activation(backend.relu(t, max_value=6), name='pw_conv_%d_relu' % block_id)(t)
