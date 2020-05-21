from keras.layers import Conv2D, Input, PReLU, MaxPool2D, Permute, Flatten, Dense
from keras.models import Model

# Capture faces and decide on the existence of huamn face
def create_Pnet(weight_path):
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model

# Bounding box refinement
def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])

    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model
