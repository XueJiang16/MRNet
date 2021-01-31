
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
import tensorflow_addons as tfa
import keras

def _bn_relu(layer, config, dropout=0, name=None):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(config.conv_activation, name=name)(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(config.conv_dropout)(layer)

    return layer


def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        config,
        subsample_length=1):
    from keras.layers import Conv1D
    from keras import regularizers
    from keras.layers import Add
    layer1 = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=config.conv_init,
        kernel_regularizer=regularizers.l2(0.001))(layer)
    layer = layer1
    return layer


def add_conv_layers(layer, config):
    for subsample_length in config.conv_subsample_lengths:
        layer = add_conv_weight(
                    layer,
                    config.conv_filter_length,
                    config.conv_num_filters_start,
                    config,
                    subsample_length=subsample_length)
        layer = _bn_relu(layer, config)
    return layer


def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        config):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % config.conv_increase_channels_at) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(config.conv_num_skip):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                config,
                dropout=config.conv_dropout if i > 0 else 0
                )
        layer = add_conv_weight(
            layer,
            config.conv_filter_length,
            num_filters,
            config,
            subsample_length if i == 0 else 1)
    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index, num_start_filters, config):
    return 2**int(index / config.conv_increase_channels_at) \
        * num_start_filters

def channel_attention(layer):
    layer_C = keras.layers.GlobalAveragePooling1D()(layer)
    layer_C = Dense(24, activation= 'relu')(layer_C)
    layer_C = Dense(96, activation= 'sigmoid')(layer_C)
    multiplied = tf.keras.layers.Multiply()([layer, layer_C])
    return multiplied

def add_fpn_layer(layer1,layer2,filters1,filters2,upsample,config):
    from keras.layers import Conv1D
    from keras import regularizers
    layer1= Conv1D(
        filters=filters2,
        kernel_size=1,
        strides=1,
        padding='same')(layer1)
    layer1 = _bn_relu(layer1, config)
    layer1 = channel_attention(layer1)
    layer2 = Conv1D(
        filters=filters2,
        kernel_size=1,
        strides=1,
        padding='same')(layer2)
    layer2 = _bn_relu(layer2, config)
    layer2 = channel_attention(layer2)
    layer2 = tf.keras.layers.UpSampling1D(size=upsample)(layer2)
    layer = tf.keras.layers.Add()([layer1,layer2])
    return layer


# ================================fpn-1
def add_resnet_layers(layer, config):
    layer = add_conv_weight(
        layer,
        config.conv_filter_length,
        config.conv_num_filters_start,
        config,
        subsample_length=1)
    layer = _bn_relu(layer, config)
    for index, subsample_length in enumerate(config.conv_subsample_lengths):
        num_filters = get_num_filters_at_index(
            index, config.conv_num_filters_start, config)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            config)
        if index == 3:
            layer1 = layer
            filter1 = num_filters
        elif index ==7:
            layer2 =layer
            filter2 = num_filters
    layer1 = _bn_relu(layer1, config)
    layer2 = _bn_relu(layer2, config)
    layer = add_fpn_layer(layer1,layer2,filter1,filter2,4,config)
    return layer, layer2


def add_output_layer(layer, config):
    layer = tfa.layers.AdaptiveAveragePooling1D(output_size=4)(layer)
    from keras.layers import Dropout
    # layer = Dropout(0.3)(layer)
    layer = keras.layers.Flatten()(layer)
    layer = Dense(80, activation='relu')(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(50, activation='relu')(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(30, activation='relu')(layer)
    layer = Dense(config.num_categories)(layer)
    return Activation('softmax')(layer)


def add_compile(model, config):
    from keras.optimizers import SGD
    optimizer = SGD(lr=config.lr_schedule(0), momentum=0.9)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])


def build_network(config):
    inputs = Input(shape=config.input_shape,
                   dtype='float32',
                   name='inputs')
    layer1,layer2 = add_resnet_layers(inputs, config)
    output_1 = add_output_layer(layer1, config)
    output_2 = add_output_layer(layer2, config)
    output = 0.5 * output_1 + 0.5 * output_2
    model = Model(inputs=[inputs], outputs=[output])
    add_compile(model, config)
    return model

if __name__ == '__main__':
    from Config import Config
    # import time
    config = Config()
    model = build_network(config)
    model.summary()
