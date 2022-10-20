import glob
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, add, multiply, BatchNormalization, Dense, Dropout, Input, Average, Conv1D, Concatenate, AvgPool1D, UpSampling1D, Reshape, Activation
from custom_metrics import *

NB_RESIDUES = 20
UPPER_LENGTH_LIMIT = 704
CLASSIFIER_COMPILE_SETTINGS = {'optimizer': 'adam', 'loss': mcc_cc_loss, 'metrics': masked_acc}


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=2), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv1D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv1D(filters=inter_shape, kernel_size=3, strides=shape_x[1] // shape_g[1], padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv1D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling1D(size=shape_x[1] // shape_sigmoid[1])(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[2])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv1D(filters=shape_x[2], kernel_size=1, strides=1, padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output


def dense_classifier(input, nb_units: int = 1024, dropout = 0.5):
    model = Dense(nb_units, activation='relu')(input)
    model = Dropout(dropout)(model)
    model = Dense(nb_units // 2, activation='relu')(model)
    model = Dropout(dropout)(model)
    return model


def unet_conv_layer(input, filters):
    out = Conv1D(filters, 7, padding='same', activation='relu')(input)
    return out


def unet_contracting(input, filters=64):

    def down_block(input, filters):
        conv = unet_conv_layer(input, filters)
        conv = Dropout(0.1)(conv)
        conv = unet_conv_layer(conv, filters)
        maxpool = AvgPool1D()(conv)

        return maxpool, conv

    step_down, bridge_1 = down_block(input, filters)
    step_down, bridge_2 = down_block(step_down, filters)
    step_down, bridge_3 = down_block(step_down, 2 * filters)
    step_down, bridge_4 = down_block(step_down, 2 * filters)

    return step_down, bridge_1, bridge_2, bridge_3, bridge_4


def unet_expanding(step_down, bridge_1, bridge_2, bridge_3, bridge_4, filters=64):

    def up_block(input, bridged, filters):
        att = AttnGatingBlock(bridged, input, filters)
        conv = unet_conv_layer(input, filters)
        conv = Concatenate(axis=2)([att, UpSampling1D()(conv)])
        conv = Dropout(0.1)(conv)
        up = unet_conv_layer(conv, filters)

        return up

    step_up = up_block(step_down, bridge_4, 2 * filters)
    step_up = up_block(step_up, bridge_3, 2 * filters)
    step_up = up_block(step_up, bridge_2, filters)
    step_up = up_block(step_up, bridge_1, filters)

    step_up = Dense(filters, activation="relu")(step_up)

    return step_up


def unet(inputs, nb_filters, output_filters=64):
    step_downs = []
    b1s = []
    b2s = []
    b3s = []
    b4s = []

    for input, filter in zip(inputs, nb_filters):
        step_down, bridge_1, bridge_2, bridge_3, bridge_4 = unet_contracting(input, filter)
        step_downs.append(step_down)
        b1s.append(bridge_1)
        b2s.append(bridge_2)
        b3s.append(bridge_3)
        b4s.append(bridge_4)

    step_down = Concatenate(axis=2)(step_downs)
    bridge_1 = Concatenate(axis=2)(b1s)
    bridge_2 = Concatenate(axis=2)(b2s)
    bridge_3 = Concatenate(axis=2)(b3s)
    bridge_4 = Concatenate(axis=2)(b4s)

    step_up = unet_expanding(step_down, bridge_1, bridge_2, bridge_3, bridge_4, output_filters)

    return step_up


def unet_classifier():
    input_lm_embedds = Input(shape=(UPPER_LENGTH_LIMIT, 1024, 1))
    flat_lm_embedds = Reshape((UPPER_LENGTH_LIMIT, 1024))(input_lm_embedds)

    input_sequence = Input(shape=(UPPER_LENGTH_LIMIT, NB_RESIDUES, 1))
    flat_sequence = Reshape((UPPER_LENGTH_LIMIT, NB_RESIDUES))(input_sequence)

    model = unet([flat_lm_embedds, flat_sequence], [64, 32], 64)

    out_q8_relu = Dense(8, activation="relu")(model)
    out_q8 = Activation("softmax", name='q8')(out_q8_relu)

    model = Model(inputs=[input_lm_embedds, input_sequence], outputs=out_q8)
    model.compile(**CLASSIFIER_COMPILE_SETTINGS)

    return model   
