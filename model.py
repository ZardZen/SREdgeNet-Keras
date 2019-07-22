from keras.layers import Add, Conv2D, Input, Lambda, Activation, Concatenate
from keras.models import Model
import numpy as np
import tensorflow as tf

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def Normalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)

def Denormalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)
  
def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x

def res_block(x_in, filters, scaling):

    x = Conv2D(filters, 3, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x

def upsample(x, scale, num_filters):

    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return SubpixelConv2D(factor)(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
  
def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None, tanh_activation=False):
    x_in = Input(shape=(None, None, 3))
    x = Normalization()(x_in)
   
    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    if tanh_activation:
        x = Activation('tanh')(x)
        x = Denormalization_m11()(x)

    else:
        x = Denormalization()(x)

    return Model(x_in, x, name="edsr")


def rcf(input_shape=None):
    # Input
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: x / 255, name='pre-process')(inputs)

    # Block 1
    x1_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x1_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b1_conv1_out')(x1_conv1)
    x1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1_conv1)
    x1_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b1_conv2_o2')(x1_conv2)

    x1_add = Add()([x1_conv1_out, x1_conv2_out])
    b1 = Conv2D(1, (1, 1), activation=None, padding='same')(x1_add)
    b1 = side_branch(x1_add,1)

    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1_conv2)


    # Block 2
    x2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    x2_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b2_conv1_out')(x2_conv1)
    x2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x2_conv1)
    x2_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b2_conv2_out')(x2_conv2)

    x2_add = Add()([x2_conv1_out, x2_conv2_out])
    b2 = side_branch(x2_add,2)


    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2_conv2)


    # Block 3
    x3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    x3_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b3_conv1_out')(x3_conv1)
    x3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x3_conv1)
    x3_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b3_conv2_out')(x3_conv2)
    x3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x3_conv2)
    x3_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b3_conv3_out')(x3_conv3)

    x3_add = Add()([x3_conv1_out, x3_conv2_out, x3_conv3_out])
    b3 = side_branch(x3_add,4)

    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3_conv3)


    # Block 4
    x4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    x4_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b4_conv1_out')(x4_conv1)
    x4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x4_conv1)
    x4_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b4_conv2_out')(x4_conv2)
    x4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x4_conv2)
    x4_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b4_conv3_out')(x4_conv3)

    x4_add = Add()([x4_conv1_out, x4_conv2_out, x4_conv3_out])
    b4 = side_branch(x4_add,8)
    
    x4 = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='block4_pool')(x4_conv3)


    # Block 5
    x5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x4)
    x5_conv1_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b5_conv1_out')(x5_conv1)
    x5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x5_conv1)
    x5_conv2_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b5_conv2_out')(x5_conv2)
    x5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x5_conv2)
    x5_conv3_out = Conv2D(21, (1, 1), activation='relu', padding='same', name='b5_conv3_out')(x5_conv3)

    x5_add = Add()([x5_conv1_out, x5_conv2_out, x5_conv3_out]) 
    b5 = side_branch(x5_add,8)

    # fuse
    fuse = Concatenate(axis= -1)([b1, b2, b3, b4, b5])
    fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse)

    # outputs

    outputs = Activation('sigmoid', name='ofuse')(fuse)

    # model
    model = Model(inputs, outputs)
    return model

def merge(scale, num_filters=128, num_res_blocks=16, res_block_scaling=None, tanh_activation=False):
    x_in = Input(shape=(None, None, 3))
    x_mask = Input(shape=(None,None,1))
    #x = Normalization()(x_in)
    x = b = Concatenate(axis= -1)([x_in, x_mask])
    
    a = Conv2D(num_filters,3,padding='same')(x_mask)
   
    c = b = Conv2D(num_filters, 3, padding='same')(b)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    x = Add()([c, b, a])
    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

#     if tanh_activation:
#         x = Activation('tanh')(x)
#         x = Denormalization_m11()(x)

#     else:
#         x = Denormalization()(x)

    return Model([x_in,x_mask], x)

def SREdgeNet():
  
  
  
  
