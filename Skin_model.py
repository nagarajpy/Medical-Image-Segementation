"""from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

from keras.models import Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Multiply, Add,SeparableConv2D,Activation,average
from keras.layers import UpSampling2D,Conv2DTranspose,GlobalAveragePooling2D,GlobalMaxPooling2D

from custom_layers.layers import MaxPoolingWithArgmax2D 
from custom_layers.layers import MaxUnpooling2D



#U-net
def build_unet(input_shape):
 
    inputs = Input(input_shape, name ='main_input')

    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    convB = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    convB = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convB)
    
    up4 = (UpSampling2D(size = (2,2))(convB))
    merge4 = concatenate([conv3,up4], axis = 3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    up5 = (UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv2,up5], axis = 3)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up6 = (UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv1,up6], axis = 3)
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    conv7 = Conv2D(1, 1, activation = 'sigmoid', name = 'boundary_output')(conv6)
    model = Model(inputs = inputs, outputs = conv7)
    return model





 