# The MIT-Zero License

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from enum import Enum
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam, RMSprop
from losses import Loss


class ArCnnModel():
    def __init__(self,
                 input_dim,
                 num_filters,
                 growth_factor,
                 num_layers,
                 dropout_rate_encoder,
                 dropout_rate_decoder,
                 batch_norm_encoder,
                 batch_norm_decoder,
                 learning_rate,
                 optimizer_enum,
                 pre_trained=None):

        # PianoRoll Input Dimensions
        self.input_dim = input_dim
        # Number of filters in the convolution
        self.num_filters = num_filters
        # Growth rate of number of filters at each convolution
        self.growth_factor = growth_factor
        # Number of Encoder and Decoder layers
        self.num_layers = num_layers
        # A list of dropout values at each encoder layer
        self.dropout_rate_encoder = dropout_rate_encoder
        # A list of dropout values at each decoder layer
        self.dropout_rate_decoder = dropout_rate_decoder
        # A list of flags to ensure if batch_nromalization at each encoder
        self.batch_norm_encoder = batch_norm_encoder
        # A list of flags to ensure if batch_nromalization at each decoder
        self.batch_norm_decoder = batch_norm_decoder
        # Path to pretrained Model
        self.pre_trained = pre_trained
        # Learning rate for the model
        self.learning_rate = learning_rate
        # Optimizer to use while training the model
        self.optimizer_enum = optimizer_enum
        if self.num_layers < 1:
            raise ValueError(
                "Number of layers should be greater than or equal to 1")

    # Number of times Conv2D to be performed
    CONV_PER_LAYER = 2

    def down_sampling(self,
                      layer_input,
                      num_filters,
                      batch_normalization=False,
                      dropout_rate=0):
        '''
        :param: layer_input: Input Layer to the downsampling block
        :param: num_filters: Number of filters
        :param: batch_normalization: Flag to check if batch normalization to be performed
        :param: dropout_rate: To regularize overfitting
        '''
        encoder = layer_input
        for _ in range(self.CONV_PER_LAYER):
            encoder = Conv2D(num_filters, (3, 3),
                             activation='relu',
                             padding='same')(encoder)
            pooling_layer = MaxPooling2D(pool_size=(2, 2))(encoder)
            if dropout_rate:
                pooling_layer = Dropout(dropout_rate)(pooling_layer)
            if batch_normalization:
                pooling_layer = BatchNormalization()(pooling_layer)
        return encoder, pooling_layer

    def up_sampling(self,
                    layer_input,
                    skip_input,
                    num_filters,
                    batch_normalization=False,
                    dropout_rate=0):
        '''
        :param: layer_input: Input Layer to the downsampling block
        :param: num_filters: Number of filters
        :param: batch_normalization: Flag to check if batch normalization to be performed
        :param: dropout_rate: To regularize overfitting
        '''
        decoder = concatenate(
            [UpSampling2D(size=(2, 2))(layer_input), skip_input])
        if batch_normalization:
            decoder = BatchNormalization()(decoder)
        for _ in range(self.CONV_PER_LAYER):
            decoder = Conv2D(num_filters, (3, 3),
                             activation='relu',
                             padding='same')(decoder)

        if dropout_rate:
            decoder = Dropout(dropout_rate)(decoder)
        return decoder

    def get_optimizer(self, optimizer_enum, learning_rate):
        '''
        Use either Adam or RMSprop.
        '''
        if OptimizerType.ADAM == optimizer_enum:
            optimizer = Adam(lr=learning_rate)
        elif OptimizerType.RMSPROP == optimizer_enum:
            optimizer = RMSprop(lr=learning_rate)
        else:
            raise Exception("Only Adam and RMSProp optimizers are supported")
        return optimizer

    def build_model(self):
        # Create a list of encoder sampling layers
        down_sampling_layers = []
        up_sampling_layers = []
        inputs = Input(self.input_dim)
        layer_input = inputs
        num_filters = self.num_filters
        # encoder samplimg layers
        for layer in range(self.num_layers):
            encoder, pooling_layer = self.down_sampling(
                layer_input=layer_input,
                num_filters=num_filters,
                batch_normalization=self.batch_norm_encoder[layer],
                dropout_rate=self.dropout_rate_encoder[layer])

            down_sampling_layers.append(encoder)
            layer_input = pooling_layer  # Get the previous pooling_layer_input
            num_filters *= self.growth_factor

        # bottle_neck layer
        bottle_neck = Conv2D(num_filters, (3, 3),
                             activation='relu',
                             padding='same')(pooling_layer)
        bottle_neck = Conv2D(num_filters, (3, 3),
                             activation='relu',
                             padding='same')(bottle_neck)
        num_filters //= self.growth_factor

        # upsampling layers
        decoder = bottle_neck
        for index, layer in enumerate(reversed(down_sampling_layers)):
            decoder = self.up_sampling(
                layer_input=decoder,
                skip_input=layer,
                num_filters=num_filters,
                batch_normalization=self.batch_norm_decoder[index],
                dropout_rate=self.dropout_rate_decoder[index])
            up_sampling_layers.append(decoder)
            num_filters //= self.growth_factor

        output = Conv2D(1, 1, activation='linear')(up_sampling_layers[-1])
        model = Model(inputs=inputs, outputs=output)
        optimizer = self.get_optimizer(self.optimizer_enum, self.learning_rate)
        model.compile(optimizer=optimizer, loss=Loss.built_in_softmax_kl_loss)
        if self.pre_trained:
            model.load_weights(self.pre_trained)
        model.summary()
        return model


class OptimizerType(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"
