# jiayou
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class SEANetResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_sizes=[3, 1],
                 causal=False, padding='same', compress=2, dilation_rate=1, true_skip=True, **kwargs):
        super(SEANetResnetBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.causal = causal
        self.padding = padding
        self.compress = compress
        self.true_skip = true_skip
        self.hidden = n_filters // compress
        self.dilation_rate = dilation_rate


        self.conv1 = tf.keras.layers.Conv1D(filters=self.hidden, kernel_size=kernel_sizes[0],  # 3
                                            activation=None, use_bias=False, padding='same', dilation_rate=1)


        self.elu1 = tf.keras.layers.ELU(alpha=1.0)
        self.bn1 = tf.keras.layers.GroupNormalization()


        self.conv2 = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_sizes[1],  # 1
                                            activation=None, use_bias=False, padding='same', dilation_rate=1)
        self.elu2 = tf.keras.layers.ELU(alpha=1.0)
        self.bn2 = tf.keras.layers.GroupNormalization()

        if true_skip:
            self.shortcut = layers.Lambda(lambda x: x)

        else:
            self.shortcut = layers.Conv1D(filters=n_filters, kernel_size=1, use_bias=False)

    def get_config(self):
        config = super(SEANetResnetBlock, self).get_config().copy()
        config.update({'n_filters': self.n_filters,
                       'causal': self.causal,
                       'padding': self.padding,
                       'compress': self.compress,
                       'true_skip': self.true_skip,
                       'dilation_rate': self.dilation_rate})
        return config

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.elu1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(inputs)
        out = tf.keras.layers.add([out, shortcut])

        return out

class SEANetEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters, n_residual_layers=1,
                 ratios=[4, 4, 2], dilation_rate=1, kernel_size=3, last_kernel_size=3,
                 residual_kernel_size=3, causal=False, padding='same',
                 true_skip=False, compress=2, name=None, **kwargs):
        super(SEANetEncoder, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.causal = causal
        self.padding = padding
        self.true_skip = true_skip
        self.compress = compress
        self.hop_length = np.prod(self.ratios)
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        mult = 1
        self.initial_conv = tf.keras.layers.Conv1D(filters=mult * self.n_filters,
                                                   kernel_size=self.kernel_size,
                                                   padding=self.padding, dilation_rate=1,
                                                   # dilation_rate=self.dilation_rate,
                                                   activation=None)

        self.initial_elu = tf.keras.layers.ELU(alpha=1.0)
        self.bn1 = tf.keras.layers.GroupNormalization()

        self.res_blocks = []
        self.downsampling = []
        for i, ratio in enumerate(self.ratios):  # 4个
            res_block_layers = []
            for j in range(self.n_residual_layers):
                res_block_layers.append(
                    SEANetResnetBlock(mult * self.n_filters, kernel_sizes=[self.residual_kernel_size, 1],
                                      # dilation_rate=[self.dilation_rate ** j, 1],
                                      causal=self.causal, padding='same', dilation_rate=1,
                                      compress=self.compress,
                                      true_skip=True))
            self.res_blocks.append(res_block_layers)

            self.downsampling.append(
                tf.keras.layers.Conv1D(mult * self.n_filters * 2, kernel_size=ratio * 2, strides=ratio,
                                       padding=self.padding, activation=None))
            self.downsampling.append(tf.keras.layers.ELU(alpha=1.0))
            self.bn2 = tf.keras.layers.GroupNormalization()

            mult *= 2

        self.final_conv = tf.keras.layers.Conv1D(mult * self.n_filters, self.last_kernel_size, padding=self.padding,
                                                 dilation_rate=1,
                                                 activation=None)
        self.final_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn3 = tf.keras.layers.GroupNormalization()

        self.bottleneck = tf.keras.layers.Dense(32, activation='elu')

    def get_config(self):

        config = super(SEANetEncoder, self).get_config().copy()

        config.update({

            'n_filters': self.n_filters,
            'n_residual_layers': self.n_residual_layers,
            'ratios': self.ratios,
            'kernel_size': self.kernel_size,
            'last_kernel_size': self.last_kernel_size,
            'residual_kernel_size': self.residual_kernel_size,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
            'true_skip': self.true_skip,
        })

        return config

    def call(self, inputs):

        x = self.initial_conv(inputs)
        x = self.initial_elu(x)
        x = self.bn1(x)

        for res_block_layers, downsampling_layer, elu_layer in zip(self.res_blocks, self.downsampling[::2],
                                                                   self.downsampling[1::2]):
            for layer in res_block_layers:
                x = layer(x)

            x = downsampling_layer(x)
            x = elu_layer(x)

        x = self.final_conv(x)
        x = self.final_elu(x)
        x = self.bn3(x)
        x = self.bottleneck(x)
        return x


class SEANetDecoder(tf.keras.layers.Layer):
    #
    def __init__(self, n_filters, n_residual_layers=1,
                 ratios=[4, 4, 2], kernel_size=3, last_kernel_size=3,
                 residual_kernel_size=3, causal=False, padding='same', dilation_rate=1,
                 true_skip=False, compress=1, name=None, **kwargs):
        super(SEANetDecoder, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.causal = causal
        self.padding = padding
        self.true_skip = true_skip
        self.compress = compress
        self.hop_length = np.prod(self.ratios)
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        mult = int(2 ** len(self.ratios))
        self.bn = tf.keras.layers.GroupNormalization()

        self.initial_conv = tf.keras.layers.Conv1D(self.n_filters, kernel_size=3, padding=self.padding, dilation_rate=1,
                                                   activation=None)
        self.initial_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn1 = tf.keras.layers.GroupNormalization()

        self.res_blocks = []
        self.upsampling = []

        for i, ratio in enumerate(self.ratios):
            res_block_layers = []
            for j in range(self.n_residual_layers):
                res_block_layers.append(
                    SEANetResnetBlock(mult * self.n_filters // 2, kernel_sizes=[self.residual_kernel_size, 1],
                                      causal=self.causal, padding='same', dilation_rate=1, compress=self.compress,
                                      true_skip=self.true_skip))
            self.res_blocks.append(res_block_layers)

            self.upsampling.append(
                tf.keras.layers.Conv1DTranspose(mult * self.n_filters // 2, kernel_size=ratio * 2, strides=ratio,
                                                padding=self.padding, dilation_rate=1, activation=None))
            self.upsampling.append(tf.keras.layers.ELU(alpha=1.0))

            mult //= 2

        self.final_conv = tf.keras.layers.Conv1D(self.n_filters, self.last_kernel_size,
                                                 padding=self.padding, dilation_rate=1,
                                                 activation=None)
        self.crop_layer = tf.keras.layers.Cropping1D(cropping=(0, 1))  # jian qu zui hou yi ge yuan su


        self.final_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn2 = tf.keras.layers.GroupNormalization()

    def get_config(self):
        # config = super().get_config()
        config = super(SEANetDecoder, self).get_config().copy()
        config.update({

            'n_filters': self.n_filters,
            'n_residual_layers': self.n_residual_layers,
            'ratios': self.ratios,
            'kernel_size': self.kernel_size,
            'last_kernel_size': self.last_kernel_size,
            'residual_kernel_size': self.residual_kernel_size,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
            'true_skip': self.true_skip,
        })
        return config

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.initial_conv(x)
        x = self.initial_elu(x)
        x = self.bn1(x)

        for res_block_layers, upsampling_layer, elu_layer in reversed(
                list(zip(self.res_blocks, self.upsampling[::2], self.upsampling[1::2]))):
            for layer in res_block_layers:
                x = layer(x)
            x = upsampling_layer(x)
            x = elu_layer(x)

        x = self.final_conv(x)
        x = self.crop_layer(x)

        x = self.final_elu(x)
        x = self.bn2(x)

        return x