import tensorflow as tf
import numpy as np
import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv1D, LSTM, Lambda, LeakyReLU, Add, Concatenate, \
    Conv1DTranspose, LayerNormalization, Multiply
from sklearn.cluster import KMeans

def recordData(self, encoded_input):
    self.count += 1
    if self.scalar_quantization is True:
        if self.train_quantizer == True and self.count >= 3 and self.counter < self.buffer_length:
            tmp = encoded_input.numpy().squeeze()
            tmp = np.reshape(tmp,(1, tmp.shape[0]*tmp.shape[1]))
            L = tmp.shape[1]
            np.put(self.out_data,range(self.counter,min(self.counter+L,self.buffer_length)), tmp.squeeze())
            #self.out_data.extend(tmp.squeeze())
            self.counter += L
    elif self.scalar_quantization is False:
        if self.train_quantizer == True and self.count >= 3 and self.counter < self.buffer_length:

            tmp = encoded_input.numpy().squeeze()
            col = tmp.shape[0]
# =============================================================================
#             if random.random() < 0.1:
#                 savemat(self.name + '_' + str(time.time()) + '.mat', {'data':tmp})
#             #tmp_vec_out = self.vec_out_data
# =============================================================================
            if self.counter + col <= self.buffer_length:
                self.vec_out_data[self.counter:self.counter+col,:] = tmp
            self.counter += col

# we return the quantized version and the original
def applyQuantization(self, tensor_in):
    if self.train_quantizer == False:
        if self.scalar_quantization is True:
            x = tensor_in.numpy().squeeze(axis=0)
            rows = x.shape[0]
            cols = x.shape[1]
            tmp = np.reshape(x, (rows * cols, 1))
            tmp_quantized = self.quantizer_centers[self.quantizer.predict(tmp)]
            tmp_quantized = np.expand_dims(np.reshape(tmp_quantized, (rows, cols)), axis=0)
            return tf.convert_to_tensor(tmp_quantized), tensor_in
        elif self.scalar_quantization is False:
            tmp = tensor_in.numpy().squeeze(axis=0)
            rows = tmp.shape[0]
            cols = tmp.shape[1]
            tmp_quantized = self.quantizer_centers[self.quantizer.predict(tmp)]
            tmp_quantized = np.expand_dims(np.reshape(tmp_quantized, (rows, cols)), axis=0)
            return tf.convert_to_tensor(tmp_quantized)
    else:
        return tensor_in


def trainQuantizer(self, n_clusters=32):
    self.train_quantizer = False
    if self.scalar_quantization is True:
        tmp = np.reshape(self.out_data, (self.out_data.shape[0] * self.out_data.shape[1], 1))
        self.quantizer = KMeans(n_clusters=n_clusters, random_state=0).fit(tmp)
        self.quantizer_centers = self.quantizer.cluster_centers_
    elif self.scalar_quantization is False:  # vector quantization
        self.quantizer = KMeans(n_clusters=n_clusters, random_state=0).fit(self.vec_out_data[0:self.counter, :])
        self.quantizer_centers = self.quantizer.cluster_centers_

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
            # if true_skip is True, the shortcut layer will be defined as an identity layer
            #  P145
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
        # out = tf.keras.layers.Add()([out, shortcut])
        # print('encoder out shape:', out.shape)
        return out



class SEANetEncoder(tf.keras.layers.Layer):
    def __init__(self, n_filters, n_residual_layers=1,
                 ratios=[4, 4, 2], dilation_rate=1, kernel_size=3, last_kernel_size=3,
                 residual_kernel_size=3, causal=False, padding='same',
                 true_skip=False, compress=2, name=None, buffer_length=1000000, scalar_quantization=False, **kwargs):
        # dimention是时间序列的长度，用于设置输入tensor的形状
        # dimention影响输入，filter影像输出
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
        self.hop_length = np.prod(self.ratios)  # compress 640
        self.dilation_rate = dilation_rate
        self.count = 0
        self.counter = 0
        self.buffer_length = buffer_length
        self.out_data = np.empty((1, buffer_length), dtype=np.float32)  # []
        self.vec_out_data = np.empty((buffer_length, 32), dtype=np.float32)  # []
        self.train_quantizer = True
        self.quantizer = None
        self.scalar_quantization = scalar_quantization

    def build(self, input_shape):
        mult = 1
        # (none,none,64)

        self.initial_conv = tf.keras.layers.Conv1D(filters=mult * self.n_filters,
                                                   kernel_size=self.kernel_size,
                                                   padding=self.padding, dilation_rate=1,
                                                   # dilation_rate=self.dilation_rate,
                                                   activation=None)
        # (none,none,512)
        self.initial_elu = tf.keras.layers.ELU(alpha=1.0)
        self.bn1 = tf.keras.layers.GroupNormalization(groups=16)

        self.res_blocks = []
        self.downsampling = []

        # 这里输入不进去
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
            self.bn2 = tf.keras.layers.GroupNormalization(groups=16)

            # 存储了每个ratio下的残差块。为单独的列表

            mult *= 2

        self.final_conv = tf.keras.layers.Conv1D(mult * self.n_filters, self.last_kernel_size, padding=self.padding,
                                                 dilation_rate=1,
                                                 activation=None)
        self.final_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn3 = tf.keras.layers.GroupNormalization(groups=16)

        self.bottleneck = tf.keras.layers.Dense(32, activation='elu')

    def get_config(self):
        # config = super().get_config()
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
        # print('encoded_inp shape', inputs.shape)
        # x = self.bn(inputs)
        encoded_input = self.initial_conv(inputs)
        encoded_input = self.initial_elu(encoded_input)
        encoded_input = self.bn1(encoded_input)

        for res_block_layers, downsampling_layer, elu_layer in zip(self.res_blocks, self.downsampling[::2],
                                                                   self.downsampling[1::2]):
            for layer in res_block_layers:  # 遍历所有的residual layer
                encoded_input = layer(encoded_input)

            encoded_input = downsampling_layer(encoded_input)
            encoded_input = elu_layer(encoded_input)

        encoded_input = self.final_conv(encoded_input)
        encoded_input = self.final_elu(encoded_input)
        encoded_input = self.bn3(encoded_input)
        encoded_input = self.bottleneck(encoded_input)
        recordData(self, encoded_input)
        encoded_input = applyQuantization(self, encoded_input)
        return encoded_input


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
        # self.zero_pad = tf.keras.layers.ZeroPadding1D(padding=(0, 1))

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