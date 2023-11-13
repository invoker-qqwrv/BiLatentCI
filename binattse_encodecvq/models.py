# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv1D, LSTM, Lambda, LeakyReLU, Add, Concatenate, \
    Conv1DTranspose, LayerNormalization, Multiply
from sklearn.cluster import KMeans


class _ChannelNorm(tf.keras.layers.Layer):
    def __init__(self, encoded_len, channel_size, **kwargs):
        super(_ChannelNorm, self).__init__(**kwargs)
        self.encoded_len = encoded_len
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                             self.channel_size), dtype=self.dt),
                                 trainable=True, name="CN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                            self.channel_size), dtype=self.dt),
                                trainable=True, name="CN_beta")

    def get_config(self):
        config = super(_ChannelNorm, self).get_config().copy()
        config.update({
            'encoded_len': self.encoded_len,
            'channel_size': self.channel_size})
        return config

    def call(self, inputs):
        E = tf.reshape(
            tf.reduce_mean(inputs, axis=[2], name="E_CN_rm"), [-1, self.encoded_len, 1], name="E_CN_reshape")
        Var = tf.reshape(tf.reduce_mean((inputs - E) ** 2,
                                        axis=[2], name="Var_CN_rm"), [-1, self.encoded_len, 1], name="Var_CN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta


class _GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, channel_size, **kwargs):
        super(_GlobalNorm, self).__init__(**kwargs)
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                             self.channel_size), dtype=self.dt),
                                 trainable=True, name="GN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                            self.channel_size), dtype=self.dt),
                                trainable=True, name="GN_beta")

    def get_config(self):
        config = super(_GlobalNorm, self).get_config().copy()
        config.update({
            'channel_size': self.channel_size})
        return config

    def call(self, inputs):
        E = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2], name="E_GN_rm"), [-1, 1, 1], name="E_GN_reshape")
        Var = tf.reshape(
            tf.reduce_mean((inputs - E) ** 2, axis=[1, 2], name="Var_GN_rm"), [-1, 1, 1], name="Var_GN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta
def recordData(self, encoded_input):
    self.count += 1
    if self.scalar_quantization is True:
        if self.train_quantizer == True and self.count >= 3 and self.counter < self.buffer_length:
            tmp = encoded_input.numpy().squeeze()
            tmp = np.reshape(tmp, (1, tmp.shape[0] * tmp.shape[1]))
            L = tmp.shape[1]
            np.put(self.out_data, range(self.counter, min(self.counter + L, self.buffer_length)), tmp.squeeze())
            # self.out_data.extend(tmp.squeeze())
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
                self.vec_out_data[self.counter:self.counter + col, :] = tmp
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

class Encoder(tf.keras.layers.Layer):  # (none,none,64)
    def __init__(self, N, L, **kwargs):  # 初始化层的参数
        super(Encoder, self).__init__(**kwargs)
        self.N = N  # number of filter
        self.L = L  # kernel size of the convolutional layer

    def build(self, inputs):  # 用来创建encoder layer
        # print("encoder Build function called!")

        self.encoder = tf.keras.layers.Conv1D(  # 对输入数据进行一维卷积变换。因为处理的是一维数据
            filters=self.N,
            kernel_size=self.L,
            strides=self.L // 2,  # 小于filter/kerne，多次作用相同数据。导致大的output size
            activation="linear",  # 只要返回不变的输入值就好了。不用引入非线性
            name="encode_conv1d")

    def get_config(self):  # 将层的配置序列化为字典。以便层以json或者yaml文件保存到磁盘方便后来加载，然后将层的超参数加到字典中。
        config = super(Encoder, self).get_config().copy()  # 创建返回字典的副本
        config.update({  # 将层的超参数和实例添加到字典中
            'layers': self.encoder,
            'n': self.N,
            'l': self.L})
        return config  # 返回更新后的字典

    def call(self, inputs):
        # print('encoder inputs shape',inputs.shape)
        encoded_input = self.encoder(inputs=tf.expand_dims(inputs, -1))

        # print('encoded_input shape:',encoded_input.shape)
        # encoder_weights = self.encoder.weights
        # print('Encoder weights:', encoder_weights)
        return encoded_input

# 如果假设输入维度是（batch_size,sequeuce_length,input_dim）.输出tensor的维度就是（batch_size,encoded_length,N）
# sequeuce_length是输入序列的长度。input_dim是输入序列中每个特征的维度。它决定了第一层输入单元的数量
# 每一个filter会得到一个feature map，所以最后一个维度是n
class SEANetResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_sizes=[3, 1],
                 causal=False, padding='same', compress=2, dilation_rate = 1,true_skip=True, **kwargs):
        super(SEANetResnetBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.causal = causal
        self.padding = padding
        self.compress = compress
        self.true_skip = true_skip
        self.hidden = n_filters // compress
        self.dilation_rate = dilation_rate

        
        self.conv1 = tf.keras.layers.Conv1D(filters=self.hidden, kernel_size=kernel_sizes[0],  # 3
                                            activation=None, use_bias=False, padding='same', dilation_rate = 1)
        
        self.elu1 = tf.keras.layers.ELU(alpha=1.0)
        self.bn1 = tf.keras.layers.GroupNormalization()

        
        self.conv2 = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_sizes[1],  # 1
                                            activation=None, use_bias=False, padding='same',dilation_rate = 1)
        self.elu2 = tf.keras.layers.ELU(alpha=1.0)
        self.bn2 = tf.keras.layers.GroupNormalization()

        if true_skip:
            self.shortcut = layers.Lambda(lambda x: x)
            # if true_skip is True, the shortcut layer will be defined as an identity layer
            #  P145
        else:
            self.shortcut = layers.Conv1D(filters=n_filters, kernel_size=1, use_bias=False)
    # def get_config(self):  # 将层的配置序列化为字典。以便层以json或者yaml文件保存到磁盘方便后来加载，然后将层的超参数加到字典中。
    #     config = super(Encoder, self).get_config().copy()  # 创建返回字典的副本
    #     config.update({  # 将层的超参数和实例添加到字典中
    #         'layers': self.encoder,
    #         'n': self.N,
    #         'l': self.L})
    #     return config  # 返回更新后的字典
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


# 以上就是个标准的残差模块
# keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid',
#                                   dilation_rate=1, activation=None, use_bias=True,
#                                   kernel_initializer='glorot_uniform', bias_initializer='zeros',
#                                   kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#                                   kernel_constraint=None, bias_constraint=None)

class SEANetEncoder(tf.keras.layers.Layer):
    def __init__(self,  n_filters, n_residual_layers=1,
                 ratios=[4,4,2],dilation_rate = 1, kernel_size=3, last_kernel_size=3,
                 residual_kernel_size=3, causal=False, padding='same',
                 true_skip=False, compress=2, name=None,buffer_length=1000000, scalar_quantization=False, **kwargs):
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
        self.hop_length = np.prod(self.ratios)#compress 640
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
                                                   padding=self.padding,dilation_rate = 1,
                                                   # dilation_rate=self.dilation_rate,
                                                   activation=None)
        # (none,none,512)
        self.initial_elu = tf.keras.layers.ELU(alpha=1.0)
        self.bn1 = tf.keras.layers.GroupNormalization(groups = 16)

        self.res_blocks = []
        self.downsampling = []

        # 这里输入不进去
        for i, ratio in enumerate(self.ratios):  # 4个
            res_block_layers = []
            for j in range(self.n_residual_layers):
                res_block_layers.append(
                    SEANetResnetBlock(mult * self.n_filters, kernel_sizes=[self.residual_kernel_size, 1],
                                      # dilation_rate=[self.dilation_rate ** j, 1],
                                      causal=self.causal, padding='same',dilation_rate = 1,
                                      compress=self.compress,
                                      true_skip=True))
            self.res_blocks.append(res_block_layers)

            self.downsampling.append(
                tf.keras.layers.Conv1D(mult * self.n_filters * 2, kernel_size=ratio * 2, strides=ratio,
                                       padding=self.padding, activation=None))
            self.downsampling.append(tf.keras.layers.ELU(alpha=1.0))
            self.bn2 = tf.keras.layers.GroupNormalization(groups = 16)

            # 存储了每个ratio下的残差块。为单独的列表

            mult *= 2

        self.final_conv = tf.keras.layers.Conv1D(mult * self.n_filters, self.last_kernel_size, padding=self.padding,dilation_rate = 1,
                                                 activation=None)
        self.final_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn3 = tf.keras.layers.GroupNormalization(groups = 16)

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

        encoded_input= self.final_conv(encoded_input)
        encoded_input = self.final_elu(encoded_input)
        encoded_input = self.bn3(encoded_input)
        encoded_input = self.bottleneck(encoded_input)
        recordData(self, encoded_input)
        encoded_input = applyQuantization(self, encoded_input)
        return encoded_input


class SEANetDecoder(tf.keras.layers.Layer):
    #
    def __init__(self,n_filters, n_residual_layers=1,
                 ratios=[4,4,2],  kernel_size=3, last_kernel_size=3,
                 residual_kernel_size=3, causal=False, padding='same',dilation_rate = 1,
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

        self.initial_conv = tf.keras.layers.Conv1D(self.n_filters, kernel_size=3, padding=self.padding,dilation_rate = 1,
                                                            activation=None)
        self.initial_elu = tf.keras.layers.ELU(alpha=1.0)

        self.bn1 = tf.keras.layers.GroupNormalization()

        self.res_blocks = []
        self.upsampling = []

        for i, ratio in enumerate(self.ratios):
            res_block_layers = []
            for j in range(self.n_residual_layers):
                res_block_layers.append(
                    SEANetResnetBlock(mult * self.n_filters//2, kernel_sizes=[self.residual_kernel_size, 1],
                                      causal=self.causal, padding='same', dilation_rate = 1,compress=self.compress,
                                      true_skip=self.true_skip))
            self.res_blocks.append(res_block_layers)

            self.upsampling.append(
                tf.keras.layers.Conv1DTranspose(mult * self.n_filters // 2, kernel_size=ratio * 2, strides=ratio,
                                                padding=self.padding, dilation_rate = 1,activation=None))
            self.upsampling.append(tf.keras.layers.ELU(alpha=1.0))

            mult //= 2



        self.final_conv = tf.keras.layers.Conv1D(self.n_filters, self.last_kernel_size,
                                                          padding=self.padding,dilation_rate = 1,
                                                          activation=None)
        self.crop_layer = tf.keras.layers.Cropping1D(cropping=(0, 1))#jian qu zui hou yi ge yuan su
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
        # x = self.zero_pad(x)

        # print('decoded x shape',x.shape)
        x = self.final_elu(x)
        x = self.bn2(x)

        return x

class TCN(tf.keras.layers.Layer):
    def __init__(self, N, L, B, H, S, P, X, R, causal, skip, duration, sample_rate, GPU, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.S = S
        self.P = P
        self.X = X
        self.R = R
        self.GPU = GPU
        self.skip = skip
        self.causal = causal
        self.duration = duration
        self.sample_rate = sample_rate

        if self.causal:
            self.padding = "causal"
        else:
            self.padding = "same"

    def build(self, inputs):
        if self.causal:
            self.encoded_len = (int(self.duration * self.sample_rate) - self.L) // (self.L // 2) + 1
            inp_norm = _ChannelNorm(self.encoded_len, self.N, name="Input_Channel_Norm")
            frst_norm = _ChannelNorm(self.encoded_len, self.H, name="First_Channel_Norm")
            scnd_norm = _ChannelNorm(self.encoded_len, self.H, name="Second_Channel_Norm")
        else:
            inp_norm = _GlobalNorm(self.N, name="Input_Global_Norm")
            frst_norm = _GlobalNorm(self.H, name="First_Global_Norm")
            scnd_norm = _GlobalNorm(self.H, name="Second_Global_Norm")

        layers = {"input_norm": inp_norm,
                  "bottleneck": tf.keras.layers.Conv1D(self.B, 1, 1, name="bottleneck")}

        for r in range(self.R):
            for x in range(self.X):
                this_block = "block_{}_{}_".format(r, x)

                layers[this_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1, name="first_1x1_conv")

                layers[this_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="first_PReLU")

                layers[this_block + "first_norm"] = frst_norm

                if self.GPU:
                    layers[this_block + "dw_conv"] = tf.keras.layers.Conv1D(self.H, self.P,
                                                                            padding=self.padding,
                                                                            dilation_rate=2 ** x, groups=self.H)

                layers[this_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="second_PReLU")

                layers[this_block + "second_norm"] = scnd_norm

                layers[this_block + "out_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name="out_1x1_conv")

                layers[this_block + "res_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name="res_1x1_conv")

                layers[this_block + "skip_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.S, kernel_size=1, name="skip_1x1_conv")

        self.lrs = layers

    def get_config(self):
        config = super(TCN, self).get_config().copy()
        config.update({
            'layers': self.lrs,
            'n': self.N,
            'l': self.L,
            'b': self.B,
            'h': self.H,
            's': self.S,
            'p': self.P,
            'x': self.X,
            'r': self.R,
            'gpu': self.GPU,
            'c': self.causal,
            'skip': self.skip,
            'sr': self.sample_rate,
            'duration': self.duration,
            'enc_len': self.encoded_len})
        return config

    def call(self, encoded_input):
        self.encoded_len = encoded_input.shape[1]
        norm_input = self.lrs["input_norm"](encoded_input)
        block_input = self.lrs["bottleneck"](norm_input)
        skip_connections = 0

        for r in range(self.R):

            for x in range(self.X):

                now_block = "block_{}_{}_".format(r, x)

                block_output = self.lrs[now_block + "first_1x1_conv"](block_input)

                block_output = self.lrs[now_block + "first_PReLU"](block_output)

                block_output = self.lrs[now_block + "first_norm"](block_output)

                if self.GPU:
                    block_output = self.lrs[now_block + "dw_conv"](block_output)

                block_output = self.lrs[now_block + "second_PReLU"](block_output)

                block_output = self.lrs[now_block + "second_norm"](block_output)

                residual = self.lrs[now_block + "res_1x1_conv"](block_output)

                skip = self.lrs[now_block + "skip_1x1_conv"](block_output)

                block_input = residual + block_input

                skip_connections = skip_connections + skip

        if self.skip:
            return skip_connections
        else:
            return block_input


class Masker(tf.keras.layers.Layer):
    def __init__(self, N, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.N = N

    def build(self, inputs):
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1], name="decode_PReLU")
        self.decode = tf.keras.layers.Conv1D(self.N, 1, 1, name="1x1_conv_decoder")

    def get_config(self):
        config = super(Masker, self).get_config().copy()
        config.update({
            'n': self.N,
            'prelu': self.prelu,
            'decode': self.decode})
        return config

    def call(self, inputs, encoded_input):
        inputs = self.prelu(inputs)
        sep_output = self.decode(inputs)
        # print('sep_output shape:', sep_output.shape)  # (None,3999,64)
        mask = tf.keras.activations.softmax(sep_output)
        # print('mask shape:', mask.shape)  # (None,3999,64)
        return mask * encoded_input  # encoded_input(None,3999,64)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, L, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.L = L

    def build(self, inputs):
        self.deconv = tf.keras.layers.Conv1DTranspose(1, self.L, strides=self.L // 2,
                                                      use_bias=False, name="1d_deconv")

    def get_config(self):
        config = super(Decoder, self).get_config().copy()
        config.update({
            'l': self.L,
            'deconv': self.deconv})
        return config

    def call(self, inputs):
        output = tf.squeeze(self.deconv(inputs), -1)
        return output


class Model():
    def __init__(self, args):
        self.N = args.N
        self.L = args.L
        self.B = args.B
        self.H = args.H
        self.S = args.S
        self.P = args.P
        self.X = args.X
        self.R = args.R
        self.GPU = args.GPU
        self.skip = args.skip
        self.top = args.topology
        self.causal = args.causal
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        self.model_name = self.top + "_auto_encoder"

        self.encoder_left = Encoder(self.N, self.L, name="Encoder_left")

        self.encoder_right = Encoder(self.N, self.L, name="Encoder_right")

        self.TCN_left = TCN(self.N, self.L, self.B, self.H, self.S, self.P,
                            self.X, self.R,
                            self.causal, self.skip, self.duration,
                            self.sample_rate, self.GPU, name="TCN_left")

        self.TCN_right = TCN(self.N, self.L, self.B, self.H, self.S, self.P,
                             self.X, self.R,
                             self.causal, self.skip, self.duration,
                             self.sample_rate, self.GPU, name="TCN_right")

        self.attention1 = tf.keras.layers.Multiply(name="Attention_layer_1")

        self.attention2 = tf.keras.layers.Multiply(name="Attention_layer_2")

        self.masker_left = Masker(self.N, name="Masker_left")

        self.masker_right = Masker(self.N, name="Masker_right")

        self.decoder_left = Decoder(self.L, name="Decoder_left")

        self.decoder_right = Decoder(self.L, name="Decoder_right")

        self.seaencoder_left = SEANetEncoder(n_filters=64, name='seaencoder_left')  # 这里还是把参数加上吧，如果可以跑
        self.seaencoder_right = SEANetEncoder(n_filters=64, name='seaencoder_left')  # 这里还是把参数加上吧，如果可以跑

        self.seadecoder_left = SEANetDecoder(n_filters=64, name='seadecoder_left')
        self.seadecoder_right = SEANetDecoder(n_filters=64, name='seadecoder_left')

        self.seaencoder_left_tcn = SEANetEncoder(n_filters=512, name='seaencoder_left_tcn')  # 这里还是把参数加上吧，如果可以跑
        self.seaencoder_right_tcn = SEANetEncoder(n_filters=512, name='seaencoder_left_tcn')  # 这里还是把参数加上吧，如果可以跑

        self.seadecoder_left_tcn = SEANetDecoder(n_filters=512, name='seadecoder_left_tcn')
        self.seadecoder_right_tcn = SEANetDecoder(n_filters=512, name='seadecoder_left_tcn')

    def call(self):

        input_left = tf.keras.Input(shape=(None,), name="Input_left")

        input_right = tf.keras.Input(shape=(None,), name="Input_right")

        enc_inp_l = self.encoder_left(input_left)

        enc_inp_r = self.encoder_right(input_right)

        # kernel1 = self.attention2([enc_inp_l, enc_inp_r])  # (2,3999,64)
        seaencoder_l = self.seaencoder_left(enc_inp_l)
        seaencoder_r = self.seaencoder_right(enc_inp_r)

        seadecoder_l = self.seadecoder_left(seaencoder_l)
        seadecoder_r = self.seadecoder_right(seaencoder_r)

        # seaautoencoder = self.seaautoencoder(enc_inp_l)
        # print('dynamic shape of enc_inp_r tf', tf.shape(seaautoencoder))
        # print('static shape of enc_inp_r tf', tf.Tensor.get_shape(seaautoencoder))
        kernel_l = self.attention2([seadecoder_l, enc_inp_r])  # self.attention2([enc_inp_l, enc_inp_r])
        # kernel1_right = self.attention2([enc_inp_l, enc_inp_r])

        #上面是left，下面是right

        kernel_r = self.attention2([enc_inp_l, seadecoder_r])
        skp_l = self.TCN_left(kernel_r)  # (2,3999,512)

        skp_r = self.TCN_right(kernel_l)  # (2,3999,512)

        kernel_r_out = self.seaencoder_left_tcn(skp_r)
        kernel_r_out = self.seadecoder_left_tcn(kernel_r_out)

        #只加了这两行代码，程序就不能跑了
        #这个kernel_r_out和kernel_l_out是完全一样的吗
        #这里用同一个encoder和decoder处理数据就不会出现这个问题。而不是加以区分左右。
        kernel_l_out = self.seaencoder_right_tcn(skp_l)
        kernel_l_out = self.seadecoder_right_tcn(kernel_l_out)

        kernel_r = self.attention1([skp_l, kernel_r_out])
        kernel_l = self.attention1([kernel_l_out, skp_r])  # (2,3999,512)
        # print('kernel2 shape:', kernel2.shape)

        masked_left = self.masker_left(kernel_r, enc_inp_l)  # enc_inp_l(None,3999,64)
        # print('masked shape:', masked_left.shape)  # (2,3999,64)

        masked_right = self.masker_right(kernel_l, enc_inp_r)  # 经过mask操作后，这里的输出应该和encoded_input一样

        out_left = self.decoder_left(masked_left)

        out_right = self.decoder_right(masked_right)

        model = tf.keras.Model(inputs=[input_left, input_right],
                               outputs=[out_left, out_right], name=self.model_name)

        for i, w in enumerate(model.weights):
            split_name = w.name.split('/')
            if len(split_name) > 1:
                new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i) + '/' + split_name[-1]
                model.weights[i]._handle_name = new_name
            else:
                new_name = "new" + '_' + str(i) + '/' + "layer_name" + '_' + str(i) + '/' + split_name[-1]

        return model


class SISDR(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SISDR, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        def calc_snr(s_hat, s):
            def norm(x):
                return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)

            s_target = tf.reduce_sum(s_hat * s, axis=-1,
                                     keepdims=True) * s / norm(s)
            num = norm(s_target)
            den = norm(s_hat - s_target)
            return 10. * tf.experimental.numpy.log10(num / den)

        sdr = calc_snr(y_pred, y_true)
        return tf.reduce_mean(-sdr)
