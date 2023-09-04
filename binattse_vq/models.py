# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import savemat
import random
import time


class  _ChannelNorm(tf.keras.layers.Layer):
    def __init__(self, encoded_len, channel_size, **kwargs):
        super(_ChannelNorm, self).__init__(**kwargs)
        self.encoded_len = encoded_len
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape = (1, 1,
                                                               self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "CN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape = (1, 1, 
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "CN_beta") 
    def get_config(self):
        config = super(_ChannelNorm, self).get_config().copy()
        config.update({
            'encoded_len': self.encoded_len,
            'channel_size': self.channel_size})
        return config 

    def call(self, inputs): 
        E = tf.reshape(
                tf.reduce_mean(inputs, axis=[2], name="E_CN_rm"), [-1, self.encoded_len, 1], name="E_CN_reshape") 
        Var = tf.reshape(tf.reduce_mean((inputs - E)**2,
                                        axis=[2], name="Var_CN_rm"),[-1, self.encoded_len, 1], name="Var_CN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta
    
class  _GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, channel_size, **kwargs):
        super(_GlobalNorm, self).__init__(**kwargs)
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape = (1, 1,
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "GN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape = (1, 1, 
                                                              self.channel_size), dtype=self.dt), 
                                                                trainable=True, name = "GN_beta")
    
    def get_config(self):
        config = super(_GlobalNorm, self).get_config().copy()
        config.update({
            'channel_size': self.channel_size})
        return config 
    
    def call(self, inputs):
        E = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2], name="E_GN_rm"), [-1, 1, 1], name="E_GN_reshape")
        Var = tf.reshape(
            tf.reduce_mean((inputs - E)**2, axis=[1, 2], name="Var_GN_rm"), [-1, 1, 1], name="Var_GN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta


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
            tmp = np.reshape(x, (rows*cols,1))
            tmp_quantized = self.quantizer_centers[self.quantizer.predict(tmp)]
            tmp_quantized = np.expand_dims(np.reshape(tmp_quantized, (rows, cols)), axis=0)
            return tf.convert_to_tensor(tmp_quantized),tensor_in 
        elif self.scalar_quantization is False:
            tmp = tensor_in.numpy().squeeze(axis=0)                
            rows = tmp.shape[0]
            cols = tmp.shape[1]
            tmp_quantized =  self.quantizer_centers[self.quantizer.predict(tmp)]
            tmp_quantized = np.expand_dims(np.reshape(tmp_quantized, (rows, cols)), axis=0)
            return tf.convert_to_tensor(tmp_quantized), tensor_in 
    else:
        return tensor_in,tensor_in


def trainQuantizer(self, n_clusters = 32):
    self.train_quantizer = False
    if self.scalar_quantization is True:
        tmp = np.reshape(self.out_data,(self.out_data.shape[0]*self.out_data.shape[1],1))
        self.quantizer = KMeans(n_clusters=n_clusters, random_state=0).fit(tmp)
        self.quantizer_centers = self.quantizer.cluster_centers_
    elif self.scalar_quantization is False: # vector quantization
        self.quantizer = KMeans(n_clusters=n_clusters, random_state=0).fit(self.vec_out_data[0:self.counter,:])
        self.quantizer_centers = self.quantizer.cluster_centers_

    
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, L, buffer_length, scalar_quantization, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.N = N
        self.L = L
        self.count = 0
        self.counter = 0
        self.buffer_length = buffer_length
        self.out_data = np.empty((1,buffer_length),dtype=np.float32)#[]
        self.vec_out_data = np.empty((buffer_length,N),dtype=np.float32)#[]
        self.train_quantizer = True
        self.quantizer = None
        self.scalar_quantization = scalar_quantization

            
    def build(self, inputs):
        self.encoder = tf.keras.layers.Conv1D(
                filters=self.N,
                kernel_size=self.L,
                strides=self.L//2,
                activation="linear",
                name="encode_conv1d")
        
    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'layers': self.encoder,
            'n': self.N,
            'l': self.L})
        return config 
    
    def call(self, inputs):
        encoded_input = self.encoder(inputs=tf.expand_dims(inputs, -1))
        #print(encoded_input)
        
        #encoded_input = applyQuantizer(self, encoded_input, scalar=True)

        recordData(self,encoded_input)
        encoded_input_quantized, encoded_input = applyQuantization(self, encoded_input)

        
        return encoded_input_quantized, encoded_input
    

        
        
    

                

class TCN(tf.keras.layers.Layer):
    def __init__(self, N, L, B, H, S, P, X, R, causal, skip, duration, sample_rate, GPU, buffer_length, scalar_quantization, **kwargs):
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
        self.count = 0
        self.count = 0
        self.counter = 0
        self.buffer_length = buffer_length
        self.out_data = np.empty((1,buffer_length),dtype=np.float32)#[]
        self.vec_out_data = np.empty((buffer_length,S),dtype=np.float32)#[]
        self.train_quantizer = True
        self.quantizer = None
        self.scalar_quantization = scalar_quantization
                 
        if self.causal:
            self.padding = "causal"
        else:
            self.padding = "same"
        
    def build(self, inputs):
        if self.causal:
            self.encoded_len = (int(self.duration * self.sample_rate) - self.L) // (self.L // 2) + 1
            inp_norm  = _ChannelNorm(self.encoded_len, self.N, name = "Input_Channel_Norm")
            frst_norm = _ChannelNorm(self.encoded_len, self.H, name = "First_Channel_Norm")
            scnd_norm = _ChannelNorm(self.encoded_len, self.H, name = "Second_Channel_Norm")
        else:
            inp_norm  = _GlobalNorm(self.N, name = "Input_Global_Norm")
            frst_norm = _GlobalNorm(self.H, name = "First_Global_Norm")
            scnd_norm = _GlobalNorm(self.H, name = "Second_Global_Norm")
        
        layers = {"input_norm": inp_norm,
            "bottleneck": tf.keras.layers.Conv1D(self.B, 1, 1, name = "bottleneck")}
        
        for r in range(self.R):
            for x in range(self.X):
                this_block = "block_{}_{}_".format(r, x)
                
                layers[this_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1, name = "first_1x1_conv")
                
                layers[this_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name = "first_PReLU")
                
                layers[this_block + "first_norm"] = frst_norm
                
                if self.GPU:
                    layers[this_block + "dw_conv"] = tf.keras.layers.Conv1D(self.H, self.P, 
                                                    padding = self.padding, 
                                                    dilation_rate = 2**x, groups=self.H)
                
                layers[this_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name = "second_PReLU")
                
                layers[this_block + "second_norm"] = scnd_norm
                
                layers[this_block + "out_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name = "out_1x1_conv") 
                
                layers[this_block + "skip_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.S, kernel_size=1, name = "skip_1x1_conv")
       
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
                #print(self.lrs.keys())
                residual = self.lrs[now_block + "out_1x1_conv"](block_output)
                
                skip = self.lrs[now_block + "skip_1x1_conv"](block_output)
                
                block_input = residual + block_input

                skip_connections = skip_connections + skip
                
        
        recordData(self,skip_connections)      
        
            
        #     skip_connections = self.applyQuantization(skip_connections)
        
        if self.skip:
            return applyQuantization(self, skip_connections)
        else:
            return block_input
        
    # def trainQuantizer(self, n_clusters = 512):
    #     tmp = np.array(self.out_data)
    #     tmp = np.reshape(tmp,(1,tmp.shape[0]*tmp.shape[1]))
    #     self.quantizer = KMeans(n_clusters=n_clusters, random_state=0).fit(np.transpose(tmp))
    #     self.quantizer_centers = self.quantizer.cluster_centers_
    #     self.train_quantizer = False
        
    # def applyQuantization(self,x):
    #     tmp = np.reshape(x.numpy().squeeze(axis=0), (3999*512,1))
    #     tmp_quantized = self.quantizer_centers[self.quantizer.predict(tmp)]
    #     tmp_quantized = np.expand_dims(np.reshape(tmp_quantized, (3999, 512)), axis=0)
    #     return tf.convert_to_tensor(tmp_quantized) 
    

class Masker(tf.keras.layers.Layer):
    def __init__(self, N, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.N = N
        
    def build(self, inputs):   
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1], name = "decode_PReLU")
        self.decode = tf.keras.layers.Conv1D(self.N, 1, 1, name = "1x1_conv_decoder")

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
        mask = tf.keras.activations.softmax(sep_output)
        return mask * encoded_input

class Decoder(tf.keras.layers.Layer):
    def __init__(self,  L, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.L = L

    def build(self, inputs):  
        self.deconv = tf.keras.layers.Conv1DTranspose(1, self.L, strides = self.L//2,
                                                      use_bias=False, name = "1d_deconv")
        
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
        self.GPU = True#args.GPU
        self.skip = args.skip
        self.top = args.topology
        self.causal = args.causal
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        self.model_name = self.top + "_auto_encoder"
        self.buffer_length = 1000000 #args.buffer_length
                 
        self.encoder_left = Encoder(self.N, self.L, buffer_length=self.buffer_length, scalar_quantization=False, name = "Encoder_left")

        self.encoder_right = Encoder(self.N, self.L, buffer_length=self.buffer_length,scalar_quantization=False,name = "Encoder_right")

        self.TCN_left = TCN(self.N, self.L, self.B, self.H, self.S, self.P, 
                                                           self.X, self.R,  
                                                           self.causal, self.skip, self.duration, 
                                                           self.sample_rate, self.GPU, buffer_length=self.buffer_length, scalar_quantization=False, name = "TCN_left")

        self.TCN_right = TCN(self.N, self.L, self.B, self.H, self.S, self.P, 
                                                           self.X, self.R,  
                                                           self.causal, self.skip,  self.duration, 
                                                           self.sample_rate, self.GPU, buffer_length=self.buffer_length, scalar_quantization=False,name = "TCN_right")
             
        self.attention1 = tf.keras.layers.Multiply(name = "Attention_layer_1")
        
        self.attention2 = tf.keras.layers.Multiply(name = "Attention_layer_2")
       
        self.masker_left = Masker(self.N, name = "Masker_left")
        
        self.masker_right = Masker(self.N, name = "Masker_right")
          
        self.decoder_left = Decoder(self.L, name = "Decoder_left")
        
        self.decoder_right = Decoder(self.L, name = "Decoder_right")
        
       # self.linearLayer = Linear()
        

    def call(self):

            input_left  = tf.keras.Input(shape = (None,), name = "Input_left")
            
            input_right = tf.keras.Input(shape = (None,), name = "Input_right")
            
            enc_inp_l_quantized, enc_inp_l = self.encoder_left(input_left)
                  
            enc_inp_r_quantized, enc_inp_r = self.encoder_right(input_right)

            
            kernel1_left = self.attention2([enc_inp_l, enc_inp_r_quantized]) #self.attention2([enc_inp_l, enc_inp_r])
            kernel1_right = self.attention2([enc_inp_l_quantized, enc_inp_r])
            
            skp_l_quantized, skp_l = self.TCN_left(kernel1_left)
            
            skp_r_quantized, skp_r = self.TCN_right(kernel1_right) 
            
            kernel2_left =  self.attention1([skp_l, skp_r_quantized])
            kernel2_right =  self.attention1([skp_l_quantized, skp_r])

            masked_left = self.masker_left(kernel2_left, enc_inp_l)

            masked_right = self.masker_right(kernel2_right, enc_inp_r)  

            out_left = self.decoder_left(masked_left)

            out_right = self.decoder_right(masked_right)
        
            model = tf.keras.Model(inputs = [input_left, input_right], 
                                   outputs = [out_left , out_right], name = self.model_name)

            
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
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        def calc_snr(s_hat, s):
            def norm(x):
                return tf.reduce_sum(x**2, axis=-1, keepdims=True)
            s_target = tf.reduce_sum(s_hat * s, axis=-1, 
                                      keepdims=True) * s / norm(s)
            num = norm(s_target)
            den = norm(s_hat - s_target)
            return 10. * tf.experimental.numpy.log10(num / den)
        
        sdr = calc_snr(y_pred, y_true)
        return tf.reduce_mean(-sdr)
