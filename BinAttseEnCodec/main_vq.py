# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""
 
import os
import sys
import time
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import json
import models
import warnings
import xlsxwriter
import numpy as np
from models import Model
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
from utils import setup, write_to_audio
from data_generator import DataGenerator
from keras.utils.layer_utils import count_params
from keras import backend as K
from models import trainQuantizer
import librosa


def get_all_outputs(model, input_data, learning_phase=1):
    print("test")
    outputs = [layer.output for layer in model.layers[1:]]
    layers_fn = K.function([model.input, K.learning_phase()],outputs)
    return layers_fn([input_data, learning_phase])
    
def train(args):
        
    loss = models.SISDR()

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    model_dir = os.path.join(args.model_dir, (args.topology + '_' + 'model[{}]').format(time_stamp))
    
    train_ds = DataGenerator("train", args).fetch()
    
    valid_ds = DataGenerator("valid", args).fetch()
   
    opt = tf.optimizers.Adam()
    
    os.makedirs(model_dir)
        
    json.dump(vars(args), open(os.path.join(model_dir,'params.json'), 'w'), indent=4)
    
    log_path = os.path.join(model_dir, 'train_log[{}].csv'.format(time_stamp))
    
    checkpoint_path = os.path.join(model_dir, 'model_best[{}].h5'.format(time_stamp))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = .8, patience=3, min_lr = 0.),
        tf.keras.callbacks.CSVLogger(log_path),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                           monitor='val_loss', verbose=0, save_best_only=True, 
                                           mode='auto', save_freq='epoch')]
    
    physical_devices = tf.config.list_physical_devices()
    
    if len(physical_devices) == 1:
        print("\nUsing CPU... \n")
        warnings.warn("Depwthwise comvolution not available when using CPU for processing.")
        args.GPU = False
    
    else:
        print("\nUsing GPU... \n")
   
    model = Model(args).call()
    
    model.summary()
    
    #tf.keras.utils.plot_model(model, to_file = os.path.join(model_dir,'model.pdf'), show_layer_names = False)
    
    model.compile(optimizer=opt, loss = loss, )
    
    history = model.fit(train_ds, validation_data = valid_ds, 
                    epochs = args.max_epoch, callbacks = callbacks)

    plt.style.use('ggplot')
    
    history_dict = history.history
    
    loss     = [ -_ for _ in history_dict['loss']]
    val_loss = [ -_ for _ in history_dict['val_loss']]
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(14,5))
    plt.plot(epochs, loss, marker='.', label='Training')
    plt.plot(epochs, val_loss, marker='.', label='Validation')
    plt.title('Training and validation performance')
    plt.xlabel('Epoch')
    plt.grid(axis='x')
    plt.ylabel('SI-SDR [dB]')
    plt.legend(loc='lower right')
    
    plt.savefig(os.path.join(model_dir,'learning_curves.pdf'))

def evaluate(args, model):         
    print(args)
    loss = models.SISDR()
    
    model_time_stamp = model[-20:-1]
   
    model_dir = os.path.join(args.model_dir, model)
    
    log_path = os.path.join(model_dir, 'test_log[{}].csv'.format(model_time_stamp))
       
    params_file = os.path.join(model_dir, "params.json")
    
    with open(params_file) as f:
        args = json.load(f)
                
    args = namedtuple("args", args.keys())(*args.values())
    
    train_ds = DataGenerator("train", args).fetch()
    test_ds  = DataGenerator("valid", args).fetch()
    
    physical_devices = tf.config.list_physical_devices("GPU")
    
    if not physical_devices:
        args.GPU = False
        


    model = Model(args).call()
    
    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)),by_name = True)
    
    model.compile(loss = loss, run_eagerly=True)
    # mix, _ = librosa.load("/home/hinrichs/CI-Projekt/RebiHear/nobackup/BinAttSE-main/data/valid/S06001_mixed_CH1.wav", 8000, mono = False)
    # mix_tensor = tf.convert_to_tensor(mix[0,:])
    # out = model.predict(mix_tensor)

    
    test_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)
    test_logger.on_test_begin = test_logger.on_train_begin
    test_logger.on_test_batch_end = test_logger.on_epoch_end
    test_logger.on_test_end = test_logger.on_train_end
    
    print("\nEvaluating model... \n", flush=True)
    
    #train_ds = train_ds.take(35)
    #test_ds = test_ds#.take(50)
    # for raw_record in train_ds:
    #   example = tf.train.Example()
    #   #example.ParseFromString(raw_record[0][0].numpy())
    #   count += raw_record[0][0].numpy().shape[1]
      
    
    print("Running on Train", flush=True)
    _, snr_left, snr_right = model.evaluate(train_ds, callbacks = test_logger, verbose = 0)    
    train_snr = np.mean([-snr_left , -snr_right]) 
    
    #start = time.time()
    #print("Running on Validation", flush=True)
    #_, snr_left, snr_right = model.evaluate(test_ds, callbacks = test_logger, verbose = 0)
    # snr_left = -snr_left
    # snr_right = -snr_right
    #test_snr = np.mean([-snr_left , -snr_right])  
    
    #print("Mean SI-SDR (prequantization): {:.2f} dB".format(test_snr), flush=True)
    #end = time.time()
    #print("Elapsed time (Evaluation) " + str(end-start), flush=True)
    
    bits = [3,4,5,6,10,8]
    for b in bits:
            start = time.time()
            trainQuantizer(model.layers[4], n_clusters=2**b) 
            trainQuantizer(model.layers[5], n_clusters=2**b) 
            # trainQuantizer(model.layers[5], n_clusters=2**b) 
            # trainQuantizer(model.layers[6], n_clusters=2**b) 
            
        
            _, snr_left, snr_right = model.evaluate(test_ds, callbacks = test_logger, verbose = 0)
            print([-snr_left , -snr_right])
            test_snr = np.mean([-snr_left , -snr_right]) 
    #tmp = get_all_outputs(model, test_ds)
            
            print(F"Mean SI-SDR (quantization,b={b}): {test_snr:.2f} dB", flush=True)
            testQuantizedModel(args, model, b)
            end = time.time()
            
            print(F"Elapsed time (b={b}): " + str(end-start), flush=True)
            


    #tmp = get_all_outputs(model, test_ds)
    
    #get_layer_output = K.function([model.layers[0].input], [model.layers[3].output])
    #layer_output = get_layer_output([test_ds])[0]                 
   
    
      
    with open(os.path.join(model_dir, "Evaluation results[" + model_time_stamp + "].txt"), "w") as f:
        f.write("Topology: {:s}\n".format(args.topology))
        f.write("Mean SI-SDR: {:.2f} dB\n".format(test_snr))
        f.write("Left SI-SDR: {:.2f} dB\n".format(snr_left))
        f.write("Right SI-SDR: {:.2f} dB\n".format(snr_right))
        f.write("Number of parameters: {:d}\n".format(count_params(model.trainable_variables)))
        f.write("S: {:d}".format(args.S))
    f.close()

    workbook = xlsxwriter.Workbook(os.path.join(model_dir, "Evaluation results[" + model_time_stamp + "].xlsx"))
    worksheet = workbook.add_worksheet()
    data= (
        ['Topology', args.topology],
        ['MeanSDR',  test_snr],
        ['LeftSDR',  snr_left],
        ['RightSDR', snr_right],
        ['Params',   count_params(model.trainable_variables)],
        ['S', args.S])
    
    row = 0
    col = 0
    
    for var, value in (data):
        worksheet.write(row, col, var)
        worksheet.write(row, col + 1, value)
        row += 1
    
    workbook.close()
    
def testQuantizedModel(args, model, bits):         
    trained_models = os.listdir(args.model_dir)
    model_dir = os.path.join(args.model_dir, trained_models[1])      
    save_path = os.path.join(model_dir, f"audio_bit={bits}")
    params_file = os.path.join(model_dir, "params.json")
    
    with open(params_file) as f:
        args = json.load(f)
                
    args = namedtuple("args", args.keys())(*args.values())
    
    test_ds = DataGenerator("test", args).fetch()
    
    physical_devices = tf.config.list_physical_devices("GPU")
    
    if not physical_devices:    # workbook.close()

        args.GPU = False

        
    if not os.path.isdir(save_path):
        os.mkdir(save_path) 
        write_to_audio(model, test_ds, args, save_path)
    else:
        write_to_audio(model, test_ds, args, save_path)




def test(args, model):         
    
    loss = models.SISDR()
    
    model_time_stamp = model[-20:-1]
   
    model_dir = os.path.join(args.model_dir, model)
      
    save_path = os.path.join(model_dir, "audio")
    
    params_file = os.path.join(model_dir, "params.json")
    
    with open(params_file) as f:
        args = json.load(f)
                
    args = namedtuple("args", args.keys())(*args.values())
    
    test_ds = DataGenerator("test", args).fetch()
    
    physical_devices = tf.config.list_physical_devices("GPU")
    
    if not physical_devices:
        args.GPU = False

    model = Model(args).call()
    
    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))
    
    model.compile(loss = loss)
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path) 
        write_to_audio(model, test_ds, args, save_path)
    else:
        write_to_audio(model, test_ds, args, save_path)

def main():
    print("Starting run")
    args = setup()
    print(args)
    if args.mode == "train":
            train(args)
    elif args.mode == "evaluate":
        trained_models = os.listdir(args.model_dir)
        for tm in  trained_models[1:]:
            evaluate(args, tm)
    else:
        trained_models = os.listdir(args.model_dir)
        del trained_models[0]
        for tm in  trained_models:
            test(args, tm)
 
if __name__ == '__main__':
    main()  
