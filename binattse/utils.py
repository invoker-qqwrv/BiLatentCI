# -*- coding: utf-8 -*-
"""
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""


import argparse
from tqdm import tqdm
from scipy.signal import resample
from clarity.evaluator.mbstoi.mbstoi import mbstoi
from scipy.io import wavfile
import sys
import scipy.io
import os
import json
import models
import xlsxwriter
import numpy as np
from models import Model
from data_generator import DataGenerator
from keras.utils.layer_utils import count_params
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def setup():
    parser = argparse.ArgumentParser(description = 'Main configuration')

    parser.add_argument('-top', '--topology', type=str, default = "Double_attention")

    parser.add_argument('-mo', '--mode', type=str, default='train')

    parser.add_argument('-gpu',  '--GPU', type=bool, default = False)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-ld', '--model_dir', type=str, default='./models')
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-sr', '--sample_rate', type=int, default=8000)
    parser.add_argument('-c',  '--causal', type=bool, default = True)
    parser.add_argument('-me', '--max_epoch', type=int, default=3)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-k',  '--skip', type=bool, default = True)
    parser.add_argument('-d',  '--duration', type=int, default=4)
    parser.add_argument('-compress', '--compress', type=int, default=1)

    #autoencoder
    parser.add_argument('-n_filters', '--n_filters', type=int, default=64)
    parser.add_argument('-n_residual_layers', '--n_residual_layers', type=int, default=1)
    parser.add_argument('-ratios', '--ratios', type=list, default=[4,4,2])
    parser.add_argument('-dilation_rate', '--dilation_rate', type=int, default=2)
    parser.add_argument('-kernel_size', '--kernel_size', type=int, default=3)
    parser.add_argument('-last_kernel_size', '--last_kernel_size', type=int, default=3)
    parser.add_argument('-residual_kernel_size', '--residual_kernel_size', type=int, default=3)
    parser.add_argument('-true_skip', '--true_skip', type=bool, default=False)
    parser.add_argument('-padding', '--padding', type=str, default='same')



    parser.add_argument('-N', '-N', type=int, default=64)  # Number of filters in autoencoder
    parser.add_argument('-L', '-L', type=int, default=16)  # Length of the filters (in samples)
    parser.add_argument('-B', '-B', type=int, default=64)  # Number of channels in bottleneck 1 × 1-conv block
    parser.add_argument('-H', '-H', type=int, default=64)  # Number of channels in convolutional blocks
    parser.add_argument('-S', '-S', type=int, default=512)  # Kernel size in convolutional blocks
    parser.add_argument('-P', '-P', type=int, default=128)  # Number of convolutional blocks in each repeat
    parser.add_argument('-X', '-X', type=int, default=2)  # Number of repeats
    parser.add_argument('-R', '-R', type=int, default=2)  # Number of speakers

    args = parser.parse_args()
    print(args)
    return args
def train(args):
    loss = models.SISDR()

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = os.path.join(args.model_dir, (args.topology + '_' + 'model[{}]').format(time_stamp))
    train_ds = DataGenerator("train", args).fetch()
    valid_ds = DataGenerator("valid", args).fetch()
    opt = tf.optimizers.Adam()
    os.makedirs(model_dir)
    json.dump(vars(args), open(os.path.join(model_dir, 'params.json'), 'w'), indent=4)
    log_path = os.path.join(model_dir, 'train_log[{}].csv'.format(time_stamp))
    checkpoint_path = os.path.join(model_dir, 'model_best[{}].h5'.format(time_stamp))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.8, patience=3, min_lr=0.),
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
    model.compile(optimizer=opt, loss = loss)
    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=args.max_epoch, callbacks=callbacks)
    plot_learning_curves(history, model_dir)



def plot_learning_curves(history, model_dir):
    plt.style.use('ggplot')
    history_dict = history.history
    loss = [-_ for _ in history_dict['loss']]
    val_loss = [-_ for _ in history_dict['val_loss']]
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(14, 5))
    plt.plot(epochs, loss, marker='.', label='Training')
    plt.plot(epochs, val_loss, marker='.', label='Validation')
    plt.title('Training and validation performance')
    plt.xlabel('Epoch')
    plt.grid(axis='x')
    plt.ylabel('SI-SDR [dB]')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_dir, 'learning_curves.pdf'))

def log_evaluation_results(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model):
    with open(os.path.join(model_dir, "Evaluation results[" + model_time_stamp + "].txt"), "w") as f:
        f.write("Topology: {:s}\n".format(args.topology))
        f.write("Mean SI-SDR: {:.2f} dB\n".format(test_snr))
        f.write("Left SI-SDR: {:.2f} dB\n".format(snr_left))
        f.write("Right SI-SDR: {:.2f} dB\n".format(snr_right))
        f.write("Number of parameters: {:d}\n".format(count_params(model.trainable_variables)))
        f.write("S: {:d}".format(args.S))
    f.close()

def log_evaluation_excel(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model):
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


def load_and_evaluate_model(args, model_dir, model_path, model_time_stamp, mode):
    loss = models.SISDR()

    model = Model(args).call()
    model.load_weights(model_path)
    print('trained model loaded')
    model.compile(loss=loss)

    test_ds = DataGenerator("valid", args).fetch()

    test_logger = tf.keras.callbacks.CSVLogger(f'{mode}_log[{model_time_stamp}].csv', separator=',', append=False)
    test_logger.on_test_begin = test_logger.on_train_begin
    test_logger.on_test_batch_end = test_logger.on_epoch_end
    test_logger.on_test_end = test_logger.on_train_end

    print(f"\n{mode.capitalize()}ing model... \n")
    _, snr_left, snr_right = model.evaluate(test_ds, callbacks=test_logger, verbose=0)
    snr_left = -snr_left
    snr_right = -snr_right
    test_snr = np.mean([snr_left, snr_right])
    print("Mean SI-SDR: {:.2f} dB".format(test_snr))
    log_evaluation_results(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model)
    log_evaluation_excel(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model)

    save_path = os.path.join(model_dir, "audio")
    if not os.path.exists('audio'):
        os.makedirs('audio')
    test_ds = DataGenerator("test", args).fetch()
    model.compile(loss=loss)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        write_to_audio(model, test_ds, args, save_path)
    else:
        write_to_audio(model, test_ds, args, save_path)

def evaluate_test(args):
    trained_models = os.listdir(args.model_dir)

    for tm in trained_models:
        model_time_stamp = tm[-20:-1]
        model_dir = os.path.join(args.model_dir, tm)
        params_file = os.path.join(model_dir, "params.json")
        with open(params_file) as f:
            args_dict = json.load(f)
        args = namedtuple("args", args_dict.keys())(*args_dict.values())
        model_path = os.path.join(model_dir, f"model_best[{model_time_stamp}].h5")
        load_and_evaluate_model(args, model_dir, model_path, model_time_stamp, mode="evaluate")

def predict_audio(args):
    print('1')
    model_dir = "results"
    print('1')
    model_path = os.path.join(model_dir, "model_best.h5")
    model_time_stamp = "kw-model"
    save_path = os.path.join(model_dir, "audio")
    print('111')
    load_and_evaluate_model(args, model_dir, model_path, model_time_stamp, mode="predict")

def write_to_audio(model, ds, args, path):
    '''
    写出模型预测的音频文件
    '''
    c = 0
    for inp in ds:
        c += 1
    j = 1
    i = 0

    print("\nWriting predictions to audio...\n")

    for inp in tqdm(ds, total = c, position = 0, leave = True):

        prediction_left, prediction_right = pad_and_separate(inp, args, model)

        l = len(prediction_left[-1])

        prediction_left = resample(prediction_left[-1], int(l * (44100 / args.sample_rate) + 1))

        prediction_right = resample(prediction_right[-1], int(l * (44100 / args.sample_rate) + 1))

        predicted = tf.stack([prediction_left, prediction_right], 0)

        predicted = predicted/np.max(np.abs(predicted))

        fname = os.path.join(path, '{}_HA-output.wav'.format(i))
        string = tf.audio.encode_wav(tf.transpose(predicted), 44100)
        tf.io.write_file(fname, string, name='{}_HA-output.wav'.format(i))
        j += 1
        i += 1
        if j == c-1:
            print("\nDone!")


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

def pad_and_separate(inp, args, model):
    inp=np.array(inp)
    original_length = inp[0].shape[-1]
    slices = (original_length -tf.math.floormod(original_length,
                                                args.duration*args.sample_rate).numpy())//(args.duration*args.sample_rate) + 1
    left_inp_pad = np.zeros((1, slices*args.sample_rate*args.duration), dtype = np.float32)
    right_inp_pad = np.zeros((1, slices*args.sample_rate*args.duration), dtype = np.float32)
    left_inp_pad[0][0:original_length] = inp[0][0]
    right_inp_pad[0][0:original_length] = inp[1][0]

    for i in range(slices):
        prediction = model((left_inp_pad[:,i*args.sample_rate*args.duration:(i+1)*args.sample_rate*args.duration],
                            right_inp_pad[:, i*args.sample_rate*args.duration:(i+1)*args.sample_rate*args.duration]))
        if i == 0:
            prediction_left = prediction[0]
            prediction_right = prediction[1]
        else:
            prediction_left = tf.concat([prediction_left, prediction[0]], axis = 1)
            prediction_right = tf.concat([prediction_right, prediction[1]], axis = 1)

    return prediction_left[:,0:original_length], prediction_right[:,0:original_length]


def extract_numerical_part(filename):

    return int(''.join(filter(str.isdigit, filename)))

def computeMBSTOI(clean_folder_path, noisy_folder_path,model):
    args = setup()

    test_ds = DataGenerator("test", args).fetch()

    write_to_audio(model, test_ds, args, noisy_folder_path)

    clean_files = []
    noisy_files = []

    for filename in os.listdir(clean_folder_path):
        if filename.endswith("target_CH1.wav"):
            clean_files.append(filename)

    for filename in os.listdir(noisy_folder_path):
        if filename.endswith("output.wav"):
            noisy_files.append(filename)

    clean_files = sorted(clean_files, key=extract_numerical_part)
    noisy_files = sorted(noisy_files, key=extract_numerical_part)

    total_mbstoi = 0.0
    num_files = len(clean_files)
    num_files_noisy = len(noisy_files)


    mbstoi_scores = []

    for i in range(num_files):
        clean_file = clean_files[i]
        noisy_file = noisy_files[i]

        clean_file_path = os.path.join(clean_folder_path, clean_file)
        noisy_file_path = os.path.join(noisy_folder_path, noisy_file)

        fs_clean, clean_audio = wavfile.read(clean_file_path)
        fs_noisy, noisy_audio = wavfile.read(noisy_file_path)

        clean_audio_left = clean_audio[:, 0]
        clean_audio_right = clean_audio[:, 1]
        noisy_audio_left = noisy_audio[:, 0]
        noisy_audio_right = noisy_audio[:, 1]

        mbstoi_score = mbstoi(clean_audio_left, clean_audio_right, noisy_audio_left, noisy_audio_right, fs_clean)

        total_mbstoi += mbstoi_score
        mbstoi_scores.append(mbstoi_score)

    average_mbstoi = total_mbstoi / num_files


    # np.savetxt("average_mbstoi.txt", [average_mbstoi], fmt='%.6f')

    return np.array(mbstoi_scores)


def flatten(weights):
    flattened_weights = []
    for w in weights:
        if isinstance(w, np.ndarray):
            if w.size > 0:
                flattened_weights.extend(w.flatten())
            else:
                flattened_weights.append(w)  # jia yi ge kong de
        else:
            flattened_weights.append(w)

    return np.array(flattened_weights)


def unflatten(flattened_weights, original_shapes):
    weights = []
    current_index = 0

    for shape in original_shapes:
        weight_size = np.prod(shape)

        if current_index + weight_size > len(flattened_weights):
            weight = np.zeros(shape)
        else:
            weight = np.array(flattened_weights[current_index:current_index+weight_size]).reshape(shape)
        weights.append(weight)
        current_index += weight_size

    return weights

def bernoulli(n):

    d = -np.ones(n)
    mask = np.random.randint(2, size=n)
    d[mask==1] = 1
    return d

def kieferWolfowitz(steps, clean_folder_path, noisy_folder_path, savepath,
                    performance=False, logpath=None, startstep=0):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) is a Variant of the Kiefer Wolfowitz Algorithm
    """

    # alpha gamma from spall1998
    alpha = 0.602
    gamma = 0.101
    a_0 = 1
    upperbound = 100
    lowerbound = -100
    # should be determined with smac. Now from ortmann ma

    A = 11337
    c_0 = 0.0129

    args = setup()

    model = Model(args).call()
    model_folders = [folder for folder in os.listdir('models') if os.path.isdir(os.path.join('models', folder))]
    if len(model_folders) == 0:
        print("no trained models foundy.")
        exit()


    model_folder_path = os.path.join('models', model_folders[0])
    model_files = [file for file in os.listdir(model_folder_path) if file.endswith('.h5')]
    if len(model_files) == 0:
        print("no .h5 model files found.")
        exit()

    model_path = os.path.join(model_folder_path, model_files[0])

    from main import  get_model
    model = get_model(args)
    model.load_weights(model_path)

    # model.load_weights('models/Double_attention_model[24-08-2023_23-36-47]/model_best[24-08-2023_23-36-47].h5')
    weights = model.get_weights()
    if len(weights) > 0:
        print("model loaded successfully.")
    else:
        print("failed to load")

    original_weight_shapes = []

    for layer in model.layers:
        layer_weights = layer.get_weights()
        for weight_tensor in layer_weights:
            original_weight_shapes.append(weight_tensor.shape)
    weights = flatten(weights)

    best_mbstoi = -np.inf
    best_weights = None
    results = []

    for step in range(startstep, steps + 1):

        a_n = a_0 / (A + step + 1) ** alpha

        c_n = c_0 / (step + 1) ** gamma

        d_n = bernoulli(len(weights))

        weights_plus = unflatten(weights + c_n * d_n,original_weight_shapes)

        model.set_weights(weights_plus)

        result_1 = computeMBSTOI(clean_folder_path, noisy_folder_path,model)

        func_plus = np.mean(result_1)

        weights_minus = unflatten(weights - c_n * d_n,original_weight_shapes)

        model.set_weights(weights_minus)

        func_minus = np.mean(computeMBSTOI(clean_folder_path, noisy_folder_path,model))

        weights = weights + a_n * ((func_plus - func_minus) / (2 * c_n * d_n))

        weights[weights > upperbound] = upperbound

        weights[weights < lowerbound] = lowerbound

        weights_unflattend = unflatten(weights,original_weight_shapes)

        model.set_weights(weights_unflattend)

        if performance and step % 50 == 0:
            eval = np.mean(computeMBSTOI(clean_folder_path, noisy_folder_path,model))

            results.append(eval)
            print('eval score:',eval)

        elif not performance:

            eval = np.mean(computeMBSTOI(clean_folder_path, noisy_folder_path,model))

            results.append(eval)
            print('eval score:', eval)

            if eval > best_mbstoi:
                best_mbstoi = eval
                best_weights = model.get_weights()

    if savepath:
        model.set_weights(best_weights)
        model.save_weights(savepath)
        np.savetxt("results/best_mbstoi.txt", [best_mbstoi], fmt='%.6f')

    return results

def plotMBSTOI(results, output_file=None):
    plt.plot(results)
    plt.xlabel('Step')
    plt.ylabel('MBSTOI score')
    plt.title('MBSTOI')

    if output_file:
        plt.savefig(output_file)
    plt.show()


def run_optimization_and_plot():
    steps = 2
    clean_folder_path = 'data/test'
    noisy_folder_path = 'audio'
    logpath = 'results/result.txt'
    savepath = 'results/model_best.h5'
    if not os.path.exists('results'):
        os.makedirs('results')
    
    output_file_path = 'results/KW_process.txt'
    with open(output_file_path, 'w') as f:

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        results = kieferWolfowitz(steps, clean_folder_path, noisy_folder_path,
                                savepath, performance=False, logpath=logpath, startstep=0)

        plotMBSTOI(results, output_file='results/mbstoi_plot.png')
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    output_data = {
        'captured_output': open(output_file_path).read()
    }
    output_mat_path = 'results/output_data.mat'
    scipy.io.savemat(output_mat_path, output_data)
    print(f"output saved to {output_mat_path}")

