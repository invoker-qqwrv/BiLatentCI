import os
import sys

sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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


def train(args, resume_training=False):
    loss = models.SISDR()

    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    model_dir = os.path.join(args.model_dir, (args.topology + '_' + 'model[{}]').format(time_stamp))

    if resume_training:
        # Get the latest checkpoint directory for resuming training
        checkpoints = os.listdir(args.model_dir)
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        model_dir = os.path.join(args.model_dir, latest_checkpoint)

    train_ds = DataGenerator("train", args).fetch()

    valid_ds = DataGenerator("valid", args).fetch()

    opt = tf.optimizers.Adam()

    os.makedirs(model_dir)

    json.dump(vars(args), open(os.path.join(model_dir, 'params.json'), 'w'), indent=4)

    log_path = os.path.join(model_dir, 'train_log[{}].csv'.format(time_stamp))

    checkpoint_path = os.path.join(model_dir, 'model_best[{}].h5'.format(time_stamp))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.),
        tf.keras.callbacks.CSVLogger(log_path),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           mode='auto', save_freq='epoch')]

    physical_devices = tf.config.list_physical_devices()

    if len(physical_devices) == 1:
        print("\nUsing CPU... \n")
        warnings.warn("Depthwise convolution not available when using CPU for processing.")
        args.GPU = False
    else:
        print("\nUsing GPU... \n")

    model = Model(args).call()

    # Step 1: Train the model with FRAE layer frozen and other layers unfrozen
    for layer in model.layers:
        if layer.name not in ["seaencoder_left", "seaencoder_right", "seadecoder_left", "seadecoder_right",
                              "seaencoder_left_tcn","seaencoder_right_tcn","seadecoder_left_tcn",
                              "seadecoder_right_tcn"]:
            layer.trainable = True  # Unfreeze other layers
        else:
            layer.trainable = False  # Freeze FRAE layers

    model.compile(optimizer=opt, loss=loss)

    history_step1 = model.fit(train_ds, validation_data=valid_ds, epochs=args.max_epoch, callbacks=callbacks)

    # Step 2: Save the weights of the trained model after Step 1
    model.save_weights(os.path.join(model_dir, "step1_weights.h5"))

    # Step 3: Load the saved weights into a new model with FRAE layer frozen
    model_frae_frozen = Model(args).call()
    model_frae_frozen.load_weights(os.path.join(model_dir, "step1_weights.h5"))

    # Freeze all layers except the FRAE layers for Step 3
    for layer in model_frae_frozen.layers:
        if layer.name not in ["seaencoder_left", "seaencoder_right", "seadecoder_left", "seadecoder_right",
                              "seaencoder_left_tcn","seaencoder_right_tcn","seadecoder_left_tcn",
                              "seadecoder_right_tcn"]:
            layer.trainable = False  # Freeze other layers

    model_frae_frozen.compile(optimizer=opt, loss=loss)

    # Step 3: Train the new model to update the FRAE layer while keeping other layers fixed
    history_step3 = model_frae_frozen.fit(train_ds, validation_data=valid_ds, epochs=args.max_epoch,
                                          callbacks=callbacks)

    # The final trained model with both FRAE and other layers updated
    final_model = model_frae_frozen

    # Rest of the function remains unchanged...

    # Return the final trained model
    return final_model


def evaluate(args, model):
    loss = models.SISDR()

    model_time_stamp = model[-20:-1]

    model_dir = os.path.join(args.model_dir, model)

    log_path = os.path.join(model_dir, 'test_log[{}].csv'.format(model_time_stamp))

    params_file = os.path.join(model_dir, "params.json")

    with open(params_file) as f:
        args = json.load(f)

    args = namedtuple("args", args.keys())(*args.values())

    test_ds = DataGenerator("valid", args).fetch()

    physical_devices = tf.config.list_physical_devices("GPU")

    if not physical_devices:
        args.GPU = False

    model = Model(args).call()

    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))
    print('trained model lodaed')

    model.compile(loss=loss)

    test_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)
    test_logger.on_test_begin = test_logger.on_train_begin
    test_logger.on_test_batch_end = test_logger.on_epoch_end
    test_logger.on_test_end = test_logger.on_train_end

    print("\nEvaluating model... \n")

    _, snr_left, snr_right = model.evaluate(test_ds, callbacks=test_logger, verbose=0)
    snr_left = -snr_left
    snr_right = -snr_right
    test_snr = np.mean([snr_left, snr_right])

    print("Mean SI-SDR: {:.2f} dB".format(test_snr))

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
    data = (
        ['Topology', args.topology],
        ['MeanSDR', test_snr],
        ['LeftSDR', snr_left],
        ['RightSDR', snr_right],
        ['Params', count_params(model.trainable_variables)],
        ['S', args.S])

    row = 0
    col = 0

    for var, value in (data):
        worksheet.write(row, col, var)
        worksheet.write(row, col + 1, value)
        row += 1

    workbook.close()


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

    model.compile(loss=loss)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        write_to_audio(model, test_ds, args, save_path)
    else:
        write_to_audio(model, test_ds, args, save_path)


def main():
    args = setup()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        trained_models = os.listdir(args.model_dir)
        for tm in trained_models:
            evaluate(args, tm)
    else:
        trained_models = os.listdir(args.model_dir)
        for tm in trained_models:
            test(args, tm)


if __name__ == '__main__':
    main()