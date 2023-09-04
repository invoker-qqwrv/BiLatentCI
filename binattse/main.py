import os
import json
import models
import xlsxwriter
import numpy as np
from models import Model
from utils import setup, write_to_audio
from data_generator import DataGenerator
from keras.utils.layer_utils import count_params
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple
from utils import run_optimization_and_plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_model(args):
    model = Model(args).call()
    model.summary()
    return model

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
    model_dir = "results"  # Change this to the appropriate path
    model_path = os.path.join(model_dir, "model_best.h5")
    model_time_stamp = "model_time_stamp"  
    save_path = os.path.join(model_dir, "audio")
    load_and_evaluate_model(args, model_dir, model_path, model_time_stamp, mode="predict")
def main():
    args = setup()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate_test":
        evaluate_test(args)
    elif args.mode == "predict":
        predict_audio(args)
    elif args.mode == "KW":
        run_optimization_and_plot()

if __name__ == '__main__':
    main()
