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
from utils import run_optimization_and_plot  # Assuming this utility function exists
tf.config.experimental_run_functions_eagerly(True)
from models import trainQuantizer
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



def evaluate_test(args):
    trained_models = os.listdir(args.model_dir)
    loss = models.SISDR()

    for tm in trained_models:
        model_time_stamp = tm[-20:-1]
        model_dir = os.path.join(args.model_dir, tm)
        log_path = os.path.join(model_dir, 'test_log[{}].csv'.format(model_time_stamp))
        params_file = os.path.join(model_dir, "params.json")
        with open(params_file) as f:
            args_dict = json.load(f)
        args = namedtuple("args", args_dict.keys())(*args_dict.values())
        test_ds = DataGenerator("valid", args).fetch()

        model = get_model(args)
        model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))
        print('Trained model loaded')
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


        log_evaluation_results(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model)
        log_evaluation_excel(model_dir, model_time_stamp, args, test_snr, snr_left, snr_right, model)


        bits = [3, 4, 5]
        for b in bits:
            print("Evaluating model with quantization bit =", b)

            trainQuantizer(model.layers[4], n_clusters=2 ** b)
            trainQuantizer(model.layers[5], n_clusters=2 ** b)
            trainQuantizer(model.layers[9], n_clusters=2 ** b)
            trainQuantizer(model.layers[10], n_clusters=2 ** b)

            testQuantizedModel(args, model, b)

def evaluate_results_models():
    model_dir = "results"
    loss = models.SISDR()

    

    if os.path.isdir(model_dir) and "model_best.h5" in os.listdir(model_dir):
        model_file = os.path.join(model_dir, "model_best.h5")
        params_file = os.path.join(model_dir, "params.json")

        with open(params_file) as f:
            args_dict = json.load(f)

        args = namedtuple("args", args_dict.keys())(*args_dict.values())
        test_ds = DataGenerator("valid", args).fetch()

        model = Model(args).call()
        model.load_weights(model_file)
        print('Loaded model:', model_file)

        model.compile(loss=loss)
        print("\nEvaluating model... \n")
        _, snr_left, snr_right = model.evaluate(test_ds, verbose=0)

        snr_left = -snr_left
        snr_right = -snr_right
        test_snr = np.mean([snr_left, snr_right])
        print("Mean SI-SDR: {:.2f} dB".format(test_snr))

        

        bits = [3, 4, 5]
        for b in bits:
            print("Evaluating model with quantization bit =", b)

            trainQuantizer(model.layers[4], n_clusters=2 ** b)
            trainQuantizer(model.layers[5], n_clusters=2 ** b)
            trainQuantizer(model.layers[9], n_clusters=2 ** b)
            trainQuantizer(model.layers[10], n_clusters=2 ** b)


            testQuantizedModel(args, model, b)

def testQuantizedModel(args, model, bits):
    trained_models = os.listdir(args.model_dir)
    model_dir = os.path.join(args.model_dir, trained_models[0])
    save_path = os.path.join(model_dir, f"audio_bit={bits}")
    params_file = os.path.join(model_dir, "params.json")

    with open(params_file) as f:
        args = json.load(f)

    args = namedtuple("args", args.keys())(*args.values())

    test_ds = DataGenerator("test", args).fetch()

    physical_devices = tf.config.list_physical_devices("GPU")

    if not physical_devices:  # workbook.close()

        args.GPU = False

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        write_to_audio(model, test_ds, args, save_path)
    else:
        write_to_audio(model, test_ds, args, save_path)

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


def main():
    args = setup()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate_test":
        evaluate_test(args)
    elif args.mode == "KW":
        run_optimization_and_plot()
    elif args.mode == "evaluate_results_models":
        evaluate_results_models()
if __name__ == '__main__':
    main()