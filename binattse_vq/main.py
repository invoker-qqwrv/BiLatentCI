import os

from models import Model
from utils import setup, write_to_audio
from data_generator import DataGenerator

import tensorflow as tf
from utils import train,evaluate_test,evaluate_results_models,run_optimization_and_plot
tf.config.experimental_run_functions_eagerly(True)
from models import trainQuantizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def get_model(args):
    model = Model(args).call()
    model.summary()
    return model



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