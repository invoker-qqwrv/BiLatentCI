import os
from models import Model
from utils import setup
from utils import train,evaluate_test,predict_audio,run_optimization_and_plot

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
    elif args.mode == "predict":
        predict_audio(args)
    elif args.mode == "KW":
        run_optimization_and_plot()

if __name__ == '__main__':
    main()
