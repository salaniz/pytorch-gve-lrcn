import argparse
import torch

from models.pretrained_models import PretrainedModel

def get_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--data-path', type=str,
                        default='./data',
                        help="root path of all data")
    parser.add_argument('--save-step', type=int , default=1000,
                        help="step size for saving model checkpoints")
    parser.add_argument('--log-step', type=int , default=10,
                        help="step size for prining logging information")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="number of threads used by data loader")

    parser.add_argument('--disable-cuda', action='store_true',
                        help="disable the use of CUDA")


    # Model parameters
    parser.add_argument('--model', type=str, default='lrcn',
                        help="deep learning model",
                        choices=['lrcn'])
    parser.add_argument('--pretrained-model', type=str, default='vgg16',
                        help="[LRCN] name of pretrained model for image features",
                        choices=PretrainedModel.SUPPORTED_MODEL_NAMES)
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco'])

    parser.add_argument('--embedding-size', type=int , default=1000,
                        help='dimension of the word embedding')
    parser.add_argument('--hidden-size', type=int , default=1000 ,
                        help='dimension of hidden layers')

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)


    args = parser.parse_args()

    vars(args)["cuda"] = torch.cuda.is_available() and not args.disable_cuda

    return args
