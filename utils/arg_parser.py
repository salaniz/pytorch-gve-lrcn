import argparse
import torch

from models.pretrained_models import PretrainedModel

def get_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--data-path', type=str,
                        default='./data',
                        help="root path of all data")
    parser.add_argument('--checkpoint-path', type=str,
                        default='./checkpoints',
                        help="path checkpoints are stored or loaded")
    parser.add_argument('--log-step', type=int , default=10,
                        help="step size for prining logging information")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="number of threads used by data loader")

    parser.add_argument('--disable-cuda', action='store_true',
                        help="disable the use of CUDA")
    parser.add_argument('--cuda-device', type=int , default=0,
                        help="specify which GPU to use")
    parser.add_argument('--torch-seed', type=int,
                        help="set a torch seed")


    # Model parameters
    parser.add_argument('--model', type=str, default='lrcn',
                        help="deep learning model",
                        choices=['lrcn', 'gve', 'sc'])
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco', 'cub'])
    parser.add_argument('--pretrained-model', type=str, default='vgg16',
                        help="[LRCN] name of pretrained model for image features",
                        choices=PretrainedModel.SUPPORTED_MODEL_NAMES)
    parser.add_argument('--layers-to-truncate', type=int, default=1,
                        help="[LRCN] number of final FC layers to be removed from pretrained model")
    parser.add_argument('--sc-ckpt', type=str, default='data/cub/sentence_classifier_ckpt.pth',
                        help="[GVE] path to checkpoint for pretrained sentence classifier")
    parser.add_argument('--weights-ckpt', type=str,
                        help="[GVE] path to checkpoint for pretrained weights")
    parser.add_argument('--loss-lambda', type=float, default=0.2,
                        help="[GVE] weight factor for reinforce loss")

    parser.add_argument('--embedding-size', type=int , default=1000,
                        help='dimension of the word embedding')
    parser.add_argument('--hidden-size', type=int , default=1000,
                        help='dimension of hidden layers')

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--eval', type=str,
                        help="path of checkpoint to be evaluated")

    args = parser.parse_args()

    arg_vars = vars(args)

    # TODO: Check if there is a direct way to do this
    arg_vars["train"] = not args.eval
    arg_vars["eval_ckpt"] = args.eval
    del arg_vars["eval"]

    # GVE currently does not support pretrained models
    if arg_vars["model"] == "gve":
        arg_vars["pretrained_model"] = None

    if args.torch_seed is not None:
        torch.manual_seed(arg_vars["torch_seed"])
    else:
        arg_vars["torch_seed"] = torch.initial_seed()

    return args


def print_args(args):
    space = 30
    print("Arguments:")
    for arg, value in vars(args).items():
        print('{:{space}}{}'.format(arg, value, space=space))
    print()
