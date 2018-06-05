import time
import os
import json

from models.model_loader import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep import DataPreparation
import utils.arg_parser
from utils.logger import Logger
from utils.misc import get_split_str


import torch

if __name__ == '__main__':

    # Parse arguments
    args = utils.arg_parser.get_args()
    # Print arguments
    utils.arg_parser.print_args(args)

    device = torch.device('cuda:{}'.format(args.cuda_device) if
            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

    job_string = time.strftime("{}-{}-D%Y-%m-%d-T%H-%M-%S-G{}".format(args.model, args.dataset, args.cuda_device))

    job_path = os.path.join(args.checkpoint_path, job_string)


    # Create new checkpoint directory
    #if not os.path.exists(job_path):
    os.makedirs(job_path)

    # Save job arguments
    with open(os.path.join(job_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # Data preparation
    print("Preparing Data ...")
    split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
    data_prep = DataPreparation(args.dataset, args.data_path)
    dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
            batch_size=args.batch_size, num_workers=args.num_workers)
    if args.train:
        val_dataset, val_data_loader = data_prep.get_dataset_and_loader('val',
                args.pretrained_model, batch_size=args.batch_size, num_workers=args.num_workers)

    # TODO: If eval + checkpoint load validation set

    print()

    print("Loading Model ...")
    ml = ModelLoader(args, dataset)
    model = getattr(ml, args.model)()
    print(model, '\n')

    # TODO: Remove and handle with checkpoints
    if not args.train:
        print("Loading Model Weights ...")
        evaluation_state_dict = torch.load(args.eval_ckpt)
        model_dict = model.state_dict(full_dict=True)
        model_dict.update(evaluation_state_dict)
        model.load_state_dict(model_dict)
        model.eval()

    if args.train:
        val_dataset.set_label_usage(dataset.return_labels)

    # Create logger
    logger = Logger(os.path.join(job_path, 'logs'))

    # Get trainer
    trainer_creator = getattr(TrainerLoader, args.model)
    trainer = trainer_creator(args, model, dataset, data_loader, logger, device)
    if args.train:
        evaluator = trainer_creator(args, model, val_dataset, val_data_loader,
            logger, device)
        evaluator.train = False

    if args.train:
        print("Training ...")
    else:
        print("Evaluating ...")
        vars(args)['num_epochs'] = 1


    # Start training/evaluation
    max_score = 0
    while trainer.curr_epoch < args.num_epochs:
        if args.train:
            trainer.train_epoch()


            # Eval & Checkpoint
            checkpoint_name = "ckpt-e{}".format(trainer.curr_epoch)
            checkpoint_path = os.path.join(job_path, checkpoint_name)

            model.eval()
            result = evaluator.train_epoch()
            if evaluator.REQ_EVAL:
                score = val_dataset.eval(result, checkpoint_path)
            else:
                score = result
            model.train()

            logger.scalar_summary('score', score, trainer.curr_epoch)

            # TODO: Eval model
            # Save the models
            checkpoint = {'epoch': trainer.curr_epoch,
                          'max_score': max_score,
                          'optimizer' : trainer.optimizer.state_dict()}
            checkpoint_path += ".pth"
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(checkpoint, os.path.join(job_path,
                "training_checkpoint.pth"))
            if score > max_score:
                max_score = score
                link_name = "best-ckpt.pth"
                link_path = os.path.join(job_path, link_name)
                if os.path.islink(link_path):
                    os.unlink(link_path)
                dir_fd = os.open(os.path.dirname(link_path), os.O_RDONLY)
                os.symlink(os.path.basename(checkpoint_path), link_name, dir_fd=dir_fd)
                os.close(dir_fd)

        else:
            result = trainer.train_epoch()
            if trainer.REQ_EVAL:
                score = dataset.eval(result, "results")


    if not args.train and args.model == 'sc':
        with open('results.json', 'w') as f:
            json.dump(result, f)

