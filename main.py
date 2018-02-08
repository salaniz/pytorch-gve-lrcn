from models.model_loader import ModelLoader
from train.trainer import Trainer
from utils.data.data_prep import DataPreparation
import utils.arg_parser

import json
import torch

if __name__ == '__main__':

    # Parse arguments
    args = utils.arg_parser.get_args()
    # Print arguments
    utils.arg_parser.print_args(args)

    # Data preparation
    print("Preparing Data ...")
    data_prep = DataPreparation(args.data_path, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    dataset, data_loader = getattr(data_prep, args.dataset)(args.pretrained_model, args.train)
    print()

    print("Loading Model ...")
    ml = ModelLoader(args, dataset)
    model = getattr(ml, args.model)()
    print(model, '\n')

    if not args.train:
        print("Loading Model Weights ...")
        model.load_state_dict(torch.load("/home/stephan/HDD/lrcn-31-1000.pkl",
            map_location=lambda storage, loc: storage), strict=False)
        #model.eval()

    # Get trainer
    trainer = getattr(Trainer, args.model)(args, model, dataset, data_loader)

    if args.train:
        print("Training ...")
    else:
        print("Evaluating ...")
        vars(args)['num_epochs'] = 1

    # Start training/evaluation
    for epoch in range(args.num_epochs):
        if args.train:
            trainer.train_epoch()
        else:
            result = trainer.train_epoch()

    if not args.train:
        with open('results.json', 'w') as f:
            json.dump(result, f)
