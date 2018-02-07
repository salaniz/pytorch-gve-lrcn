from models.model_loader import ModelLoader
from train.trainer import Trainer
from utils.data.data_prep import DataPreparation
import utils.arg_parser

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

    # Get trainer
    trainer = getattr(Trainer, args.model)(args, model, dataset, data_loader)

    # Start training
    print("Training ...")
    for epoch in range(args.num_epochs):
        trainer.train_epoch()
