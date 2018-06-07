from argparse import Namespace

import collections
import torch
import utils.arg_parser
from PIL import Image
from models.model_loader import ModelLoader
from torchvision import transforms
from utils.data.data_prep import DataPreparation
from utils.misc import get_split_str

from train import lrcn_trainer

args = Namespace(pretrained_model = 'vgg16', embedding_size=1000, hidden_size=1000, layers_to_truncate=1,
                 factored=True, cuda=False, model="lrcn", dataset = "cub", train = False, data_path= "./data",
                 batch_size = 20, num_workers = 1, learning_rate=0.1, num_epochs=3, log_step=1,
                 weights_ckpt='/home/christina/Desktop/Project AI/pytorch-vision-language/gve-cub-D2018-06-04-T14-35-50-G0/ckpt-e3.pth')

# Parse arguments
#args = utils.arg_parser.get_args()
# Print arguments
utils.arg_parser.print_args(args)

if args.cuda:
    torch.cuda.set_device(args.cuda_device)

#job_string = time.strftime("{}-{}-D%Y-%m-%d-T%H-%M-%S-G{}".format(args.model, args.dataset, args.cuda_device))

#job_path = os.path.join(args.checkpoint_path, job_string)

# Create new checkpoint directory
# if not os.path.exists(job_path):
#os.makedirs(job_path)

# Save job arguments
#with open(os.path.join(job_path, 'config.json'), 'w') as f:
#    json.dump(vars(args), f)

# Data preparation
print("Preparing Data ...")
split = get_split_str(args.train)
data_prep = DataPreparation(args.dataset, args.data_path)
dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
                                                        batch_size=args.batch_size, num_workers=args.num_workers)
# Load Model
model = ModelLoader(args, dataset)

trainer = lrcn_trainer.LRCNTrainer(args, model.lrcn(), dataset, data_loader, None)
#lrcn =
#gve = model.gve()


#print(lrcn)
#print(gve)

def get_image_tensor(path):
    loader = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    image = Image.open(path)
    image = loader(image).float()
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image  # assumes that you're using GPU


tensor = get_image_tensor('/home/christina/Desktop/Project AI/pytorch-vision-language/data/cub/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')

print(tensor.size())

op = trainer.eval_step(tensor, {0: 1})

print(op)