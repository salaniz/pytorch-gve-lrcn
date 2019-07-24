# PyTorch Models for GVE and LRCN
PyTorch implementation of the following models:
* Long-term Recurrent Convolutional Networks (LRCN) [1]
* Generating Visual Explanations (GVE) [2]

## Installation
This implementation uses Python 3, PyTorch and pycocoevalcap (https://github.com/salaniz/pycocoevalcap).  
All dependencies can be installed into a conda environment with the provided environment.yml file.

1. Clone the repository
```shell
git clone https://github.com/salaniz/pytorch-gve-lrcn.git
cd pytorch-gve-lrcn
```
2. Create conda environment
```shell
conda env create -f environment.yml
```
3. Activate environment
```shell
conda activate gve-lrcn
```
4. Download scripts are provided for the datasets:
* `coco-data-setup-linux.sh` downloads COCO 2014: http://cocodataset.org/ [3]
* `cub-data-setup-linux.sh` downloads preprocessed features of CUBS-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html [4]

## Usage
1. Train LRCN on COCO
```
python main.py --model lrcn --dataset coco
```

2. Train GVE on CUB
* To train GVE on CUB we first need a sentence classifier:
```
python main.py --model sc --dataset cub
```
* Copy the saved model to the default path (or change the path to your model file) and then run the GVE training:
```
cp ./checkpoints/sc-cub-D<date>-T<time>-G<GPUid>/best-ckpt.pth ./data/cub/sentence_classifier_ckpt.pth
python main.py --model gve --dataset cub --sc-ckpt ./data/cub/sentence_classifier_ckpt.pth
```

3. Evaluation
* By default, model checkpoints and validation results are saved in the checkpoint directory using the following naming convention: `<model>-<dataset>-D<date>-T<time>-G<GPUid>`
* To make a single evaluation on the test set, you can run
```
python main.py --model gve --dataset cub --eval ./checkpoints/gve-cub-D<date>-T<time>-G<GPUid>/best-ckpt.pth
```
Note: Since COCO does not come with test set annotations, this script evaluates on the validation set when run on the COCO dataset


### Default parameters
```
data_path                     ./data
checkpoint_path               ./checkpoints
log_step                      10
num_workers                   4
disable_cuda                  False
cuda_device                   0
torch_seed                    <random>
model                         lrcn
dataset                       coco
pretrained_model              vgg16
layers_to_truncate            1
sc_ckpt                       ./data/cub/sentence_classifier_ckpt.pth
weights_ckpt                  None
loss_lambda                   0.2
embedding_size                1000
hidden_size                   1000
num_epochs                    50
batch_size                    128
learning_rate                 0.001
train                         True
eval_ckpt                     None
```

### All command line options
```
$ python main.py --help
usage: main.py [-h] [--data-path DATA_PATH]
               [--checkpoint-path CHECKPOINT_PATH] [--log-step LOG_STEP]
               [--num-workers NUM_WORKERS] [--disable-cuda]
               [--cuda-device CUDA_DEVICE] [--torch-seed TORCH_SEED]
               [--model {lrcn,gve,sc}] [--dataset {coco,cub}]
               [--pretrained-model {resnet18,resnet34,resnet50,resnet101,resnet152,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19_bn,vgg19}]
               [--layers-to-truncate LAYERS_TO_TRUNCATE] [--sc-ckpt SC_CKPT]
               [--weights-ckpt WEIGHTS_CKPT] [--loss-lambda LOSS_LAMBDA]
               [--embedding-size EMBEDDING_SIZE] [--hidden-size HIDDEN_SIZE]
               [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
               [--learning-rate LEARNING_RATE] [--eval EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        root path of all data
  --checkpoint-path CHECKPOINT_PATH
                        path checkpoints are stored or loaded
  --log-step LOG_STEP   step size for prining logging information
  --num-workers NUM_WORKERS
                        number of threads used by data loader
  --disable-cuda        disable the use of CUDA
  --cuda-device CUDA_DEVICE
                        specify which GPU to use
  --torch-seed TORCH_SEED
                        set a torch seed
  --model {lrcn,gve,sc}
                        deep learning model
  --dataset {coco,cub}
  --pretrained-model {resnet18,resnet34,resnet50,resnet101,resnet152,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19_bn,vgg19}
                        [LRCN] name of pretrained model for image features
  --layers-to-truncate LAYERS_TO_TRUNCATE
                        [LRCN] number of final FC layers to be removed from
                        pretrained model
  --sc-ckpt SC_CKPT     [GVE] path to checkpoint for pretrained sentence
                        classifier
  --weights-ckpt WEIGHTS_CKPT
                        [GVE] path to checkpoint for pretrained weights
  --loss-lambda LOSS_LAMBDA
                        [GVE] weight factor for reinforce loss
  --embedding-size EMBEDDING_SIZE
                        dimension of the word embedding
  --hidden-size HIDDEN_SIZE
                        dimension of hidden layers
  --num-epochs NUM_EPOCHS
  --batch-size BATCH_SIZE
  --learning-rate LEARNING_RATE
  --eval EVAL           path of checkpoint to be evaluated
```

## References
1. Donahue J., Hendricks L.A., Guadarrama S., Rohrbach M., Venugopalan S., Saenko K., Darrell T., "Long-term Recurrent Convolutional Networks for Visual Recognition and Description", Conference on Computer Vision and Pattern Recognition (CVPR), 2015
2. Hendricks L.A., Akata Z., Rohrbach M., Donahue J., Schiele B., Darrell T., "Generating Visual Explanations", European Conference on Computer Vision (ECCV), 2016
3. Lin T.Y., Maire M., Belongie S. et al., "Microsoft COCO: Common Objects in Context", European Conference in Computer Vision (ECCV), 2014.
4. Wah C., Branson S., Welinder P., Perona P., Belongie S., "The Caltech-UCSD Birds-200-2011 Dataset." Computation & Neural Systems Technical Report, CNS-TR-2011-001.
