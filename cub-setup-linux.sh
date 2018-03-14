#!/bin/bash

#This will download all data you need to run "Generating Visual Explanation" code.  You will need the coco evaluation toolbox as well.

data_files=( "CUB_feature_dict.p" "CUB_label_dict.p" "bilinear_preds.p" "cub_0917_5cap.tsv" "train_noCub.txt" "val.txt" "test.txt" "description_sentence_features.p" "CUB_vocab_noUNK.txt")
cider_scores=( "cider_score_dict_definition.p" "cider_score_dict_description.p" "cider_score_dict_explanation-dis.p" "cider_score_dict_explanation-label.p" "cider_score_dict_explanation.p" )

echo "Downloading data..."

mkdir -p data/cub
cd data/cub
for i in "${data_files[@]}"
do 
  echo "Downloading: " $i
  if [ ! -f $i ];
  then
    wget https://people.eecs.berkeley.edu/~lisa_anne/generating_visual_explanations/data/$i
  fi
done
cd ../..

# Unify naming
mv data/cub/train_noCub.txt data/cub/train.txt

echo "Preprocessing text data..."
python utils/cub_preprocess_captions.py --description_type bird \
                                    --splits data/cub/train.txt,data/cub/val.txt,data/cub/test.txt

echo "Downloading cider scores..."
mkdir -p data/cub/cider_scores 
cd data/cub/cider_scores
for i in "${cider_scores[@]}"
do 
  echo "Downloading: " $i
  if [ ! -f $i ];
  then
    wget https://people.eecs.berkeley.edu/~lisa_anne/generating_visual_explanations/cider_scores/$i
  fi
done
cd ../../..

echo "Done downloading and pre-processing data."
