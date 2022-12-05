#!/bin/bash

# This will download all data you need to run "Generating Visual Explanation" code.  You will need the coco evaluation toolbox as well.
# The data is located on google drive at: https://drive.google.com/drive/folders/1nU9ATTLHAM6_jz-K6hoVlDzNrFcOXtyH

gdrive_file_ids=( "1RHBLIo7sp8nhKjX0NQaGLaA-0ChYIunR" "1O5hEl7BYP0o1sYJONkPFbp9T-aVrNQZ0" )
gdrive_file_names=( "data.zip" "cider_scores.zip" )

echo "Downloading data..."

mkdir -p data/cub
cd data/cub

for i in "${!gdrive_file_ids[@]}"
do
    gdown ${gdrive_file_ids[i]} -O ${gdrive_file_names[i]}
    unzip ${gdrive_file_names[i]}
    rm ${gdrive_file_names[i]}
done

mv data/* .
rm -r data/
cd ../..

# Unify naming
mv data/cub/train_noCub.txt data/cub/train.txt

echo "Preprocessing text data..."
python utils/cub_preprocess_captions.py --description_type bird \
                                    --splits data/cub/train.txt,data/cub/val.txt,data/cub/test.txt

echo "Done downloading and pre-processing data."
