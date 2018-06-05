# Check if pycocotools are already installed
cwd=$(pwd)
read -p "Download COCO 2014 data to $cwd/data/coco (~20GB)? [y/N]:" -n 1 -r
echo # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "INFO: Downloading COCO 2014 data."
    # Download annotations to ./data/coco directory
    mkdir -p data/coco
    cd data/coco
    # Captions
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
    rm annotations_trainval2014.zip
    # Train images
    wget http://images.cocodataset.org/zips/train2014.zip
    unzip train2014.zip
    rm train2014.zip
    # Validation images
    wget http://images.cocodataset.org/zips/val2014.zip
    unzip val2014.zip
    rm val2014.zip
else
    echo "INFO: Not downloading COCO data."
fi

echo "INFO: Done."
