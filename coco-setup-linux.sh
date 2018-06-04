# Check if pycocotools are already installed
cwd=$(pwd)
if python -c "import pycocotools" &> /dev/null; then
    echo "INFO: Pycocotools is already installed. To do a fresh local install, uninstall the current version first."
else
    echo "INFO: Installing pycocotools locally."
    # Install Python COCO API
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext --inplace
    rm -rf build
    cd $cwd
    mv cocoapi/PythonAPI/pycocotools .
    rm -rf cocoapi

    # Install COCO evaluation tools
    git clone https://github.com/salaniz/pycocoevalcap.git
fi


echo # Move to a new line
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
