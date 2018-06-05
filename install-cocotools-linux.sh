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
echo "INFO: Done."
