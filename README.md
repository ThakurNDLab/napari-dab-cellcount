A napari plugin to count cells using a U-Net for cell identification

Model uses ResNet architecture layers as encoders, and requires custom trained model weights (model trained using resnet50 imagenet1_v1 pretrained weights as starting point)

Current Model works well for DAB stained neuronal cells of 30micron mouse brain sections

Install using:
pip install napari-dab-cellcount

If you want to use GPU:
pip install cuda-python
pip install nvidia-pyindex
pip install nvidia-cudnn
pip install napari-dab-cellcount

Run using:
napari
