A napari plugin to count cells using a U-Net for cell identification

Model uses ResNet architecture layers as encoders, and requires custom trained model weights (model trained using resnet34 imagenet1_v1 pretrained weights as starting point)

Current Model works well for DAB stained neuronal cells of 30micron mouse brain sections

Install using:
pip install napari_dab_cellcount

Run using:
napari napari_dab_cellcount
