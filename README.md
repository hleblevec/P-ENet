# P-ENet: A pipelined dataflow architecture implementing a FPGA-oriented redesign of ENet

This repository contains the code to train and implement P-ENet, which is an improved version of ENet for deployment on the TySOM-3A-ZU19EG FPGA board. 

### Training & Evaluating
The code for training and validation of the model is greatly inspired by RegSeg: https://github.com/RolandGao/RegSeg

It uses the Pytorch library, as well as Brevitas (https://github.com/Xilinx/brevitas) to perform Quantization-Aware Training. 
To use the code, make sure you have the correct dependencies. You need Python >= 3.10, and the libraries in `train/requirements.txt`.
Make sure you have downloaded the Cityscapes and/or CamVid datasets, and run 
```
CITYSCAPES_DATASET=cityscapes_dataset csCreateTrainIdLabelImgs
``` 
where `cityscapes_dataset` is the path of the Cityscapes folder (see https://github.com/mcordts/cityscapesScripts for troubleshooting).

Training can be done by running:
```
python train.py -c config -d path_to_dataset
```
using one of the config files provided in `train/configs`.

|Dataset | Setup | Config file|
|--------|-------|------------|
|Cityscapes| Float | cityscapes_float_1000epochs.yaml|
|Cityscapes| QAT | cityscapes_1000epochs.yaml|
|Cityscapes| QAT+FT | cityscapes_ft_200epochs.yaml|
|CamVid| Float | camvid_float_1000epochs.yaml|
|CamVid| QAT | camvid_1000epochs.yaml|

Additionally, you can use the checkpoints available in `train/checkpoints` to perform validation using 
```
python validate.py -c config -p pretrained_path -d path_to_dataset
```
|Dataset | Setup | Val mIoU | Checkpoint file|
|--------|-------|----------| -----------|
|Cityscapes | QAT | 69.76% | p-enet_cityscapes_1000epochs_run1 |
|Cityscapes | QAT+FT | 70.33% | p-enet_cityscapes_ft_200epochs_run1 |
|CamVid | QAT | 76.26% | p-enet_cityscapes_ft_200epochs_run1 |



### Hardware
The deployment on TySOM-3A-ZU19EG is done using the FINN framework to generate the bitfile. See https://finn.readthedocs.io/en/latest/getting_started.html for installation and requirements.

We use a modified version of the FINN repository that can be obtained by executing  
```
./get_finn.sh
```

To build P-ENet_M, do
```
cd finn/
./run-docker.sh
```
Then within the docker environment:
```
cd ../hardware
python build.py -m models/p-enet_cityscapes.onnx -f configs/p-enet_m_config.json -o output
```

The generated outputs will be under the `output` directory.