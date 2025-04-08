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
where `cityscapes_dataset` is the path of the Cityscapes file (see https://github.com/mcordts/cityscapesScripts for troubleshooting).

Training can be done by running:
```
python train.py -c config -d path_to_dataset
```
using one of the config files provided in `train/configs`.

Additionally, you can use the checkpoints available in `train/checkpoints` to perform validation using 
```
python validate.py -c config -p pretrained_path
```

### Hardware
The deployment on TySOM-3A-ZU19EG is done using the FINN framework to generate the bitfile. See https://finn.readthedocs.io/en/latest/getting_started.html for installation and requirements.

We use a modified version of the FINN repository that can be obtained by executing the `get_finn.sh` script.
To build P-ENet_M, go in the finn folder and run
```
./run-docker.sh
```
Then within the docker environment go to the `hardware` directory and run:
```
python build.py -m models/p-enet_cityscapes.onnx -f configs/p-enet_m_config.json -o output
```

The generated outputs will be under the `output` directory.