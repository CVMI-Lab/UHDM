# Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing

**Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing** (ECCV 2022)  
Xin Yu, Peng Dai, Wenbo Li, Lan Ma, Jiajun Shen, Jia Li, [Xiaojuan Qi](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en).
<br>[Paper(Coming soon)](---), [Project Page](https://xinyu-andy.github.io/uhdm-page/), [Dataset](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=sharing)

![Example 1](./figures/result.png)

## Introduction
When photographing the contents displayed on the digital screen, an inevitable frequency aliasing between the camera’s 
color filter array (CFA) and the screen’s LCD subpixel widely exists. The captured images are thus mixed with colorful 
stripes, named moire patterns, which severely degrade images’ perceptual qualities. Although a plethora of dedicated 
demoire methods have been proposed in the research community recently, yet is still far from achieving promising results 
in the real-world scenes. The key limitation of these methods is that they all only conduct research on low-resolution or 
synthetic images. However, with the rapid development of mobile devices, modern widely-used mobile phones typically allow 
users to capture 4K resolution (i.e.,ultra-high-definition) images, and thus the effectiveness of these methods on this
practical scenes is not promised. In this work, we explore moire pattern removal for ultra-high-definition images. 
First, we propose the first ultra-high-definition demoireing dataset (UHDM), which contains 5,000 real-world 4K 
resolution image pair, and conduct a benchmark study on the current state of the art. Then, we analyze limitations 
of the state of the art and summarize the key issue of them, i.e., not scale-robust. To address their deficiencies, 
we deliver a plug-and-play semantic-aligned scale-aware module which helps us to build a frustratingly simple baseline 
model for tackling 4K moire images. Our framework is easy to implement and fast for inference, achieving state-of-the-art 
results on four demoireing datasets while being much more lightweight. 
We hope our investigation could inspire more future research in this more practical setting in image demoireing.


<img src="./figures/cost.png" width="55%" ></img>


## Environments

First you have to make sure that you have installed all dependencies. To do so, you can create an anaconda environment called `esdnet` using

```
conda env create -f environment.yaml
conda activate esdnet
```

Our implementation has been tested on one NVIDIA 3090 GPU with cuda 11.2.

## Dataset
![Data](./figures/dataset.png)
We provide the 4K dataset UHDM for you to evaluate a pretrained model or train a new model.
To this end, you can download them [here](https://drive.google.com/drive/folders/1DyA84UqM7zf3CeoEBNmTi_dJ649x2e7e?usp=sharing), 
or you can simply run the following command for automatic data downloading:
```
bash scripts/download_data.sh
```
Then the dataset will be available in the folder `uhdm_data/`.

## Train
To train a model from scratch, simply run:

```
python train.py --config CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.
We have included configuration files for each dataset under the folder `config/`.

For example, if you want to train our lightweight model ESDNet on UHDM dataset, run:
```
python train.py --config ./config/uhdm_config.yaml
```
   

## Test
To test a model, you can also simply run:

```
python test.py --config CONFIG.yaml
```

where you need to specify the value of `TEST_EPOCH` in the `CONFIG.yaml` to evaluate a model trained after specific epochs, 
or you can also specify the value of `LOAD_PATH` to directly load a pre-trained checkpoint.

We provide pre-trained models [here](https://drive.google.com/drive/folders/12buOOBKDBdQ65gM8U1rRNpSHppQ_u9Lr?usp=sharing). 
To download the checkpoints, you can also simply run:

```
bash scripts/download_model.sh
```

Then the checkpoints will be included in the folder `pretrain_model/`. 


## Contact
If you have any questions, you can email me (yuxin27g@gmail.com).


