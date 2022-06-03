# pytorch deeplabv3plus with DP and DDP
This repository is an implementation of deeplabv3+ with dp and ddp.

-- DP : DataParallel (DP). Implements data parallelism at the module level.

-- DDP : DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines.

#### Environment
∘ Ubuntu 20.04.4 LTS

∘ NVIDIA GeForce RTX 3090(24GB RAM) *2 

∘ 128GB RAM

<br/>

### Available Models
∘ (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)

∘ (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper]](https://arxiv.org/abs/1703.02719)

∘ (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper]](https://arxiv.org/abs/1807.10221)

∘ (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1702.08502) 

∘ (**PSPNet**) Pyramid Scene Parsing Network [[Paper]](http://jiaya.me/papers/PSPNet_cvpr17.pdf) 

∘ (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper]](https://arxiv.org/abs/1606.02147)

∘ (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015) [[Paper]](https://arxiv.org/abs/1505.04597)

∘ (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016) [[Paper]](https://arxiv.org/pdf/1511.00561)

∘ (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015) [[Paper]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 




<br/>
<br/>

* #### Creating a virtual environment
```shell
conda create -n semseg python=3.7
```  

* #### Install python libraries.
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```  
```shell
pip install -r requirements.txt
```  

* #### Prepare the dataset (1)
     <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>
```shell
$ bash wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ tar -xvf VOCtrainval_11-May-2012.tar
```

* #### Prepare the dataset (2)
     augment the dataset using the additionnal annotation [[Paper]](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf)

Download - (1) : [Additional Dataset 1](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=)

> Then add the files to the path below -> ``` voc2012_trainval/ImageSets/Segmentation ```

Download - (2) : [Additional Dataset 2](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)

> Then add the files to the path below -> ``` voc2012_trainval/SegmentationClassAug/ ```

<br/>
<br/>

* #### Train (DP)
     
```shell
python train_dp.py
```


* #### Train (DDP)

```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
```

     
















