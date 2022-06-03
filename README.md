# deeplabv3plus_DPDDP
This repository is an implementation of deeplabv3+ with dp and ddp.

-- DP : DataParallel (DP). Implements data parallelism at the module level.

-- DDP : DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines.

#### Environment
∘ Ubuntu 20.04.4 LTS

∘ NVIDIA GeForce RTX 3090(24GB RAM) *2 

∘ 128GB RAM


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
$ tar -xvf VOCtrainval_11-May-2012.tar -C /content/data
```








