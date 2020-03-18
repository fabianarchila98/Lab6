# Lab 6 - HED

Today's lab will be based on this repository: https://github.com/buntyke/pytorch-hed. 

Pytorch implementation of [Holistically-nested Edge Detection (HED)][1]. This repo can be used to train a HED model and perform test inference. The implementation was evaluated on the BSDS dataset.

<p align="center"><img src="https://carlosjuliopardoblog.files.wordpress.com/2019/04/image.png" width=500"/></p>

* [Usage](#usage)
* [Files](#files)
* [Acknoledgement](#acknowledgement)
* [References](#references)

## Usage

* Create `data` folder inside the repository, download and extract BSDS dataset into folder:
  ```
  $ mkdir data; cd data
  $ wget http://vcl.ucsd.edu/hed/HED-BSDS.tar
  $ tar -xvf HED-BSDS.tar
  $ rm HED-BSDS.tar
  $ cd HED-BSDS/
  $ head -n 10 train_pair.lst > val_pair.lst
  $ cd ../../
  ```
* Download the VGG pretrained model to initialize training
  ```
  $ mkdir model; cd model/
  $ wget https://download.pytorch.org/models/vgg16-397923af.pth
  $ mv vgg16-397923af.pth vgg16.pth
  $ cd ..
  ```
* Train HED model by running `train.py:
  ```
  $ python train.py 
  ```
  The trained model along with validation results are stored in the train folder.

## Files

* [train.py](train.py): Script to train HED model.
* [trainer.py](trainer.py): Helper class to train model and perform validation.
* [model.py](model.py): HED model definition given through several class implementations.
* [dataproc.py](dataproc.py): Dataset class implementation used in Trainer class.
* [test.py](test.py): Script to test HED model.

## Evaluation

To evaluate the method you will use the official BSDS evaluation metrics. To do this, you'll need to download the complete BSDS500 dataset (not the one downloaded before) and this repository: 

```
git clone https://github.com/s-gupta/py-bsds500
cd py-bsds500
python setup.py build_ext --inplace
python verify.py .
```

## Deadline

March 18, 23:59pm

## Acknowledgement

The source code is derived from three different implementations available online. Thanks to [@s9xie][2] for original caffe implementation. Thanks to [@EliasVansteenkiste][3], [@xlliu][4], [@BinWang-shu][5] for the pytorch implementations.

[1]: https://arxiv.org/abs/1504.06375 "HED"

[2]: https://github.com/s9xie/hed "Caffe"

[3]: https://github.com/EliasVansteenkiste/edge_detection_framework "Pytorch 1"

[4]: https://github.com/xlliu7/hed.pytorch "Pytorch 2"

[5]: https://github.com/BinWang-shu/pytorch_hed "Pytorch 3"
