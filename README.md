### NeuralMerger
Tensorflow implementation of [Unifying and Merging Well-trained Deep Neural Networks for Inference Stage](https://arxiv.org/abs/1805.04980).

## Requirements

- Python 2.7 or 3.x
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow 1.4.0 or higher](https://github.com/tensorflow/tensorflow)

## Datasets
Sound20 (https://github.com/ivclab/Sound20)
MNIST-Fashion (https://github.com/zalandoresearch/fashion-mnist)
Multi-View Clothing (https://github.com/MVC-Datasets/MVC)
The OUI-Adience (https://www.openu.ac.il/home/hassner/Adience/publications.html)

## How to run
Download TFRecords Data and Well-trained model weight

    $ python download.py

To merge Lenet-Sound and Lenet-Fashion (experiment 1 in paper):

    $ python main.py --net=lenetsound_lenetfashion --merger_dir='./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/' --batch_size=64

To merge VGG-Clothing and ZF-Gender (experiment 2 in paper):

    $ python main.py --net=vggclothing_zfgender --merger_dir='./weight_loader/weight/vggclothing_zfgender/merge_ACCU/' --batch_size=16






