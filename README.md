# NeuralMerger
Official implementation of [Unifying and Merging Well-trained Deep Neural Networks for Inference Stage](https://arxiv.org/abs/1805.04980).

## Usage
**Fine-tuning**: Finetune the merged model of two well-trained neural networks (Tensorflow implementation).

**Inference**: Test the speed of the merged model (C implementation).

    NeuralMerger
        ├─────── Fine-tuning
        └─────── Inference


1.Clone the NeuralMerger repository:

    $ git clone --recursive https://github.com/ivclab/NeuralMerger.git


2.Follow the instruction in [Fine-tuning](https://github.com/ivclab/NeuralMerger/tree/master/Fine-tuning) and get the well-trained merged model.
  

3.Test the well-trained merged model on [Inference](https://github.com/ivclab/NeuralMerger/tree/master/Inference).


## Citation
Please cite following paper if these codes help your research:

    @inproceedings{
      Title   = {Unifying and Merging Well-trained Deep Neural Networks for Inference Stage},
      Author  = {Chou, Yi-Min and Chan, Yi-Ming and Lee, Jia-Hong and Chiu, Chih-Yi and Chen, Chu-Song}, 
      booktitle = {International Joint Conference on Artificial Intelligence, IJCAI-ECAI},
      year    = {2018}
    }

