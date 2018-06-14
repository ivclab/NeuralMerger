# NeuralMerger
Official implementation of [Unifying and Merging Well-trained Deep Neural Networks for Inference Stage](https://arxiv.org/abs/1805.04980).

## Usage

NeuralMerger
    ├─────── Fine-tuning
    └─────── Inference

Fine-tuning: Finetune the merged model of two well-trained neural networks (Tensorflow implementation).
Inference:   Test the speed of the merged model (C implementation).

1. Clone the NeuralMerger repository

   $ git clone --recursive https://github.com/ivclab/NeuralMerger.git

2. Follow the instruction in [finetuning](https://github.com/ivclab/NeuralMerger/tree/master/Fine-tuning) and get the well-trained merged model.

3. Test the merged model on Inference Code.


## Citation
Please cite following paper if these codes help your research:

    @Article{
      Title   = {Unifying and Merging Well-trained Deep Neural Networks for Inference Stage},
      Author  = {Chou, Yi-Min and Chan, Yi-Ming and Lee, Jia-Hong and Chiu, Chih-Yi and Chen, Chu-Song}, 
      Journal = {International Joint Conference on Artificial Intelligence, IJCAI-ECAI},
      year    = {2018}
    }

