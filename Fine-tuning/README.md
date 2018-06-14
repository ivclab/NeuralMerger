# Fine-tuning
## Requirements
- Python 2.7
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow 1.4.0 or higher](https://github.com/tensorflow/tensorflow)

## Datasets
- Sound20 (https://github.com/ivclab/Sound20)
- MNIST-Fashion (https://github.com/zalandoresearch/fashion-mnist)
- Multi-View Clothing (https://github.com/MVC-Datasets/MVC)
- The OUI-Adience (https://www.openu.ac.il/home/hassner/Adience/publications.html)

## How to run
Download TFRecords Data and Well-trained model weight

    $ python download.py

To merge Lenet-Sound and Lenet-Fashion:
    
    
    # $ python finetuning.py --net=TASK_NAME --merger_dir=MERGED_MODEL_DIR
    $ python finetuning.py --net=lenetsound_lenetfashion --merger_dir='./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/' --batch_size=64
    

