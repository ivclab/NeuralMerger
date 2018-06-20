# Fine-tuning
## Requirements
- Python 2.7
- [requests](https://github.com/requests/requests)
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow 1.4.0 or higher](https://github.com/tensorflow/tensorflow)

## Datasets
- Sound20 (https://github.com/ivclab/Sound20)
- MNIST-Fashion (https://github.com/zalandoresearch/fashion-mnist)
- Multi-View Clothing (https://github.com/MVC-Datasets/MVC)
- The OUI-Adience (https://www.openu.ac.il/home/hassner/Adience/publications.html)

## How to run
- Download TFRecords Data and Well-trained Neural Network weight:

```bash
$ python download.py
```

- To fine-tune the merged model of **Lenet-Sound** and **Lenet-Fashion**:

```bash
# $ python finetuning.py --net=TASK_NAME --merger_dir=MERGED_MODEL_DIR
$ python finetuning.py --net=lenetsound_lenetfashion --merger_dir=./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/ --batch_size=64  --save_model=True
```

- The merged model (after fine-tuning) will be saved in `./logs/FOLDER_NAME/`

## Experimental results 

The experimental results (mean values over five repetition) of mergning **Lenet-Sound** and **Lenet-Fashion** :

Network Model | Sound Accuracy | Fashion Accuracy | Model Size 
----------------- | ---------------- | ---------------- | ----------------
Lenet-Sound     |**78.08%** | -         | 17.1 MB
Lenet-Fashion   |  -        |**91.57%** | 17.0 MB
Merged ACCU     |  78.06%   | 91.08%    | 3.3 MB
Merged LIGHT    |  77.66%   | 90.89%    | 2.3 MB


    
    

