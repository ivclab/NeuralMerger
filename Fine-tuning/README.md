# Fine-tuning
## Prerequisition
- Python 2.7
- [TensorFlow 1.4.0 or higher](https://github.com/tensorflow/tensorflow)

## Datasets
- Sound20 (https://github.com/ivclab/Sound20)
- MNIST-Fashion (https://github.com/zalandoresearch/fashion-mnist)
- Multi-View Clothing (https://github.com/MVC-Datasets/MVC)
- The OUI-Adience (https://www.openu.ac.il/home/hassner/Adience/publications.html)

## How to Run
- Install required packages:
```bash
$ pip install -r requirements.txt
```

- Download TFRecords Data and Well-trained Neural Network weight:
```bash
$ python download.py
```

- Fine-tune the merged model of **Lenet-Sound** and **Lenet-Fashion**:
```bash
# $ python finetuning.py --net=TASK_NAME --merger_dir=MERGED_MODEL_DIR
$ python finetuning.py --net=lenetsound_lenetfashion --merger_dir=./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/ --batch_size=64  --save_model=True
```

- Fine-tune the merged model of **VGGAvg-Clothing** and **ZF-Gender**:
```bash
$ python finetuning.py --net=vggclothing_zfgender --merger_dir=./weight_loader/weight/vggclothing_zfgender/merge_ACCU/  --save_model=True --lr_rate=0.00006 --batch_size=16
```

- The merged model (after fine-tuning) will be saved in `./logs/FOLDER_NAME/`

## Experimental Results 

The experimental results (mean values over five repetition) of mergning **Lenet-Sound** and **Lenet-Fashion** :

Network Model | Sound Accuracy | Fashion Accuracy | Model Size 
----------------- | ---------------- | ---------------- | ----------------
Lenet-Sound     |**78.08%** | -         | 17.1 MB
Lenet-Fashion   |  -        |**91.57%** | 17.0 MB
Merged ACCU     |  78.06%   | 91.08%    | 3.3 MB
Merged LIGHT    |  77.66%   | 90.89%    | 2.3 MB

The experimental results (mean values over five repetition) of mergning **VGGAvg-Clothing** and **ZF-Gender** :

Network Model | Clothing Accuracy | Gender Accuracy | Model Size 
----------------- | ---------------- | ---------------- | ----------------
VGG-Clothing     |89.97% | -         | 134.7 MB
ZF-Gender   |  -        |**83.43%** | 233.1 MB
Merged ACCU     |  **90.69%**   | 82.71%    | 30.4 MB
Merged LIGHT    |  90.02%   | 81.53%    | 18.3 MB


    
    

