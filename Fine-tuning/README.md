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

- Fine-tune the merged model of **VGG-Clothing** and **ZF-Gender**:
```bash
$ python finetuning.py --net=vggclothing_zfgender --merger_dir=./weight_loader/weight/vggclothing_zfgender/merge_ACCU/  --save_model=True --lr_rate=0.00006 --batch_size=16
```

- The merged model (after fine-tuning) will be saved in `./logs/FOLDER_NAME/`




    
    

