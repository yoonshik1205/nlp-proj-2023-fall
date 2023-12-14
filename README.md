# 2023 Fall NLP course project

### Please set the dataset and run the training code following the instruction in default_README file.

### The code is heavily based on [AttnGAN](https://github.com/taoxugit/AttnGAN) official github code.

### To train and evaluate, please check cfg files with proper data path and model weights.

### The training is consisting of two steps: 
1. Pretraining DAMSM module
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
```
2. Fully training AttnGAN
```
python main.py --cfg cfg/bird_attn2.yml --gpu 0
```

### To evaluate
```
python main.py —cfg cfg/eval_bird.yml —gpu 0
```
