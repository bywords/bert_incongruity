# Detecting incongruity between headline and body text using BERT

## Dependencies
```
python==3.6
torch==1.2.0
```

If the ``Library not loaded'' error occurs in Mac OS, run the command below:
```
brew install libomp
```

## Dataset download

### NELA 2018 ver.

```bash
scp -P 9922 kaist@kdialogue.snu.ac.kr:~/incon_dataset/NELA_2018_more_info/* data/
```

### NIPS ver.

```bash
scp -P 9922 kaist@kdialogue.snu.ac.kr:~/incon_dataset/headline_swap_news/train/* data_nips_incon/
```

- Be cautious that the dataset size is huge.
- The names of datasets should be train/val/test.tsv

### Create sample dataset for test purposes

```bash
shuf -n 1000 train.tsv > train_sample.tsv
```


## Train

```bash
python main.py --mode train --data_dir [DATA_DIR] --model pool --model_file model.pt --freeze True
```



