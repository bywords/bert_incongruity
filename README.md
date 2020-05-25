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
- The names of datasets should be train/dev/test.tsv

### Create sample dataset for test purposes

```bash
shuf -n 1000 train.tsv > train_sample.tsv
```


## Commands for Run

### Train (BDE)

```bash
python main_bde.py --mode train --data_dir [DATA_DIR]
```


### Test (BDE)

```bash
python main_bde.py --mode test --data_dir [DATA_DIR]
```

### Inference on real-world articles (BDE)

```bash
python main_bde.py --mode [MODE] --data_dir [DATA_DIR]
```

- MODE: real_old, real_new, real_covid

### Test (Bert-NSP)

```bash
python main_nsp_freeze.py --mode test --data_dir [DATA_DIR]
```


### Inference on real-world articles (Bert-NSP)

```bash
python main_nsp_freeze.py --mode [MODE] --data_dir [DATA_DIR]
```

- MODE: real_old, real_new, real_covid