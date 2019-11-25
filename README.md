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

```bash
scp -P 9922 kaist@kdialogue.snu.ac.kr:~/incon_dataset/NELA_2018_more_info/* data/
```

## Train

```bash
python main.py --mode train --data_dir [DATA_DIR] --model pool --model_file model.pt --freeze True --exp_id bertpool_freezeTrue
```

* DATA_DIR: File names of each dataset should be equal to one another.