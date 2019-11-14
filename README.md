# Detecting incongruity between headline and body text using BERT

## Dependencies
```
python==3.6
torch==1.1.0
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
python main.py --mode train --model_file model.pt --output_dir model
```