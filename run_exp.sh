#!/usr/bin/env bash

python main.py --mode train --data_dir data_new --model pool --model_file model.pt --freeze True --seed 3 --max_epochs 2
python main.py --mode train --data_dir data_new --model pool --model_file model.pt --freeze True --seed 4 --max_epochs 2
python main.py --mode train --data_dir data_new --model pool --model_file model.pt --freeze True --seed 5 --max_epochs 2