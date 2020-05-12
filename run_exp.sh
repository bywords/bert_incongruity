#!/usr/bin/env bash

python main.py --mode train --data_dir data_nips_incon --model pool --model_file model.pt --freeze False --seed 3 --max_epochs 2
