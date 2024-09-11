#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:MLV_IR_OD
echo $PYTHONPATH

python utils/converter.py \
--label_dir datasets/hscai \
--save_dir datasets/hscai