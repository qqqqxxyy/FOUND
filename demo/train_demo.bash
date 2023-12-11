#!/bin/bash
cd ../
echo 'DATASET_DIR=/data/users/quxiangyu/datasets/'
export DATASET_DIR=/data/users/quxiangyu/datasets/
echo 'python main_found_train.py --dataset-dir $DATASET_DIR'
python main_found_train.py --dataset-dir $DATASET_DIR
