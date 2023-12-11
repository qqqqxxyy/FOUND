#!/bin/bash
cd ../
echo 'DATASET_DIR=/data/users/quxiangyu/datasets/'
export DATASET_DIR=/data/users/quxiangyu/datasets/

echo 'python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR'
python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR
