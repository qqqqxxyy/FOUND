##### 12_11
1.跑通训练和测试的命令(包括储存weight,iou检测)
    [X] 训练
    [X] 测试
#!/bin/bash
cd ../
echo 'DATASET_DIR=/data/users/quxiangyu/datasets/'
export DATASET_DIR=/data/users/quxiangyu/datasets/

echo 'python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR'
python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR

python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR --model-weights outputs/FOUND-DUTS-TR-vit_small8/decoder_weights_niter200.pt


2.训练和测试数据集模块改voc12/cub,参数加载模块改成beta_CFN模式

3.将FOUND中tensorboard日志模块集成进beta_CFN之中
    将beta_CFN中的参数加载模块从基于json的配制成基于yaml的

##### 12_08
```bash
# Root directory of all datasets, both training and evaluation
export DATASET_DIR=/data/users/quxiangyu/datasets/

python main_found_train.py --dataset-dir $DATASET_DIR
```

##### 12_07
**training**
python main_found_train.py --dataset-dir $DATASET_DIR


##### 12_04
**evaluation**
python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir /data/users/quxiangyu/datasets/VOC2012



##### 12_01
image path:
/data/users/quxiangyu/datasets/ILSVRC/data/validation_data/ILSVRC2012_val_00001194.JPEG
python main_visualize.py --img-path /data/users/quxiangyu/datasets/ILSVRC/data/validation_data/ILSVRC2012_val_00001194.JPEG