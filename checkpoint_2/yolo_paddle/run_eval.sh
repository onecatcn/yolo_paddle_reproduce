export PYTHONPATH=.

python metric_eval.py  \
    --dataset_dir /home/aistudio/VOCdevkit/ \
    --trained_model ../../weights_trans/yolo_paddle.pdparams \
    --gpu