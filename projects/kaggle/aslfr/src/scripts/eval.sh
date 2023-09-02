./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --pretrain_online=0 \
  --tfrecords \
  --mn=$1.eval \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  --wandb=0 \
  --gpus=1 \
  $*
