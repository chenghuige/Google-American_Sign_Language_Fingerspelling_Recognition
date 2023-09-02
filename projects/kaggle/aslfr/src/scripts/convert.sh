./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --pretrain_online=0 \
  --tfrecords \
  --mn=$1.eval \
  --restore_configs \
  --save_final=0 \
  --gpus=1 \
  --force_convert \
  --steps=-1 \
  --cf \
  $*
