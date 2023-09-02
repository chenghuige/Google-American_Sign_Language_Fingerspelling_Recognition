sh ./run.sh flags/encode \
  --encoder_layers=1 \
  --encoder_units=128 \
  --n_frames=64 \
  --cls_loss=type \
  --small \
  --use_aug=0 \
  --mix_sup=0 \ 
  --cls_pooling=max \
 --seed=123 \
  --mn=tiny \
  $*
