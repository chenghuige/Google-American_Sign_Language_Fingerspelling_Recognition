sh ./run.sh flags/encode \
  --encoder=conv1d_transformer \
  --encoder_layers=1 \
  --n_frames=64 \
  --ep=5 \
  --mn=tiny \
  --force_convert \
  $*
