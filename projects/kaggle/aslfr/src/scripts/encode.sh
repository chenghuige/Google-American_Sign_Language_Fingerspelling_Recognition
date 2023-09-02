sh ./run.sh $1 \
  --model=model2 \
  --method=encode \
  --loss=ctc \
  --encode_pool_size=2 \
  $*
