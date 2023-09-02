for (( i=0; i<5; i++ ))
do
  ddp ./main.py --flagfile=$1 --fold=$i $*
done
