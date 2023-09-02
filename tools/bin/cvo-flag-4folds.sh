for (( i=0; i<4; i++ ))
do
  ddp ./main.py --flagfile=$1 --fold=$i  $*
done
ddp ./main.py --flagfile=$1 --online $*
