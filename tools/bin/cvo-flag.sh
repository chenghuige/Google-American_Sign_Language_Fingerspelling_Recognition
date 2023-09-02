for (( i=0; i<5; i++ ))
do
  ddp ./main.py --allnew --flagfile=$1 --fold=$i  $*
done
ddp ./main.py --allnew --flagfile=$1 --online $*
