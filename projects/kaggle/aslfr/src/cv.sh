for (( i=0; i<4; i++ ))
do
  py main.py --flagfile=$1 --fold=$i $*
done
