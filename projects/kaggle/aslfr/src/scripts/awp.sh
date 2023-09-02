sh run.sh $1 --exit_epoch=15 $*
sh run.sh $1 --awp_train --adv_start_epoch=0 --adv_lr=0.1 $*
