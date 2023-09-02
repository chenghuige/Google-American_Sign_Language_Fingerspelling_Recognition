sh ddp-run2.sh $1 --exit_epoch=15 $*
sh ddp-run2.sh $1 --awp_train --adv_start_epoch=0 --adv_lr=0.1 --adv_eps=0 $*
