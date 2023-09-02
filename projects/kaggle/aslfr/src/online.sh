sh run.sh flags/encode1 --ep=300 --exit_epoch=15 --mn=encode1.awp.0807 --online $*
sh run.sh flags/encode1 --ep=300 --awp_train --adv_start_epoch=0 --adv_lr=0.1 --pretrain=encode1.awp.0807 --pretrain_restart=0 --pretrain_online --mn=encode1.awp.ft.0807 --online $*
sh run.sh flags/encode1 --mix_sup=0 --awp_train --adv_start_epoch=0 --adv_lr=0.1 --lr=1e-4 --ep=100  --pretrain=encode1.awp.ft.0807 --pretrain_restart --pretrain_online --mn=encode1.awp.ft2.0807 --online $*
