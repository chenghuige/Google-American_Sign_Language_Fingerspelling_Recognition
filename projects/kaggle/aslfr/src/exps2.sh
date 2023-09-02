# sh ddp-run2.sh flags/encode --awp_train --adv_start_epoch=15 --adv_eps=0 --adv_lr=0.1 --ep=300 --mn=encode.torch.awp.ep-300
# sh ddp-run2.sh flags/encode --mix_sup=0 --lr=1e-4 --awp_train --adv_start_epoch=0 --adv_eps=0 --adv_lr=0.1  --pretrain=encode.torch.awp.ep-300 --mn=encode.torch.awp.ep-300.ft-100
sh ddp-run2.sh flags/encode --awp_train --adv_start_epoch=15 --adv_eps=0 --adv_lr=0.1  --ep=300 --mn=encode.torch.awp.ep-300 --online
sh ddp-run2.sh flags/encode --mix_sup=0 --lr=1e-4 --awp_train --adv_start_epoch=0 --pretrain=encode.torch.awp.ep-300  --mn=encode.torch.awp.ep-300.ft-100 --online
