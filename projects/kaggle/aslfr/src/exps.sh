# sh ddp-run.sh flags/encode --ep=300 --mn=encode.torch.ep-300
# sh ddp-run.sh flags/encode --mix_sup=0 --lr=1e-4 --pretrain=encode.torch.ep-300  --mn=encode.torch.ep-300.ft-100
# sh ddp-run.sh flags/encode --mix_sup=0 --lr=1e-4 --awp_train --adv_start_epoch=0 --adv_lr=0.1 --adv_eps=0 --pretrain=encode.torch.ep-300 --mn=encode.torch.ep-300.ft-awp-100 

sh ddp-run2.sh flags/encode3 --ep=300 --mn=encode3.ep-300 --online
#sh ddp-run2.sh flags/encode --mix_sup=0 --lr=1e-4 --pretrain=encode.torch.ep-300  --mn=encode.torch.ep-300.ft-100 --online
sh ddp-run2.sh flags/encode3 --mix_sup=0 --lr=1e-4 --awp_train --adv_start_epoch=0 --adv_lr=0.1 --adv_eps=0 --pretrain=encode3.ep-300 --mn=encode3.ep-300.ft-awp-100 --online
