#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

feat="" # feats.scp
unit=""
remove_space=false
unk="<unk>"
space="<space>"
nlsyms=""
wp_model=""
wp_nbest=1
text=

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data_dir> <dict>";
    exit 1;
fi

data=$1
dict=$2

if [ -z ${text} ]; then
    text=${data}/text
fi

make_tsv.py --feat ${feat} \
    --utt2num_frames ${data}/utt2num_frames \
    --utt2spk ${data}/utt2spk \
    --text ${text} \
    --dict ${dict} \
    --unit ${unit} \
    --remove_space ${remove_space} \
    --unk ${unk} \
    --space ${space} \
    --nlsyms ${nlsyms} \
    --wp_model ${wp_model} \
    --wp_nbest ${wp_nbest}
