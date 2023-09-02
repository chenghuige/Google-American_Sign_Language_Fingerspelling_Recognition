# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args options for the LM task."""

import configargparse
from distutils.util import strtobool
import logging
from omegaconf import OmegaConf
import os

from neural_sp.bin.train_utils import load_config
from neural_sp.bin.args_common import add_args_common

logger = logging.getLogger(__name__)


def parse_args_train(input_args):
    parser = build_parser()
    user_args, _ = parser.parse_known_args(input_args)

    config = OmegaConf.load(user_args.config)

    # register module specific arguments
    parser = register_args_lm(parser, user_args, user_args.lm_type)
    user_args = parser.parse_args()

    # merge to omegaconf
    for k, v in vars(user_args).items():
        if k not in config:
            config[k] = v

    return config


def parse_args_eval(input_args):
    parser = build_parser()
    user_args, _ = parser.parse_known_args(input_args)

    # Load a yaml config file
    dir_name = os.path.dirname(user_args.recog_model[0])
    config = load_config(os.path.join(dir_name, 'conf.yml'))

    # register module specific arguments to support new args after training
    parser = register_args_lm(parser, user_args, config.lm_type)
    user_args = parser.parse_args()

    # Overwrite to omegaconf
    for k, v in vars(user_args).items():
        if 'recog' in k or k not in config:
            config[k] = v
            logger.info('Overwrite configration: %s => %s' % (k, v))

    return config, dir_name


def register_args_lm(parser, args, lm_type):
    if 'gated_conv' in lm_type:
        from neural_sp.models.lm.gated_convlm import GatedConvLM as module
    elif lm_type == 'transformer':
        from neural_sp.models.lm.transformerlm import TransformerLM as module
    elif lm_type == 'transformer_xl':
        from neural_sp.models.lm.transformer_xl import TransformerXL as module
    else:
        from neural_sp.models.lm.rnnlm import RNNLM as module
    if hasattr(module, 'add_args'):
        parser = module.add_args(parser, args)
    return parser


def build_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser = add_args_common(parser)
    # features
    parser.add_argument('--min_n_tokens', type=int, default=1,
                        help='minimum number of input tokens')
    parser.add_argument('--dynamic_batching', type=strtobool, default=False,
                        help='')
    # topology
    parser.add_argument('--lm_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'gated_conv_custom',
                                 'gated_conv_8', 'gated_conv_8B', 'gated_conv_9',
                                 'gated_conv_13', 'gated_conv_14', 'gated_conv_14B',
                                 'transformer', 'transformer_xl'],
                        help='type of language model')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='number of dimensions in the embedding layer')
    parser.add_argument('--n_units_null_context', type=int, default=0, nargs='?',
                        help='')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='tie input and output embedding')
    # optimization
    parser.add_argument('--bptt', type=int, default=200,
                        help='BPTT length')
    # initialization
    parser.add_argument('--pretrained_model', type=str, default=False, nargs='?',
                        help='')
    # regularization
    parser.add_argument('--dropout_in', type=float, default=0.0,
                        help='dropout probability for the input embedding layer')
    parser.add_argument('--dropout_hidden', type=float, default=0.0,
                        help='dropout probability for the hidden layers')
    parser.add_argument('--dropout_out', type=float, default=0.0,
                        help='dropout probability for the output layer')
    parser.add_argument('--logits_temp', type=float, default=1.0,
                        help='')
    parser.add_argument('--backward', type=strtobool, default=False, nargs='?',
                        help='')
    parser.add_argument('--adaptive_softmax', type=strtobool, default=False,
                        help='use adaptive softmax')
    # contextualization
    parser.add_argument('--shuffle', type=strtobool, default=False, nargs='?',
                        help='shuffle utterances per epoch')
    parser.add_argument('--serialize', type=strtobool, default=False, nargs='?',
                        help='serialize text according to onset in dialogue')
    # evaluation parameters
    parser.add_argument('--recog_n_caches', type=int, default=0,
                        help='number of tokens for cache')
    parser.add_argument('--recog_cache_theta', type=float, default=0.2,
                        help='theta parameter for cache')
    parser.add_argument('--recog_cache_lambda', type=float, default=0.2,
                        help='lambda parameter for cache')
    parser.add_argument('--recog_mem_len', type=int, default=0,
                        help='number of tokens for memory in TransformerXL during evaluation')
    return parser
