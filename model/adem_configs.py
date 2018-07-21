import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
from layers.rnncells import StackedLSTMCell, StackedGRUCell
from configs import Config

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {
    'cornell': data_dir.joinpath('cornell'),
    'ubuntu': data_dir.joinpath('ubuntu'),
    'smart_ko': data_dir.joinpath('smart_ko'), 
    'smart_ko_adem': data_dir.joinpath('smart_ko_adem'), 
    'smart_ko_word': data_dir.joinpath('smart_ko_word')
}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}
username = Path.home().name
save_dir = Path(os.getcwd())


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AdemConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_path = self.data_dir.joinpath('score.pkl')


def get_adem_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--eval_batch_size', type=int, default=80)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Generation
    parser.add_argument('--max_unroll', type=int, default=30)
    parser.add_argument('--sample', type=str2bool, default=False,
                        help='if false, use beam search for decoding')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1)

    # Model
    parser.add_argument('--model', type=str, default='ADEM',
                        help='one of {HRED, VHRED, VHCR,}')
    # Currently does not support lstm
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=500)
    parser.add_argument('--tie_embedding', type=str2bool, default=True)
    parser.add_argument('--encoder_hidden_size', type=int, default=1000)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--decoder_hidden_size', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--context_size', type=int, default=1000)
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='Tanh')

    # VAE model
    parser.add_argument('--z_sent_size', type=int, default=100)
    parser.add_argument('--z_conv_size', type=int, default=100)
    parser.add_argument('--word_drop', type=float, default=0.0,
                        help='only applied to variational models')
    parser.add_argument('--kl_threshold', type=float, default=0.0)
    parser.add_argument('--kl_annealing_iter', type=int, default=25000)
    parser.add_argument('--importance_sample', type=int, default=100)
    parser.add_argument('--sentence_drop', type=float, default=0.0)

    # Generation
    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--n_sample_step', type=int, default=1)

    # BOW
    parser.add_argument('--bow', type=str2bool, default=False)

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    # Data
    parser.add_argument('--data', type=str, default='ubuntu')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return AdemConfig(**kwargs)
