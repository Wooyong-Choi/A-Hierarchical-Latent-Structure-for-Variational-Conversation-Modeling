from solver import Solver, VariationalSolver, AdemSolver
from data_loader import get_loader
from adem_data_loader import get_adem_loader
from configs import get_config
from utils import Vocab, Tokenizer
import os
import pickle
from models import VariationalModels


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='test')

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    
    if config.model == "ADEM":
        data_loader = get_adem_loader(
            sentences=load_pickle(config.sentences_path),
            conversation_length=load_pickle(config.conversation_length_path),
            sentence_length=load_pickle(config.sentence_length_path),
            score=load_pickle(config.score_path),
            vocab=vocab,
            batch_size=config.batch_size,
            shuffle=False)
        
    else:    
        data_loader = get_loader(
            sentences=load_pickle(config.sentences_path),
            conversation_length=load_pickle(config.conversation_length_path),
            sentence_length=load_pickle(config.sentence_length_path),
            vocab=vocab,
            batch_size=config.batch_size,
            shuffle=False)
  

    if config.model in VariationalModels:
        if config.model == "ADEM":
            print("train line:69 config.model", config.model)
            solver = AdemSolver(config, None, data_loader, vocab=vocab, is_train=False)
        else:
            solver = VariationalSolver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.importance_sample()
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        solver.test()
