import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import PAD_ID, UNK_ID, SOS_ID, EOS_ID
from data_loader import DialogDataset
import numpy as np

class DialogEvalDataset(DialogDataset):
    '''
    Dataset class for ADEM training
    Added gold response and score
    
    sentences[0:-2] : Contexts
    sentences[-2] : Gold response
    sentences[-1] : Predicted response
    '''
    def __init__(self, sentences, conversation_length, sentence_length, score, vocab, data=None):
        super().__init__(sentences, conversation_length, sentence_length, vocab, data)
        
        self.score = score

    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        sentence = self.sentences[index]
        conversation_length = self.conversation_length[index]
        sentence_length = self.sentence_length[index]
        score = self.score[index]

        # word => word_ids
        sentence = self.sent2id(sentence)

        return sentence, conversation_length, sentence_length, score


def get_adem_loader(sentences, conversation_length, sentence_length, score, vocab, batch_size=100, data=None, shuffle=True):
    """Load DataLoader of given DialogEvalDataset"""
    print("adem_data_loader line:38 adem loader")
    
    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)

        # Separate
        sentences, conversation_length, sentence_length, score = zip(*data)

        # return sentences, conversation_length, sentence_length.tolist()
        return sentences, conversation_length, sentence_length, score

    adem_dataset = DialogEvalDataset(sentences, conversation_length, sentence_length, score, vocab, data=data)
#     print("adem_data_loader line:63", score)
#     input("adem_data_loader line:64")

    data_adem_loader = DataLoader(
        dataset=adem_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_adem_loader
