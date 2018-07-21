import torch
import torch.nn as nn
from collections import OrderedDict

import layers as layers

class ADEM(nn.Module):
    def __init__(self, config):
        super(ADEM, self).__init__()
        
        self.num_layers = config.num_layers
        self.rnn = config.rnn
        self.bidirectional = config.bidirectional
        self.dropout = config.dropout
        
        
        # Encoder
        self.vocab_size = config.vocab_size
        self.embed_size = config.embedding_size
        self.hidden_size = config.encoder_hidden_size
        self.bidirectional = config.bidirectional
        
        self.encoder = layers.EncoderRNN(self.vocab_size,
                                         self.embed_size,
                                         self.hidden_size,
                                         self.rnn,
                                         self.num_layers,
                                         self.bidirectional,
                                         self.dropout)
        
        # Context Encoder
        self.context_size = config.context_size

        context_input_size = (self.num_layers
                              * self.hidden_size
                              * self.encoder.num_directions)
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 self.context_size,
                                                 self.rnn,
                                                 self.num_layers,
                                                 self.dropout)
    
    def forward(self, sentences, sentence_length,
                input_conversation_length, target_sentences, decode=False):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(sentences,
                                                       sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences + batch_size, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        # encoder_hidden: [batch_size, max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # encoder_hidden_inference: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [encoder_hidden_inference[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        # encoder_hidden_input: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # context_outputs: [batch_size, max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden_input,
                                                                    input_conversation_length)
        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])
        
        return None
        
    def load_pretrained_model(self, model_path):
        vhred_state = torch.load(model_path)
        
        encoder_state = OrderedDict()
        for k, v in vhred_state.items():
            if k.find('encoder') != -1:
                encoder_state[k] = v
                
        self.load_state_dict(encoder_state)