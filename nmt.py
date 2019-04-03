# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] --vocab=<file> MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] --vocab=<file> MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --load-weights-from=<file>              previously saved model for loading the weights [default : ""] 
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --n_layers=<int>                        number of layers in encoder LSTM [default: 3]
    --optim=<int>                           type of optimizer, 0 for Adam, 1 for SGD [default: 0]
    --tie-weights=<int>                     weight tying b/w encoder embedding weights and projection layer [default: 0]
    --mha=<int>                             set to 1 for using multi-head attention [default: 0]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --use-pte=<int>                         whether to use pre-trained embeddings or not [default: 0]
    --emb-dir=<file>                        pre-trained embeddings for the required languages
    --save-emb-as=<file>                    name of file for saving pre-trained embeddings
"""

import math
import pickle
import sys
import time
from collections import namedtuple
import os
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, get_batch_tensor, get_mask, to_variable, get_pre_trained_embeddings, to_tensor
from vocab import Vocab, VocabEntry
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, vocab, dropout_rate=0.2, tie_weights=0, mha=0, pte=None):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # initialize neural network layers...
        self.encoder = Encoder(embed_size, hidden_size, n_layers, len(vocab.src), dropout_rate=dropout_rate, pte=pte)
        self.decoder = Decoder(embed_size, hidden_size, n_layers, len(vocab.tgt), dropout_rate=dropout_rate,
                               tie_weights=tie_weights, mha=mha)

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], input_len: List[int],
                 output_len: List[int],
                 max_decoding_time_step=None, beam_size=5):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        keys, values, last_hiddens = self.encoder(src_sents, input_len)
        if tgt_sents is None:
            # During decoding(testing)
            return self.decoder.decode(keys, values, last_hiddens, max_decoding_time_step, beamSize=beam_size)
        # During training
        scores = self.decoder(keys, values, last_hiddens, tgt_sents, output_len, input_len)
        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        self.eval()
        input_sentence = get_batch_tensor([src_sent], len(src_sent), self.vocab.src)
        # Following is going to call the encoder followed by the beam decoder
        # The beam decoder returns a list of topk best sentences and their scores.
        all_hypotheses = []
        hypotheses = self(input_sentence, None, [len(src_sent)], None, max_decoding_time_step, beam_size)
        # assert len(outputs) == len(scores)
        for i in range(len(hypotheses)):
            sentence = []
            for x in hypotheses[i][0][1:-1]:
                # when unk token is encountered
                if x[0] == 3 and x[1] < len(src_sent):
                    sentence.append(src_sent[x[1]])
                else:
                    sentence.append(self.vocab.tgt.id2word[x[0]])
            all_hypotheses.append(Hypothesis(sentence, hypotheses[i][1]))
        return all_hypotheses

    def evaluate_ppl(self, dev_data, loss_fn, batch_size: int = 32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.
        self.eval()
        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            input_lens = [len(x) for x in src_sents]
            output_lens = [len(x) for x in tgt_sents]
            src_batch = get_batch_tensor(src_sents, input_lens[0], self.vocab.src)
            tgt_batch = get_batch_tensor(tgt_sents, max(output_lens), self.vocab.tgt)
            decoded_scores = self(src_batch, tgt_batch, input_lens, output_lens)

            label_mask = get_mask(output_lens).squeeze(1)[:, 1:].contiguous()

            # loss = loss_fn(decoded_scores, label)
            loss = loss_fn(decoded_scores.contiguous().view(-1, len(self.vocab.tgt)),
                           tgt_batch[:, 1:].contiguous().view(-1))

            loss = (loss.view(label_mask.size()) * label_mask).sum(1).sum()

            cum_loss += loss.data.cpu().numpy()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    def load(self, model_path: str):
        """
        Load a pre-trained model

        """
        self.load_state_dict(torch.load(model_path))

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self.state_dict(), path)


class Encoder(nn.Module):
    def __init__(self, embedding_dimension, hidden_dimension, n_layers, src_vocab_size, dropout_rate, pte):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_dimension
        self.n_layers = n_layers
        self.embedding_dim = embedding_dimension
        self.embed = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embedding_dimension)
        if pte is None:
            # Xavier initialization
            nn.init.xavier_uniform(self.embed.weight)
        else:
            # Initialize embedding weights with appropriate pre-trained word embeddings
            self.embed.weight.data.copy_(torch.from_numpy(pte))
        self.lstms = nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_dimension, num_layers=n_layers,
                             bidirectional=True, dropout=dropout_rate)
        self.linear_key = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension)
        self.linear_values = nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension)
        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_init = nn.Linear(2 * hidden_dimension, hidden_dimension)

    def forward(self, input, input_len):
        h = self.embed(input).permute(1, 0, 2)  # seq_len * bs * embedding_dim
        if (self.embed.weight != self.embed.weight).any():
            print("Found NaN in embedding weights")
            exit()
        packed_h = pack_padded_sequence(h, input_len)
        h, (last_state, last_cell) = self.lstms(packed_h)
        h, _ = pad_packed_sequence(h)  # seq_len * bs * (2 * hidden_dim)
        # Averaging forward and backward representation
        h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2
        h = h.permute(1, 0, 2)  # bs * seq_len * hidden_dim
        keys = self.linear_key(h)  # bs * seq_len * hidden_dim
        values = self.linear_values(h)  # bs * seq_len * hidden_dim
        dec_init_cells = []
        dec_init_states = []
        for j in range(self.n_layers):
            idx = j * 2
            dec_init_cells.append(self.decoder_init(torch.cat([last_cell[idx], last_cell[idx + 1]], 1)))
            dec_init_states.append(F.tanh(dec_init_cells[j]))
        return keys, values, (dec_init_states, dec_init_cells)


class MyLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__(input_size, hidden_size)

        # Adding initial state as learn-able parameters
        self.h0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(1, hidden_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, h, hx, cx):
        return super(MyLSTMCell, self).forward(h, (hx, cx))


class Decoder(nn.Module):
    def __init__(self, embedding_dimension, hidden_dimension, n_layers, output_size, dropout_rate, tie_weights, mha):
        super(Decoder, self).__init__()
        self.vocab = output_size
        self.hidden_size = hidden_dimension
        self.embedding_dim = embedding_dimension
        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=self.hidden_size)
        nn.init.xavier_uniform(self.embed.weight)
        self.n_layers = n_layers
        self.mha = mha
        self.lstm_cells = nn.ModuleList([MyLSTMCell(input_size=2 * self.hidden_size, hidden_size=self.hidden_size)])
        for j in range(n_layers - 1):
            self.lstm_cells.append(MyLSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size))

        # For attention
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        if mha == 1:
            self.v_k_q_1 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_2 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_3 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.v_k_q_4 = nn.ModuleList([nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4),
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 4)])
            self.linears = nn.ModuleList([self.v_k_q_1,
                                          self.v_k_q_2,
                                          self.v_k_q_3,
                                          self.v_k_q_4])
            self.multi_head_linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # For word projection
        self.projection_layer1 = nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size, bias=False)
        self.non_linear = nn.Tanh()
        self.projection_layer2 = nn.Linear(in_features=self.hidden_size, out_features=output_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)

        if tie_weights == 1:
            self.projection_layer2.weight = self.embed.weight

    def forward(self, keys, values, last_hiddens, label, label_len, input_len):
        embed = self.embed(label)
        mask = get_mask(input_len)
        output = []
        hidden_states = []
        for j in range(len(self.lstm_cells)):
            hidden_states.append((last_hiddens[0][j].contiguous(), last_hiddens[1][j].contiguous()))

        # Initial context
        context = self.get_context(self.lstm_cells[len(self.lstm_cells) - 1].h0.expand(embed.size(0), -1).contiguous(),
                                   keys, values, mask)
        for i in range(label.size(1) - 1):
            # Using teacher forcing for training
            # TODO: Do tf with exponential decay
            h = embed[:, i, :]
            h = torch.cat((h, context), dim=1)  # bs * 512
            for j, lstm in enumerate(self.lstm_cells):
                h_x_0, c_x_0 = hidden_states[j]
                hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]

            h = self.dropout(h)

            context = self.get_context(h, keys, values, mask)
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.non_linear(h)
            h = self.projection_layer2(self.dropout(h))
            # TODO: Is log softmax required ? Or can we do do with softmax only ?
            h = self.softmax(h)

            # Accumulating the output at each time-step
            output.append(h)

        output = torch.stack(output).permute(1, 0, 2)
        return output  # bs * max_tgt_seq_len - 1 * tgt_vocab_size

    def decode(self, keys, values, last_hiddens, max_decoding_time_step, beamSize=5):
        """
        :param keys:
        :param values:
        :param max_decoding_time_step
        :return: Returns the best decoded sentence, greedy decoder
        """
        bs = 1  # batch_size for decoding
        hidden_states = []
        for j in range(len(self.lstm_cells)):
            hidden_states.append((last_hiddens[0][j].contiguous(), last_hiddens[1][j].contiguous()))
        context_init = self.get_context(self.lstm_cells[len(self.lstm_cells) - 1].h0, keys, values)

        hypotheses_scores = to_variable(torch.zeros(1))
        hypotheses = [[(1, -1)]]  # Start token 1
        finished_hypotheses = []
        finished_hypotheses_scores = []
        i = 0
        while len(finished_hypotheses) < beamSize and i < max_decoding_time_step:
            num_of_hypotheses = len(hypotheses)
            # Expand the keys and values according to number of elements in current hypotheses list
            expanded_keys = keys.expand(num_of_hypotheses, keys.size(1), keys.size(2)).contiguous()
            expanded_values = values.expand(num_of_hypotheses, values.size(1), values.size(2)).contiguous()

            h_idx = to_variable(torch.LongTensor([hyp[-1][0] for hyp in hypotheses]))
            h = self.embed(h_idx)
            h = torch.cat((h, context_init), dim=1)
            for j, lstm in enumerate(self.lstm_cells):
                h_x_0, c_x_0 = hidden_states[j]
                hidden_states[j] = lstm(h, h_x_0, c_x_0)
                h = hidden_states[j][0]
            h = self.dropout(h)
            context, attn = self.get_context(h, expanded_keys, expanded_values, getAttn=True)
            h = torch.cat((h, context), dim=1)

            # At this point, h is the embed from the 2 lstm cells. Passing it through the projection layers
            h = self.projection_layer1(h)
            h = self.non_linear(h)
            h = self.projection_layer2(self.dropout(h))
            lsm = self.softmax(h)

            remaining_no_of_hypotheses = beamSize - len(finished_hypotheses)
            new_hypothesis_scores = (hypotheses_scores.unsqueeze(1).expand_as(lsm) + lsm).view(-1)
            # Flatten the log probabilities for all hypotheses
            top_new_hypothesis_scores, top_new_hypotheses_pos = torch.topk(new_hypothesis_scores,
                                                                           k=remaining_no_of_hypotheses)
            # Book keeping for tracking the word ids
            prev_hypotheses_ids = top_new_hypotheses_pos / self.vocab
            word_ids = top_new_hypotheses_pos % self.vocab

            new_hypotheses = []
            curr_hyp_ids = []
            new_hypothesis_scores = []
            for prev_hypothesis_id, word_id, new_hypothesis_score in zip(prev_hypotheses_ids.cpu().data,
                                                                         word_ids.cpu().data,
                                                                         top_new_hypothesis_scores.cpu().data):

                # Append the top k hypothesis to the existing list
                hyp_tgt_words = hypotheses[prev_hypothesis_id] + [(word_id, np.argmax(attn[prev_hypothesis_id][0].data.cpu().numpy()))]
                if word_id == 2:
                    finished_hypotheses.append(hyp_tgt_words)
                    finished_hypotheses_scores.append(new_hypothesis_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    curr_hyp_ids.append(prev_hypothesis_id)
                    new_hypothesis_scores.append(new_hypothesis_score)

            if len(finished_hypotheses) == beamSize:
                break

            curr_hyp_ids = to_tensor(np.array(curr_hyp_ids)).long()
            if torch.cuda.is_available():
                curr_hyp_ids = curr_hyp_ids.cuda()

            # Set the states for next loop
            for j in range(len(self.lstm_cells)):
                hidden_states[j] = (hidden_states[j][0][curr_hyp_ids], hidden_states[j][1][curr_hyp_ids])
            context_init = context[curr_hyp_ids]

            hypotheses_scores = to_variable(torch.FloatTensor(new_hypothesis_scores))
            hypotheses = new_hypotheses

            i += 1

        if len(finished_hypotheses) == 0:
            finished_hypotheses = [hypotheses[0]]
            finished_hypotheses_scores = [0.0]

        ranked_hypotheses = sorted(zip(finished_hypotheses, finished_hypotheses_scores), key=lambda x: x[1],
                                   reverse=True)
        return ranked_hypotheses

    def get_context(self, h, keys, values, mask=None, getAttn=False):
        query = self.linear(h)  # bs * hidden_dim, This is the query
        if self.mha == 1:
            head = None
            for linear in self.linears:
                n_keys = linear[0](keys)
                n_query = linear[1](query)
                n_values = linear[2](values)
                attn = torch.bmm(n_query.unsqueeze(1), n_keys.permute(0, 2, 1)) * (1.0 / np.sqrt(self.hidden_size // 4))
                attn = F.softmax(attn, dim=2)
                if mask is not None:
                    attn = attn * mask
                    attn = attn / attn.sum(2).unsqueeze(2)
                if head is None:
                    head = torch.bmm(attn, n_values).squeeze(1)
                else:
                    head = torch.cat((head, torch.bmm(attn, n_values).squeeze(1)), dim=1)
            context = self.multi_head_linear(head)
        else:
            attn = torch.bmm(query.unsqueeze(1), keys.permute(0, 2, 1))  # bs * 1 * seq_len
            attn = F.softmax(attn, dim=2)
            if mask is not None:
                attn = attn * mask
                attn = attn / attn.sum(2).unsqueeze(2)
            context = torch.bmm(attn, values).squeeze(1)  # bs * hidden_size
            if getAttn:
                return context, attn
        return context


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


def train(args: Dict[str, str], vocab):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    learning_rate = float(args['--lr'])

    pre_trained_embeddings = None
    if int(args['--use-pte']) > 0:
        # Hard-coding for 6-languages for now, can be modified later if required
        pre_trained_embeddings = get_pre_trained_embeddings(args['--emb-dir'], ['az', 'be', 'gl', 'pt', 'ru', 'tr'],
                                                            args['--save-emb-as'], int(args['--embed-size']),
                                                            vocab.src)

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                n_layers=int(args['--n_layers']),
                vocab=vocab,
                tie_weights=int(args['--tie-weights']),
                mha=int(args['--mha']),
                pte=pre_trained_embeddings)

    model.apply(init_xavier)
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Begin Maximum Likelihood training:')

    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    # TODO: Add weight decay, momentum later
    if int(args['--optim']) == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    best_model_file = ''
    
    # Load weights from saved model for finetuning
    if args['--load-weights-from'] is not None:
        model.load(args['--load-weights-from'])
    
    for epoch in tqdm(range(1, int(args['--max-epoch']) + 1)):
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            batch_size = len(src_sents)
            input_lens = [len(x) for x in src_sents]
            output_lens = [len(x) for x in tgt_sents]
            src_batch = get_batch_tensor(src_sents, input_lens[0], vocab.src)
            tgt_batch = get_batch_tensor(tgt_sents, max(output_lens), vocab.tgt)

            model.train()
            optimizer.zero_grad()  # Reset the gradients

            decoded_scores = model(src_batch, tgt_batch, input_lens, output_lens)

            label_mask = get_mask(output_lens).squeeze(1)[:, 1:].contiguous()

            # loss = loss_fn(decoded_scores, label)
            loss = loss_fn(decoded_scores.contiguous().view(-1, len(vocab.tgt)),
                           tgt_batch[:, 1:].contiguous().view(-1))

            loss = (loss.view(label_mask.size()) * label_mask).sum(1).sum()
            loss_val = loss.data.cpu().numpy()
            loss = loss / batch_size
            loss.backward()  # Back propagate the gradients

            if clip_grad > 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)
            optimizer.step()  # Update the network

            report_loss += loss_val
            cum_loss += loss_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                tqdm.write('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                           'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                              report_loss / report_examples,
                                                                                              math.exp(
                                                                                                  report_loss / report_tgt_words),
                                                                                              cumulative_examples,
                                                                                              report_tgt_words / (
                                                                                              time.time() - train_time),
                                                                                              time.time() - begin_time),
                           file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                tqdm.write('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                                  cum_loss / cumulative_examples,
                                                                                                  np.exp(
                                                                                                      cum_loss / cumulative_tgt_words),
                                                                                                  cumulative_examples),
                           file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                tqdm.write('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, loss_fn, batch_size=32)  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                tqdm.write('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    tqdm.write('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    # TODO: Make save model more informative
                    best_model_file = "model_epoch_{}_ppl_{:.4f}.t7".format(epoch, float(dev_ppl))
                    model.save(model_save_path + best_model_file)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    tqdm.write('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        tqdm.write('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            tqdm.write('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        tqdm.write('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                        optimizer.param_groups[0]['lr'] = lr

                        # load model
                        model.load(model_save_path + best_model_file)

                        tqdm.write('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[
    List[Hypothesis]]:
    hypotheses = []
    decoded_file_with_scores = open("work_dir/decode_with_scores.txt", 'w')
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
        hypotheses.append(example_hyps)
        decoded_file_with_scores.write("{} -> {}\n".format(' '.join(example_hyps[0].value), example_hyps[0].score))
    decoded_file_with_scores.close()
    return hypotheses


def decode(args: Dict[str, str], vocab):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                n_layers=int(args['--n_layers']),
                vocab=vocab,
                tie_weights=int(args['--tie-weights']),
                mha=int(args['--mha']))
    if torch.cuda.is_available():
        model = model.cuda()
    model.load(args['MODEL_PATH'])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    vocab = pickle.load(open(args['--vocab'], 'rb'))
    if args['train']:
        train(args, vocab)
    elif args['decode']:
        decode(args, vocab)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
