import math
from typing import List
import torch
import numpy as np
import os


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


# Helper functions for tensor operations

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor, requires_grad=False):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


def get_batch_tensor(list_of_sentences, max_len, vocab):
    batch = np.zeros((len(list_of_sentences), max_len))
    for i, sentence in enumerate(list_of_sentences):
        batch[i, : len(sentence)] = [vocab[x] for x in sentence]
    return to_variable(to_tensor(np.array(batch)).long())


def get_mask(input_len):
    # Create mask
    a = torch.arange(max(input_len)).unsqueeze(0).expand(len(input_len), -1)
    b = to_tensor(np.array(input_len)).unsqueeze(1)
    mask = a < b
    return to_variable(mask.unsqueeze(1).type(torch.FloatTensor))


def get_pre_trained_embeddings(data_dir, files, save_file_as, embedding_dim, src_vocab):
    if os.path.exists(data_dir + save_file_as + '.npy'):
        return np.load(data_dir + save_file_as + '.npy')
    print("Reading pre-trained embeddings...")
    # Read the top_frequent word vectors in a dictionary
    embeddings = np.random.uniform(-0.25, 0.25, (len(src_vocab), embedding_dim))
    for file in files:
        count = 0
        with open(data_dir + "wiki.{}.vec".format(file), 'r', encoding='utf-8') as f:
            ignore_first_row = True
            for row in f.readlines():
                if ignore_first_row:
                    ignore_first_row = False
                    continue
                split_row = row.split(" ")
                vec = np.array(split_row[1:-1]).astype(np.float)
                if split_row[0] in src_vocab.word2id and len(vec) == embedding_dim:
                    embeddings[src_vocab.word2id[split_row[0]]] = vec
                    count += 1
    np.save(data_dir + save_file_as + '.npy', embeddings)
    print("Successfully loaded {} embeddings out of {}".format(count, len(src_vocab)))
    return np.array(embeddings)
