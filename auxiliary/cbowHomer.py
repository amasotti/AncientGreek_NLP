# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'

from abc import ABC

'''
Mein Versuch, ein CBOW for die Homerische Media zu erstellen
nach dem Muster von Brian & MacMahan - NLP with Pytorch

This file doesn't contain the actual training loop, just the classes and functions needed to build
a trainable dataset and the neural model itself.

**DISCLAIMER**: See the folder wordEmbeddings for a newer version!

'''
# imports
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Vocab(object):
    '''
    DOC

    This class extracts a vocabulary out of a list of tokens or a single token

    '''

    def __init__(self, word2index=None, boundary_mark="<MASK>", add_unknown=True, unknown="<UNKNOWN>"):

        if word2index is None:
            word2index = dict()
        self.word2index = word2index
        self.index2word = {idx: word for word, idx in self.word2index.items()}

        self.boundary = boundary_mark
        self.boundary_idx = self.add_token(self.boundary)

        self.add_unknown = add_unknown
        self.unknown = unknown
        self.unknown_idx = None

        if self.add_unknown:
            self.unknown_idx = self.add_token(self.unknown)

    def save(self):
        return {"word2index": self.word2index,
                'boundary_mark': self.boundary,
                'add_unknown': self.add_unknown,
                'unknown': self.unknown}

    @classmethod
    def load(cls, vocabulary):
        return cls(**vocabulary)

    def add_token(self, token):
        """
        Main function of this class : adds words to a dictionary and assigns them an index

        :param token: a string
        :return: the index in the vocabulary for that token.
        """
        if token in self.word2index.keys():
            idx = self.word2index[token]
        else:
            idx = len(self.word2index)
            self.word2index[token] = idx
            self.index2word[idx] = token
        return idx

    def add_batch(self, token_list):
        return [self.add_token(_) for _ in token_list]

    def lookup_token(self, token):
        try:
            if self.unknown_idx is not None:
                return self.word2index.get(token, self.unknown_idx)
            else:
                return self.word2index[token]
        except KeyError:
            print('Token not in the dictionary and UNKNOWN not implemented')

    def lookup_index(self, index):
        if index not in self.index2word:
            raise KeyError('This index was not yet assigned')
        else:
            return self.index2word[index]

    def __len__(self):
        return len(self.word2index)


class Vectorizer(object):
    def __init__(self, vocabulary):

        self.vocab = vocabulary

    def vectorize(self, context, length=-1):
        """

        Main function for the Vectorizer

        :param context: a string of words in the context of the target word
        :param length: the length of the vector, initialised as -1 as sentinel
        :return: a numpy vector with the indices of the words in the context of the target

        """

        indices = [self.vocab.lookup_token(token)
                   for token in context.split(' ')]
        if length < 0:
            length = len(indices)

        vector = np.zeros(length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = self.vocab.boundary_idx

        return vector

    @classmethod
    def from_df(cls, df):
        """

        Read from csv (pandas Dataframe already saved)
        the csv has 3 cols : context, word, split_subset
        :param df: the df
        :return: an instance of this class
        """

        vocab = Vocab()
        for idx, obs in df.iterrows():
            for word in obs.context.split(' '):
                vocab.add_token(word)
            vocab.add_token(obs.word)
        return cls(vocabulary=vocab)

    @classmethod
    def from_json(cls, dump):
        vocab = Vocab.load(dump['vocabulary'])
        return cls(vocabulary=vocab)

    def to_json(self):
        return {'vocabulary': self.vocab.save()}


class CBOWDataset(Dataset):
    """
    Class organizing the flow of the NN

    """

    def __init__(self, df, vectorizer):
        self.cbow_df = df
        self.vectorizer = vectorizer

        def find_length(context): return len(context.split(' '))
        # go through the different contexts and take the longest one

        self.max_length = max(map(find_length, self.cbow_df.context))

        self.train_df = self.cbow_df[self.cbow_df.split_subset == "train"]
        self.train_size = len(self.train_df)

        self.validation_df = self.cbow_df[self.cbow_df.split_subset == "validation"]
        self.validation_size = len(self.validation_df)

        self.test_df = self.cbow_df[self.cbow_df.split_subset == "test"]
        self.test_size = len(self.test_df)

        self.subset_dict = {'train': (self.train_df, self.train_size),
                            'test': (self.test_df, self.test_size),
                            'validation': (self.validation_df, self.validation_size)}

    @classmethod
    def load_and_create(cls, csv_file):
        df = pd.read_csv(csv_file)
        train_df = df[df.split_subset == 'train']
        print(f"Vocabulary created inside Vector")
        return cls(df, Vectorizer.from_df(train_df))

    @classmethod
    def load_all(cls, csv_file, vectorizer_path):
        df = pd.read_csv(csv_file)
        vectorizer = cls.load_vectorizer(vectorizer_path)
        return cls(df, vectorizer)

    @staticmethod
    def load_vectorizer(vectorizer_path):
        with open(vectorizer_path, encoding='utf-8') as fp:
            vectorizer = Vectorizer.from_json(json.load(fp))
        return vectorizer

    def save_vectorizer(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(self.vectorizer.to_json(), fp)

    def get_vectorizer(self):
        return self.vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self.subset_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, item):
        row = self._target_df.iloc[item]
        context = self.vectorizer.vectorize(row.context, self.max_length)
        target_index = self.vectorizer.vocab.lookup_token(row.word)
        """print("I'm in the Dataset, I'll print the target df")
        print(self._target_df.head(2))
        print("Context")
        print(row.context)
        print("Target")
        print(row.word)
        print("Index of target", str(target_index))"""

        return {'x_data': context,
                'y_target': target_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def batch_generator(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data in dataloader:
        out_data = {}
        for name, tensor in data.items():
            out_data[name] = data[name].to(device)
        yield out_data


class CBOWNetzwerk(nn.Module):
    """
    PyTorch neural network

    """

    def __init__(self, vocab_size, embedding_size, padding=0):
        super(CBOWNetzwerk, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding)
        self.fc1 = nn.Linear(in_features=embedding_size,
                             out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=150)
        self.fc3 = nn.Linear(in_features=150,
                             out_features=vocab_size)

    def forward(self, x, softmax=True):
        x = F.dropout(self.embedding(x).sum(dim=1), 0.3)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(x)

        if softmax:
            x = F.log_softmax(x, dim=1)
        return x

# ########## TRAINING ROUTINE ####################


def make_train_state(args):
    return {'stop_early': True,
            'early_stopping_step': 2,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}


def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def compute_accuracy(y_pred, y_true):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_true).sum().item()
    return n_correct / len(y_pred_indices) * 100


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
