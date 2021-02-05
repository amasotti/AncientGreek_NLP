from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

def skip_gram_dataset(corpus, word2index, window):
    """
    Given a corpus, a window_size and a dictionary with mappings word : index, it returns
    a long list of lists that can be used to train the Skip Gram version of the
    Word2Vec model
    """
    dataset = []
    for sentence in tqdm(corpus, desc="Sententence in Corpus"):
        # take each word as target separately
        for center_word in range(len(sentence)):
            # loop in the window and be careful to not jump out of the boundaries :)
            for j in range(max(center_word - window, 0), min(center_word + window, len(sentence))):
                # jump the center word
                if j != center_word:
                    # append the context words in tuples
                    dataset.append([word2index[sentence[center_word]], word2index[sentence[j]]])
    return dataset

class trainDataset_old(object):
    '''

    Convenient class to support batch generation in the training phase

    '''
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.long)

    def __getitem__(self, index):
        x = self.data[index, 0]
        y = self.data[index, 1]
        return x,y

    def __len__(self):
        return len(self.data)

class trainDataset(Dataset):
    '''

    Convenient class to support batch generation in the training phase

    '''
    def __init__(self, data, train_size=0.7, val_size=0.15):
        super(Dataset, self).__init__()
        #self.data = torch.tensor(data, dtype=torch.long)
        self.data_size = len(data)
        self._target_df = None
        self._target_size = 0

        # calculate the split size
        self.train_size = int(self.data_size * train_size)
        self.val_size = int(self.data_size * val_size)
        self.test_size = self.val_size

        # split the data
        self.train_set, self.val_set, self.test_set = self.split_data(data=data,
                                                                      train_size=self.train_size,
                                                                      val_size=self.val_size)
        self.lookup = {
            'train' : (self.train_set, self.train_size),
            'val': (self.val_set, self.val_size),
            'test': (self.test_set, self.test_size)}

        # Set training subset as target when initializing the Dataset
        self.set_split()


    def split_data(self, data, train_size, val_size):
        """
        Called only once at the beginning, given the data list returns three splitted sets
        for the three phases: training, validation and test

        """

        train_set = data[:train_size]
        train_set = torch.tensor(train_set, dtype=torch.long)

        val_set = data[train_size:train_size+val_size]
        val_set = torch.tensor(val_set, dtype=torch.long)

        test_set = data[train_size+val_size:]
        test_set = torch.tensor(test_set, dtype=torch.long)

        return train_set, val_set, test_set

    def set_split(self, split="train"):
        """
        Switch between subsets

        """
        self._target_split = split
        self._target_df, self._target_size = self.lookup[split]


    def __getitem__(self, index):

        x = self._target_df[index, 0]
        y = self._target_df[index, 1]
        return x,y

    def __len__(self):
        return self._target_size


def make_batch(dataset, batch_size, shuffle, drop_last, device):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        shuffle=shuffle
                        )

    for x, y in loader:
        yield x.to(device), y.to(device),
