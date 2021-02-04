from tqdm import tqdm


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

