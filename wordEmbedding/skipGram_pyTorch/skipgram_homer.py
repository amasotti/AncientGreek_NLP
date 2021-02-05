import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from wordEmbedding.utils.dataset import trainDataset, make_batch
from wordEmbedding.utils.modules import CBOW
from wordEmbedding.utils.utils import print_test, save_model

# popular words
paths = Namespace(
    corpus="../../data/Homer_tokenized_corpus.npy",
    w2i="../../data/vocabularies/Homer_word2index.json",
    skipDataset="../../data/vocabularies/Homer_skipgram_dataset.npy",
    vocab='../../data/vocabularies/Homer_word_frequencies.json',
    model='../../data/models/Skipgram_Pytorch_0502_beta.pth'

)
# -------------------------------------------------------------------------
#                   LOADING RAW DATA AND LOOK-UP TABLES
# -------------------------------------------------------------------------
# Load tokenized corpus (see preprocessing.py in utils)
corpus = np.load(paths.corpus, allow_pickle=True)
corpus = corpus.tolist()

# load the vocabulary (with subsampling)
with open(paths.vocab, "r", encoding="utf-8") as fp:
    vocab = json.load(fp)

# Load the Word2Index dictionary (see preprocessing.py in utils))
with open(paths.w2i, "r", encoding="utf-8") as fp:
    w2i = json.load(fp)

# Create a reverse lookup table
index2word = {i: w for w, i in w2i.items()}

# extract the dataset for training
# skipDataset = skip_gram_dataset(corpus=corpus, word2index=w2i, window=7)
# np.save(args.skipDataset, skipDataset,allow_pickle=True)

# Load tokenized corpus (see preprocessing.py in utils)
skipDataset = np.load(paths.skipDataset, allow_pickle=True)
skipDataset = skipDataset.tolist()

# -------------------------------------------------------------------------
#               SETTINGS FOR THE NEURAL MODEL
# -------------------------------------------------------------------------

TEST_WORDS = ['μηνιν', "εθηκε", "ερχομαι", "θεα", "βροτον", "ευχομαι", "ερος", "φρασαι", "εφατʼ"]

params = Namespace(
    train_size=0.7,
    val_size=0.15,
    drop_last=True,
    batch=1024 * 2,
    epochs=20,
    lr=0.002,
    device='cpu',
    cuda=False,
    embeddings=100,
    show_stats_after=50,  # after how many batches should the bars be updated
)

if not torch.cuda.is_available():
    params.cuda = False
    params.device = "cpu"
else:
    params.cuda = True
    params.device = 'cuda'

print(f"Using GPU ({params.device}) : {params.cuda}")

# Make Torch Dataset from list (splits data and transforms them into tensors)
Dataset = trainDataset(skipDataset, train_size=params.train_size, val_size=params.val_size)

# make noise distribution to sample negative examples from #FIXME: Probably we would need to delete this or adjust to splitted sets
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs / sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

model = CBOW(vocab_size=len(vocab),
             embeddings=params.embeddings,
             device=params.device,
             noise_dist=None,  # TODO: See later if this works
             negs=15)

# Load model
ckpt = torch.load(os.path.join(paths.model))
model.load_state_dict(ckpt['model_state_dict'])

# move to cuda if available
model.to(params.device)

print('\nMODEL SETTINGS:')
print(model)

losses_train = [0]
losses_val = [0]
optimizer = optim.Adam(model.parameters(), lr=params.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode="min",
                                                 factor=0.3, patience=1)
# Set bars

epoch_bar = tqdm(desc="Epochs Routine", total=params.epochs, position=0, leave=True)
train_bar = tqdm(desc="Training phase", total=Dataset.train_size / params.batch, position=1, leave=True)
val_bar = tqdm(desc="Validataion phase", total=Dataset.val_size / params.batch, position=1, leave=True)

## AND ..... GO .....
for epoch in tqdm(range(params.epochs)):
    print(f'\n===== EPOCH {epoch + 1}/{params.epochs} =====')

    # Load specific splitted dataset
    Dataset.set_split('train')
    model.train()
    print(f'DATASET SUBSET LOADED : {Dataset._target_split} with size : {len(Dataset)}')
    print('Whole Dataset size: ', Dataset.data_size)
    print('Size of the vocabulary: ', len(vocab), '\n\n')

    Loader = make_batch(dataset=Dataset,
                        device=params.device,
                        batch_size=params.batch,
                        shuffle=False,
                        drop_last=params.drop_last)
    loss = 0  # reset the actual loss

    # Batch for the training phase
    for batch_idx, (inp, target) in enumerate(Loader):
        # Training modus (the test with the small word list requires setting the mode to eval)
        model.train()

        # reset gradients
        optimizer.zero_grad()
        loss = model(inp, target)

        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        # I want to know what are you doing...
        if batch_idx % params.show_stats_after == 0:
            # update bar
            train_bar.set_postfix(loss=loss.item(), epoch=epoch)
            train_bar.update(n=params.show_stats_after)
            # Run a small test
            print_test(model, TEST_WORDS, w2i, index2word, epoch=epoch)

    save_model(model=model, epoch=epoch, losses=losses_train, fp=paths.model)

    # Load specific splitted dataset
    Dataset.set_split('val')
    print(f'DATASET SUBSET LOADED : {Dataset._target_split} with size : {len(Dataset)}')
    print('Whole Dataset size: ', Dataset.data_size)
    print('Size of the vocabulary: ', len(vocab), '\n\n')

    Loader = make_batch(dataset=Dataset,
                        device=params.device,
                        batch_size=params.batch,
                        shuffle=False,
                        # FIXME: The problem is that shuffling takes also items from other splitted sets, resulting in a out of bound Error, but I'd like somehow to shuffle the data
                        drop_last=params.drop_last)

    # Evaluation / Validation mode
    model.eval()
    loss = 0
    for batch_idx, (inp, target) in enumerate(Loader):

        loss = model(inp, target)
        losses_val.append(loss.item())
        scheduler.step(losses_val[-1])

        # I want to know what are you doing...
        if batch_idx % params.show_stats_after == 0:
            # update bar
            val_bar.set_postfix(loss=loss.item(), epoch=epoch)
            val_bar.update(n=params.show_stats_after)
            # Run a small test:
            print_test(model, TEST_WORDS, w2i, index2word, epoch=epoch)

    epoch_bar.update()

plt.figure(figsize=(100, 100))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch -- Training")

plt.plot(losses_train)
plt.savefig('losses_train.png')
plt.show()

plt.figure(figsize=(100, 100))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch -- Validation")

plt.plot(losses_val)
plt.savefig('losses_val.png')
plt.show()
