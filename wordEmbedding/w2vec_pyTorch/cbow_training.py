from modules import *
from utils import *
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace


params = Namespace(
    corpus = "../../data/Homer_tokenized_corpus.npy",
    w2i = "../../data/vocabularies/Homer_word2index.json",
    embeddings = 100,
    lr = 0.02,
    epochs = 5,
    window = 8,
    hidden = 128,
    cuda = True,
    device = "cpu"
)

if not torch.cuda.is_available():
    params.cuda = False

params.device = torch.device("cuda" if params.cuda else "cpu")

print("Using CUDA: {}".format(params.cuda))


# load the corpus (see preprocessing)
corpus = np.load(params.corpus,allow_pickle=True)
corpus = corpus.tolist()

# Load dictionary
with open(params.w2i, 'r', encoding='utf-8') as fp:
    w2i = json.load(fp)

# Create ngrams
ngrams = ngram_builder(window=params.window, corpus = corpus)

model = CBOW(vocab_size = 30931,
             embeddings=params.embeddings,
             window=params.window,
             hidden=params.hidden)

# move on the GPU if possible
model.to(params.device)

losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=params.lr)

for epoch in range(params.epochs):
    print("Starting the training")
    total_loss = 0

    # Use the ngrams and transform vectors into tensors
    for context, target in ngrams:
        context_indices = torch.tensor([w2i[w] for w in context], dtype=torch.long)
        # set the gradients to zero
        model.zero_grad()

        # call the forward method
        if params.device == 'cuda':
            context_indices.to(params.device)
            target = target.to(params.device)
        logs = model(context_indices)

        # calculate the loss
        loss = loss_function(logs, torch.tensor([w2i[target]], dtype=torch.long).to(params.device))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch} --  Total loss {total_loss}")
    losses.append(total_loss)


torch.save(model.state_dict(), "../../data/models/w2vec_pytorch")