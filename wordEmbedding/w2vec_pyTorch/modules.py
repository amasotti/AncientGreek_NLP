import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddings, context_size, hidden):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings)
        self.lin1 = nn.Linear(2*context_size*embeddings, hidden)
        self.lin2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embeddings(x).view((1, -1))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        log_probs = F.log_softmax(x)
        return x

