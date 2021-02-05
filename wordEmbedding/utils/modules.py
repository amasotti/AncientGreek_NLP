import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddings, window, hidden):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings)
        self.lin1 = nn.Linear(window * embeddings, hidden)
        self.lin2 = nn.Linear(hidden, vocab_size)

    def forward(self,x):
        x = x.to('cuda')
        x = self.embeddings(x).view((1,-1))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)
        return x.to('cuda')

    def predict(self,input,w2i):
        context = torch.tensor([w2i[w] for w in input],dtype=torch.long)
        res = self.forward(context)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        print(f"res_val : {res_val}")
        print(f"res_ind : {res_ind}")
        for arg in zip(res_val, res_ind):
            print([(key, val, res_arg[0]) for key, val in w2i.items() if val == res_arg[1]])

    def freeze_layer(self,model,layer):
        for name, child in model.name_children():
            print(name, child)
            if name == layer:
                for names, params in child.named_parameters():
                    print(names, params)
                    params.require_grad = False

    def write_embedding_to_file(self, fp):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(fp, weights)

