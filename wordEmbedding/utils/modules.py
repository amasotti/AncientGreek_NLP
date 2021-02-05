"""
NN Model taken from [n0obcoder](https://github.com/n0obcoder/Skip-Gram_Model-TensorFlow)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddings,device='cpu',noise_dist=None,negs=15):
        super(CBOW, self).__init__()

        self.vocab_size = vocab_size
        self.negs = negs
        self.device = device
        self.noise_dist = noise_dist

        self.embeddings_target = nn.Embedding(vocab_size, embeddings,padding_idx=0)
        self.embeddings_context = nn.Embedding(vocab_size, embeddings,padding_idx=0)

        self.embeddings_target.weight.data.uniform_(-1,1)
        self.embeddings_context.weight.data.uniform_(-1, 1)

    def forward(self,target, context,debug=False):
        # FIXME: Check if everything is implemented correctly
        # or if we need a softmax here

        # computing out loss
        emb_input = self.embeddings_target(target)  # bs, emb_dim
        emb_context = self.embeddings_context(context)  # bs, emb_dim

        emb_product = torch.mul(emb_input, emb_context)  # bs, emb_dim
        emb_product = torch.sum(emb_product, dim=1)  # bs

        out_loss = F.logsigmoid(emb_product)  # bs

        if self.negs > 0:
            # computing negative loss
            if self.noise_dist is None:
                self.noise_dist = torch.ones(self.vocab_size)

            num_neg_samples_for_this_batch = context.shape[0] * self.negs
            # coz bs*num_neg_samples > vocab_size
            negative_example = torch.multinomial(
                self.noise_dist, num_neg_samples_for_this_batch, replacement=True)



            negative_example = negative_example.view(context.shape[0], self.negs).to(self.device)  # bs, num_neg_samples
            emb_negative = self.embeddings_context(negative_example)  # bs, neg_samples, emb_dim
            emb_product_neg_samples = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2))  # bs, neg_samples, 1

            noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1)  # bs


            total_loss = -(out_loss + noise_loss).mean()

            return total_loss

        else:
            return -(out_loss).mean()
