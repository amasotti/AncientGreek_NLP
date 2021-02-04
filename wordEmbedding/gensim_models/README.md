# WordEmbedding models 

In this folder I've tried to build and train two Embedding Models for the homeric texts:

+ Word2Vec (Skipgram Negative)
+ FastText

From the Gensim [FastText documentation](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#sphx-glr-auto-examples-tutorials-run-fasttext-py)

<quote>
"According to a detailed comparison of Word2Vec and fastText, fastText does significantly better on syntactic tasks as compared to the original Word2Vec, especially when the size of the training corpus is small. Word2Vec slightly outperforms fastText on semantic tasks though. The differences grow smaller as the size of the training corpus increases."
</quote>

The structure of both notebooks is similar:
    + Data loading
    + Preprocessing
    + Model initialization
    + Training
    + Few explorations
    + Data visualization (semantic spaces) with the Bokeh library
    
## Parameters

### Word2Vec
The hyperparameters for the `Word2Vec` mdodel were set following the suggestions of Levy, Goldberg, Dagan (2015) - [Improving Distributional Similarity
with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016/)

**Model settings**

    
    
### TSNE (Dimensionality reduction)

See the [Official docs](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?highlight=tsne#sklearn.manifold.TSNE)

+ `n_components = 2` : reduce to 2d
+ `perplexity = 30` (default)
+ `metric='cosine'` : since we're dealing with vectors, I though cosine similarity to be a better choice than 'euclidean'
+ `init='pca'` : initialization of the embeddings. PCA tends to be more globally stable than "random init"
